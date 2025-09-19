#!/usr/bin/env python3
"""
Script to use Marker PDF with AWS Bedrock LLM models
Configurable via environment variables or .env file
"""

import os
import sys
from pathlib import Path
from typing import Optional, List
import argparse
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add marker to path
sys.path.insert(0, str(Path(__file__).parent))

from marker.converters import PdfConverter
from marker.config import BasePDFConverterConfig
from marker.services.bedrock import BedrockService
from marker.output import save_output
from marker.settings import settings
from marker.logger import get_logger

logger = get_logger()


class BedrockMarkerConfig:
    """Configuration for Marker with Bedrock"""

    def __init__(self):
        # AWS Bedrock Configuration from environment
        self.aws_region = os.getenv("AWS_REGION", "us-east-1")
        self.aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
        self.aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        self.aws_session_token = os.getenv("AWS_SESSION_TOKEN")
        self.bedrock_model_id = os.getenv(
            "BEDROCK_MODEL_ID",
            "anthropic.claude-3-5-sonnet-20241022-v2:0"
        )

        # Marker configuration
        self.output_dir = os.getenv("OUTPUT_DIR", "./outputs")
        self.debug = os.getenv("DEBUG", "false").lower() == "true"
        self.force_ocr = os.getenv("FORCE_OCR", "false").lower() == "true"
        self.output_format = os.getenv("OUTPUT_FORMAT", "markdown")
        self.max_pages = int(os.getenv("MAX_PAGES", "0")) or None
        self.workers = int(os.getenv("NUM_WORKERS", "1"))
        self.max_bedrock_tokens = int(os.getenv("MAX_BEDROCK_TOKENS", "8192"))
        self.temperature = float(os.getenv("TEMPERATURE", "0.0"))

    def get_bedrock_service(self):
        """Create Bedrock service instance"""
        config = {
            "bedrock_model_id": self.bedrock_model_id,
            "aws_region": self.aws_region,
            "aws_access_key_id": self.aws_access_key_id,
            "aws_secret_access_key": self.aws_secret_access_key,
            "aws_session_token": self.aws_session_token,
            "max_bedrock_tokens": self.max_bedrock_tokens,
            "temperature": self.temperature,
        }
        return BedrockService(config)


def convert_single_pdf(
    pdf_path: Path,
    config: BedrockMarkerConfig,
    use_llm: bool = True
) -> dict:
    """Convert a single PDF file using Bedrock"""
    logger.info(f"Converting {pdf_path}...")

    # Create converter configuration
    converter_config = BasePDFConverterConfig(
        use_llm=use_llm,
        force_ocr=config.force_ocr,
        output_format=config.output_format,
        debug=config.debug,
        max_pages=config.max_pages,
    )

    # Initialize converter with Bedrock service if using LLM
    if use_llm:
        llm_service = config.get_bedrock_service()
        converter = PdfConverter(converter_config)
        converter.llm_service = llm_service
        logger.info(f"Using Bedrock model: {config.bedrock_model_id}")
    else:
        converter = PdfConverter(converter_config)
        logger.info("Converting without LLM")

    # Convert PDF
    try:
        with open(pdf_path, "rb") as f:
            result = converter(f)

        # Save output
        output_path = Path(config.output_dir) / pdf_path.stem
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save based on format
        if config.output_format == "markdown":
            output_file = output_path.with_suffix(".md")
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(result.markdown)
            logger.info(f"Saved markdown to {output_file}")

        elif config.output_format == "json":
            output_file = output_path.with_suffix(".json")
            import json
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
            logger.info(f"Saved JSON to {output_file}")

        elif config.output_format == "html":
            output_file = output_path.with_suffix(".html")
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(result.html)
            logger.info(f"Saved HTML to {output_file}")

        return {
            "success": True,
            "output_path": str(output_file),
            "pages": len(result.pages) if hasattr(result, 'pages') else 0
        }

    except Exception as e:
        logger.error(f"Error converting {pdf_path}: {e}")
        return {
            "success": False,
            "error": str(e),
            "pdf_path": str(pdf_path)
        }


def convert_batch(
    input_dir: Path,
    config: BedrockMarkerConfig,
    use_llm: bool = True,
    pattern: str = "*.pdf"
):
    """Convert multiple PDFs in a directory"""
    pdf_files = list(input_dir.glob(pattern))

    if not pdf_files:
        logger.warning(f"No PDF files found in {input_dir}")
        return

    logger.info(f"Found {len(pdf_files)} PDF files to convert")

    results = []
    for pdf_path in pdf_files:
        result = convert_single_pdf(pdf_path, config, use_llm)
        results.append(result)

    # Summary
    successful = sum(1 for r in results if r.get("success"))
    logger.info(f"\nConversion complete: {successful}/{len(results)} successful")

    if successful < len(results):
        logger.error("Failed conversions:")
        for r in results:
            if not r.get("success"):
                logger.error(f"  - {r.get('pdf_path')}: {r.get('error')}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert PDFs using Marker with AWS Bedrock LLM"
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Input PDF file or directory containing PDFs"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: ./outputs or from .env)"
    )
    parser.add_argument(
        "--use-llm",
        action="store_true",
        default=True,
        help="Use Bedrock LLM for enhanced conversion (default: True)"
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Disable LLM usage"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Bedrock model ID (overrides .env setting)"
    )
    parser.add_argument(
        "--format",
        choices=["markdown", "json", "html"],
        default=None,
        help="Output format (default: markdown or from .env)"
    )
    parser.add_argument(
        "--force-ocr",
        action="store_true",
        help="Force OCR on all pages"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with detailed output"
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        help="Maximum number of pages to process"
    )

    args = parser.parse_args()

    # Load configuration
    config = BedrockMarkerConfig()

    # Override with command line arguments
    if args.output_dir:
        config.output_dir = str(args.output_dir)
    if args.model:
        config.bedrock_model_id = args.model
    if args.format:
        config.output_format = args.format
    if args.force_ocr:
        config.force_ocr = True
    if args.debug:
        config.debug = True
    if args.max_pages:
        config.max_pages = args.max_pages

    use_llm = not args.no_llm

    # Print configuration
    print("=" * 50)
    print("Marker PDF with AWS Bedrock")
    print("=" * 50)
    print(f"Model: {config.bedrock_model_id if use_llm else 'Disabled'}")
    print(f"Region: {config.aws_region}")
    print(f"Output format: {config.output_format}")
    print(f"Output directory: {config.output_dir}")
    print(f"Debug mode: {config.debug}")
    print(f"Force OCR: {config.force_ocr}")
    print("=" * 50)
    print()

    # Process input
    if args.input.is_file() and args.input.suffix.lower() == ".pdf":
        # Single file
        result = convert_single_pdf(args.input, config, use_llm)
        if result.get("success"):
            print(f"✓ Conversion successful: {result.get('output_path')}")
        else:
            print(f"✗ Conversion failed: {result.get('error')}")
    elif args.input.is_dir():
        # Directory of files
        convert_batch(args.input, config, use_llm)
    else:
        print(f"Error: {args.input} is not a valid PDF file or directory")
        sys.exit(1)


if __name__ == "__main__":
    main()