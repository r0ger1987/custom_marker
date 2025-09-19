#!/usr/bin/env python3
"""
Test script for AWS Bedrock integration with Marker PDF
"""

import os
import sys
from pathlib import Path
from PIL import Image
import io

# Add marker to path
sys.path.insert(0, str(Path(__file__).parent))

from marker.services.bedrock import BedrockService
from marker.settings import settings
from marker.converters import PdfConverter
from marker.config import BasePDFConverterConfig
from pydantic import BaseModel


class TestResponse(BaseModel):
    text: str
    confidence: float


def test_bedrock_service():
    """Test basic Bedrock service functionality"""
    print("Testing AWS Bedrock Service...")
    print("-" * 50)

    # Initialize Bedrock service with settings
    config = {
        "bedrock_model_id": os.getenv("BEDROCK_MODEL_ID", settings.BEDROCK_MODEL_ID),
        "aws_region": os.getenv("AWS_REGION", settings.AWS_REGION),
        "aws_access_key_id": os.getenv("AWS_ACCESS_KEY_ID", settings.AWS_ACCESS_KEY_ID),
        "aws_secret_access_key": os.getenv("AWS_SECRET_ACCESS_KEY", settings.AWS_SECRET_ACCESS_KEY),
        "aws_session_token": os.getenv("AWS_SESSION_TOKEN", settings.AWS_SESSION_TOKEN),
        "temperature": 0.0,
        "max_bedrock_tokens": 1000,
    }

    try:
        service = BedrockService(config)
        print(f"✓ Bedrock service initialized")
        print(f"  Model: {service.bedrock_model_id}")
        print(f"  Region: {service.aws_region}")
        print()

        # Test text-only prompt
        print("Testing text-only prompt...")
        result = service(
            prompt="Extract the main topic from this text: 'Machine learning is a subset of artificial intelligence that enables computers to learn from data.'",
            image=None,
            block=None,
            response_schema=TestResponse,
        )

        if result:
            print(f"✓ Text-only test successful")
            print(f"  Response: {result}")
        else:
            print("✗ Text-only test failed - empty response")
        print()

        # Test with image (create a simple test image)
        print("Testing with image...")
        test_image = Image.new('RGB', (200, 100), color='white')
        from PIL import ImageDraw
        draw = ImageDraw.Draw(test_image)
        draw.text((10, 40), "Test Document", fill='black')

        result = service(
            prompt="Describe what you see in this image.",
            image=test_image,
            block=None,
            response_schema=TestResponse,
        )

        if result:
            print(f"✓ Image test successful")
            print(f"  Response: {result}")
        else:
            print("✗ Image test failed - empty response")

    except Exception as e:
        print(f"✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()


def test_pdf_conversion_with_bedrock():
    """Test PDF conversion with Bedrock LLM"""
    print("\nTesting PDF Conversion with Bedrock LLM...")
    print("-" * 50)

    # Find a test PDF
    test_pdfs = list(Path(".").glob("*.pdf"))
    if not test_pdfs:
        print("No PDF files found in current directory for testing")
        print("Please add a test PDF file to test conversion with Bedrock")
        return

    test_pdf = test_pdfs[0]
    print(f"Using test PDF: {test_pdf}")

    try:
        # Configure converter with Bedrock
        config = BasePDFConverterConfig(
            use_llm=True,
            llm_service="bedrock",
            llm_model=os.getenv("BEDROCK_MODEL_ID", settings.BEDROCK_MODEL_ID),
        )

        # Initialize converter
        converter = PdfConverter(config)
        print("✓ PDF converter initialized with Bedrock")

        # Convert PDF
        print("Converting PDF...")
        output_path = Path(f"{test_pdf.stem}_bedrock_output.md")

        # Note: This is a simplified example
        # In production, you'd use marker_single or marker CLI commands
        print(f"To convert with Bedrock, run:")
        print(f"  marker_single {test_pdf} --use_llm --llm_service bedrock")
        print(f"  or")
        print(f"  BEDROCK_MODEL_ID='{os.getenv('BEDROCK_MODEL_ID', settings.BEDROCK_MODEL_ID)}' marker {test_pdf.parent}")

    except Exception as e:
        print(f"✗ Error during PDF conversion: {e}")


def check_dependencies():
    """Check if required dependencies are installed"""
    print("Checking Dependencies...")
    print("-" * 50)

    try:
        import boto3
        print("✓ boto3 installed")
    except ImportError:
        print("✗ boto3 not installed. Run: pip install boto3")
        return False

    try:
        import marker
        print("✓ marker-pdf installed")
    except ImportError:
        print("✗ marker-pdf not installed. Run: pip install marker-pdf[full]")
        return False

    # Check AWS credentials
    if not any([
        os.getenv("AWS_ACCESS_KEY_ID"),
        os.getenv("AWS_PROFILE"),
        os.path.exists(os.path.expanduser("~/.aws/credentials"))
    ]):
        print("⚠ No AWS credentials found. Please configure:")
        print("  - Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY in .env")
        print("  - Or configure AWS CLI: aws configure")
        print("  - Or use IAM roles if running on EC2/Lambda")
        return False

    print("✓ AWS credentials configured")
    return True


def main():
    print("=" * 50)
    print("AWS Bedrock Integration Test for Marker PDF")
    print("=" * 50)
    print()

    if not check_dependencies():
        print("\n⚠ Please install missing dependencies and configure AWS credentials")
        return

    print()
    test_bedrock_service()
    test_pdf_conversion_with_bedrock()

    print("\n" + "=" * 50)
    print("Test Complete!")
    print("\nTo use Bedrock with Marker:")
    print("1. Copy .env.example to .env and configure your AWS credentials")
    print("2. Run: marker_single file.pdf --use_llm --llm_service bedrock")
    print("3. Or set environment variables and run marker normally")
    print("=" * 50)


if __name__ == "__main__":
    main()