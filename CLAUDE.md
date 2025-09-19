# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Installation and Setup
- Install with Poetry: `poetry install`
- Install with full document support: `pip install marker-pdf[full]`
- For development dependencies: `poetry install` (includes jupyter, streamlit, pytest, etc.)

### Testing
- Run all tests: `pytest`
- Tests are located in `tests/` directory
- Configuration in `pytest.ini`

### Linting and Formatting
- This project uses Ruff for linting and formatting (configured in `.pre-commit-config.yaml`)
- Run linting: `ruff check --fix`  
- Run formatting: `ruff format`
- Pre-commit hooks are configured to run these automatically

### Running the Application
- Convert single file: `marker_single /path/to/file.pdf`
- Convert multiple files: `marker /path/to/input/folder`
- Run GUI: `marker_gui` (requires streamlit)
- Run API server: `marker_server --port 8001`
- Multi-GPU conversion: `NUM_DEVICES=4 NUM_WORKERS=15 marker_chunk_convert ../pdf_in ../md_out`

### Key CLI Options
- `--use_llm`: Enable LLM mode for higher accuracy (requires API key setup)
- `--force_ocr`: Force OCR on entire document
- `--output_format [markdown|json|html|chunks]`: Specify output format
- `--debug`: Enable debug mode with detailed logging and layout images

## Architecture Overview

Marker is a modular document conversion pipeline with these core components:

### Core Architecture
- **Providers** (`marker/providers/`): Extract information from source files (PDF, images, DOCX, etc.)
- **Builders** (`marker/builders/`): Generate initial document blocks and fill in text using provider data
- **Processors** (`marker/processors/`): Process specific block types (tables, equations, forms, etc.)
- **Renderers** (`marker/renderers/`): Convert blocks to final output formats (markdown, JSON, HTML)
- **Converters** (`marker/converters/`): Orchestrate the complete end-to-end pipeline
- **Schema** (`marker/schema/`): Block type definitions and data models

### Key Converters
- `PdfConverter`: Full PDF conversion (default)
- `TableConverter`: Table extraction only
- `OCRConverter`: OCR-only processing
- `ExtractionConverter`: Structured data extraction (beta)

### Block Types
Document structure is represented as a tree of blocks defined in `marker/schema/`:
- Text blocks: `Text`, `SectionHeader`, `Caption`
- Layout blocks: `Table`, `Figure`, `Form`, `ListGroup`
- Math blocks: `Equation`, `TextInlineMath`
- Metadata blocks: `PageHeader`, `PageFooter`, `TableOfContents`

### Processing Pipeline
1. Provider extracts raw data from source file
2. Builders create initial block structure
3. Processors enhance blocks (OCR, table formatting, equation parsing, LLM improvements)
4. Renderers convert blocks to output format

### Configuration
- Settings in `marker/settings.py` with environment variable support
- Supports custom configuration via `ConfigParser` class
- Key environment variables: `TORCH_DEVICE`, `GOOGLE_API_KEY`, `OUTPUT_DIR`

### LLM Integration
When `--use_llm` is enabled:
- Supports Gemini, OpenAI, Claude, Ollama, Azure OpenAI, and Vertex AI
- LLM processors: table merging, form extraction, image descriptions, handwriting recognition
- Configurable via `--llm_service` parameter

### Extensibility
- Add custom processors by extending `BaseProcessor`
- Create new output formats with custom renderers
- Support new input formats by writing providers
- Processors and renderers can be passed directly to converters

## Development Notes

### Model Dependencies
- Uses deep learning models from Surya OCR for text detection and recognition
- Models are automatically downloaded from `https://models.datalab.to/artifacts`
- GPU/CPU/MPS device detection is automatic but can be overridden

### Testing Patterns
- Tests use parameterized fixtures for different document types
- Filename marker: `@pytest.mark.filename("test.pdf")` to specify test documents
- Mock external services for consistent testing

### Performance Considerations
- Multi-worker processing supported for batch conversion
- Memory usage: ~5GB VRAM per worker at peak, 3.5GB average
- Projected throughput: 25 pages/second on H100 in batch mode