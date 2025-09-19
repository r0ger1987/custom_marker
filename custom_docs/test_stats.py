#!/usr/bin/env python3
"""Test statistics extraction from MarkdownOutput"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(project_root)

from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict

# Create models and converter
models = create_model_dict()
converter = PdfConverter(artifact_dict=models)

# Convert PDF
pdf_path = "inputs/Modifier_un_CAV_PROC_G_19_P_1.pdf"
result = converter(pdf_path)

print("=== Testing Statistics Extraction ===")

# Test pages count
pages_processed = len(result.metadata.get('page_stats', [])) if hasattr(result, 'metadata') and result.metadata else 0
print(f"Pages processed: {pages_processed}")

# Test images count
images_extracted = len(result.images) if hasattr(result, 'images') and result.images else 0
print(f"Images extracted: {images_extracted}")

# Test tables count
tables_found = 0
if hasattr(result, 'metadata') and result.metadata:
    page_stats = result.metadata.get('page_stats', [])
    for page_stat in page_stats:
        block_counts = dict(page_stat.get('block_counts', []))
        tables_found += block_counts.get('Table', 0)
        print(f"Page {page_stat.get('page_id', 'unknown')}: {dict(block_counts)}")

print(f"Total tables found: {tables_found}")

# Test markdown content
markdown_content = result.markdown if hasattr(result, 'markdown') else ""
print(f"Markdown content length: {len(markdown_content)}")

print("✅ Statistics extraction test successful!")