#!/usr/bin/env python3
"""Test script to verify MarkdownOutput structure"""

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

print("=== MarkdownOutput Structure ===")
print(f"Type: {type(result)}")
print(f"Attributes: {[attr for attr in dir(result) if not attr.startswith('_')]}")

if hasattr(result, 'metadata'):
    print(f"\nMetadata type: {type(result.metadata)}")
    print(f"Metadata content: {result.metadata}")

if hasattr(result, 'images'):
    print(f"\nImages count: {len(result.images) if result.images else 0}")

print(f"\nMarkdown length: {len(result.markdown) if hasattr(result, 'markdown') else 0}")
print(f"First 200 chars: {result.markdown[:200] if hasattr(result, 'markdown') else 'No markdown'}")