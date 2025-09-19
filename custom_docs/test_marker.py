#!/usr/bin/env python3
"""Test script to understand Marker API"""

from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict

# Create models
models = create_model_dict()

# Create converter
converter = PdfConverter(artifact_dict=models)

# Convert a small test
pdf_path = "inputs/Modifier_un_CAV_PROC_G_19_P_1.pdf"
result = converter(pdf_path)

print("Type:", type(result))
print("Attributes:", [attr for attr in dir(result) if not attr.startswith('_')])

# Try to access markdown content
if hasattr(result, 'markdown'):
    print("Has markdown attribute")
    print("Markdown type:", type(result.markdown))
    print("First 200 chars:", result.markdown[:200])
elif hasattr(result, 'text'):
    print("Has text attribute")
    print("Text type:", type(result.text))
    print("First 200 chars:", result.text[:200])
else:
    print("Available methods:", [m for m in dir(result) if callable(getattr(result, m)) and not m.startswith('_')])