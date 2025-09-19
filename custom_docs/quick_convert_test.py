#!/usr/bin/env python3
"""Quick test of convert_to_markdown.py without full process"""

import sys
import os
import time
from pathlib import Path

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(project_root)

from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict

print("✅ Testing conversion process...")

# Create models and converter
models = create_model_dict()
converter = PdfConverter(artifact_dict=models)

# Convert PDF
pdf_path = "inputs/Modifier_un_CAV_PROC_G_19_P_1.pdf"
result = converter(pdf_path)

# Test all the statistics extraction like in convert_to_markdown.py
print("=== Extraction des statistiques ===")

# Pages
pages_processed = len(result.metadata.get('page_stats', [])) if hasattr(result, 'metadata') and result.metadata else 0
print(f"Pages traitées : {pages_processed}")

# Images
images_extracted = len(result.images) if hasattr(result, 'images') and result.images else 0
print(f"Images extraites : {images_extracted}")

# Tables et équations
tables_found = 0
equations_found = 0

if hasattr(result, 'metadata') and result.metadata:
    page_stats = result.metadata.get('page_stats', [])
    for page_stat in page_stats:
        block_counts = dict(page_stat.get('block_counts', []))
        tables_found += block_counts.get('Table', 0)

# Markdown content
markdown_content = result.markdown if hasattr(result, 'markdown') else ""

if markdown_content:
    # Compter les équations ($ ou $$) - optionnel car très simple
    equations_found = markdown_content.count('$') // 2

print(f"Tables trouvées : {tables_found}")
print(f"Équations estimées : {equations_found}")
print(f"Longueur markdown : {len(markdown_content)}")

# Test de création de fichier
output_dir = Path("/home/roger/RAG/custom_marker/custom_docs/outputs/test_conversion")
output_dir.mkdir(parents=True, exist_ok=True)
markdown_file = output_dir / "test_output.md"

with open(markdown_file, 'w', encoding='utf-8') as f:
    f.write(markdown_content)

print(f"✅ Test réussi ! Fichier créé : {markdown_file}")
print(f"📊 Statistiques finales : {pages_processed} pages, {images_extracted} images, {tables_found} tables")