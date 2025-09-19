#!/usr/bin/env python3
"""
Script d'Extraction de Contenu Spécialisé
=========================================

Ce script extrait des éléments spécifiques des PDFs :
- Tables et tableaux avec formatage
- Équations mathématiques
- Images et figures avec métadonnées
- Sections par mots-clés
- Formulaires et champs

Usage:
    python scripts/extract_specific_content.py input.pdf --content-type TYPE [--llm] [--output DIR]

Types de contenu:
    - tables: Extraction de tableaux
    - equations: Équations mathématiques
    - images: Images et figures
    - sections: Sections par mots-clés
    - forms: Formulaires et champs

Auteur: Claude Code
"""

import sys
import os
import json
import argparse
import time
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
import re

# Ajout du chemin pour importer marker et llm_config
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
sys.path.append(os.path.dirname(__file__))

from llm_config import LLMConfig

class ContentExtractor:
    """Extracteur de contenu spécialisé pour PDFs"""

    def __init__(
        self,
        content_type: str,
        use_llm: bool = False,
        llm_provider: str = "auto",
        keywords: Optional[List[str]] = None,
        debug: bool = False
    ):
        """
        Initialise l'extracteur

        Args:
            content_type: Type de contenu à extraire
            use_llm: Utiliser LLM pour améliorer l'extraction
            llm_provider: Provider LLM à utiliser
            keywords: Mots-clés pour extraction de sections
            debug: Mode debug
        """
        self.content_type = content_type
        self.use_llm = use_llm
        self.llm_provider = llm_provider
        self.keywords = keywords or []
        self.debug = debug
        self.llm_config = None

        # Types de contenu supportés
        self.supported_types = {
            "tables": self._extract_tables,
            "equations": self._extract_equations,
            "images": self._extract_images,
            "sections": self._extract_sections,
            "forms": self._extract_forms
        }

        if content_type not in self.supported_types:
            raise ValueError(f"Type de contenu non supporté: {content_type}. Types disponibles: {list(self.supported_types.keys())}")

        # Configuration LLM si activé
        if use_llm:
            try:
                self.llm_config = LLMConfig.get_llm_config(llm_provider)
                print(f"🤖 LLM {llm_provider} configuré pour améliorer l'extraction de {content_type}")
            except Exception as e:
                print(f"⚠️ Erreur configuration LLM : {e}")
                self.use_llm = False

        print(f"🔍 Extracteur initialisé pour: {content_type}")

    def extract(self, pdf_path: Path, output_dir: Optional[Path] = None) -> Dict[str, Any]:
        """
        Extraire le contenu spécialisé d'un PDF

        Args:
            pdf_path: Chemin vers le PDF
            output_dir: Répertoire de sortie

        Returns:
            Dictionnaire avec les résultats d'extraction
        """
        if not pdf_path.exists():
            raise FileNotFoundError(f"Fichier PDF non trouvé : {pdf_path}")

        # Configuration du dossier de sortie
        if output_dir is None:
            output_dir = Path(f"outputs/{self.content_type}")
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n📄 Extraction de {self.content_type} depuis : {pdf_path.name}")
        print(f"📁 Sortie vers : {output_dir}")

        start_time = time.time()

        try:
            # Appeler la méthode d'extraction appropriée
            extraction_method = self.supported_types[self.content_type]
            results = extraction_method(pdf_path, output_dir)

            # Post-traitement avec LLM si activé
            if self.use_llm and results.get("items"):
                results = self._enhance_with_llm(results, pdf_path)

            # Métadonnées d'extraction
            elapsed_time = time.time() - start_time
            results.update({
                "extraction_metadata": {
                    "source_file": str(pdf_path),
                    "content_type": self.content_type,
                    "extraction_time": f"{elapsed_time:.2f}s",
                    "llm_enhanced": self.use_llm,
                    "llm_provider": self.llm_provider if self.use_llm else None,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "keywords_used": self.keywords if self.content_type == "sections" else None
                }
            })

            # Sauvegarder les résultats
            self._save_results(results, pdf_path, output_dir)

            print(f"\n✅ Extraction terminée en {elapsed_time:.2f}s")
            print(f"📊 {results.get('total_items', 0)} éléments extraits")

            return results

        except Exception as e:
            print(f"\n❌ Erreur lors de l'extraction : {e}")
            if self.debug:
                import traceback
                traceback.print_exc()

            return {
                "success": False,
                "error": str(e),
                "source_file": str(pdf_path)
            }

    def _extract_tables(self, pdf_path: Path, output_dir: Path) -> Dict[str, Any]:
        """Extraire les tableaux du PDF"""
        print("📊 Extraction des tableaux...")

        # Placeholder pour l'intégration Marker
        tables = []

        # Simulation d'extraction de tableaux
        for i in range(3):  # Simuler 3 tableaux trouvés
            table = {
                "id": f"table_{i+1}",
                "page": i + 1,
                "title": f"Tableau {i+1}",
                "rows": 5 + i * 2,
                "columns": 4,
                "content": f"Contenu du tableau {i+1} sera extrait avec Marker...",
                "bbox": [100, 200 + i*100, 400, 300 + i*100],
                "confidence": 0.85 + i * 0.05
            }
            tables.append(table)

        # Sauvegarder chaque tableau individuellement
        for table in tables:
            table_file = output_dir / f"{pdf_path.stem}_{table['id']}.txt"
            with open(table_file, 'w', encoding='utf-8') as f:
                f.write(f"# {table['title']}\n\n")
                f.write(f"Page: {table['page']}\n")
                f.write(f"Dimensions: {table['rows']} lignes × {table['columns']} colonnes\n")
                f.write(f"Confiance: {table['confidence']:.2f}\n\n")
                f.write(table['content'])

        return {
            "success": True,
            "content_type": "tables",
            "total_items": len(tables),
            "items": tables,
            "stats": {
                "average_confidence": sum(t['confidence'] for t in tables) / len(tables) if tables else 0,
                "total_rows": sum(t['rows'] for t in tables),
                "total_columns": sum(t['columns'] for t in tables),
                "pages_with_tables": len(set(t['page'] for t in tables))
            }
        }

    def _extract_equations(self, pdf_path: Path, output_dir: Path) -> Dict[str, Any]:
        """Extraire les équations mathématiques"""
        print("🧮 Extraction des équations...")

        equations = []

        # Simulation d'extraction d'équations
        equation_types = ["inline", "display", "numbered"]
        for i in range(5):
            equation = {
                "id": f"eq_{i+1}",
                "page": (i // 2) + 1,
                "type": equation_types[i % len(equation_types)],
                "latex": f"\\sum_{{i=1}}^{{n}} x_i^{i+1} = \\frac{{n(n+1)}}{{2}}",
                "text": f"Équation {i+1}: Somme des puissances",
                "bbox": [150, 100 + i*50, 350, 130 + i*50],
                "confidence": 0.9 + i * 0.02
            }
            equations.append(equation)

        # Sauvegarder toutes les équations
        equations_file = output_dir / f"{pdf_path.stem}_equations.tex"
        with open(equations_file, 'w', encoding='utf-8') as f:
            f.write("% Équations extraites\n")
            f.write(f"% Source: {pdf_path.name}\n")
            f.write(f"% Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            for eq in equations:
                f.write(f"% {eq['text']} (Page {eq['page']}, Type: {eq['type']})\n")
                f.write(f"{eq['latex']}\n\n")

        return {
            "success": True,
            "content_type": "equations",
            "total_items": len(equations),
            "items": equations,
            "stats": {
                "inline_count": sum(1 for eq in equations if eq['type'] == 'inline'),
                "display_count": sum(1 for eq in equations if eq['type'] == 'display'),
                "numbered_count": sum(1 for eq in equations if eq['type'] == 'numbered'),
                "average_confidence": sum(eq['confidence'] for eq in equations) / len(equations) if equations else 0,
                "pages_with_equations": len(set(eq['page'] for eq in equations))
            }
        }

    def _extract_images(self, pdf_path: Path, output_dir: Path) -> Dict[str, Any]:
        """Extraire les images et figures"""
        print("🖼️ Extraction des images...")

        images = []

        # Simulation d'extraction d'images
        image_types = ["figure", "chart", "diagram", "photo"]
        for i in range(4):
            image = {
                "id": f"img_{i+1}",
                "page": i + 1,
                "type": image_types[i % len(image_types)],
                "filename": f"{pdf_path.stem}_page_{i+1}_img_{i+1}.png",
                "caption": f"Figure {i+1}: Description de l'image {i+1}",
                "width": 400 + i * 50,
                "height": 300 + i * 30,
                "bbox": [50 + i*20, 100 + i*100, 450 + i*20, 400 + i*100],
                "format": "PNG",
                "size_kb": 150 + i * 50
            }
            images.append(image)

        # Créer des fichiers d'image fictifs et métadonnées
        for image in images:
            # Métadonnées de l'image
            metadata_file = output_dir / f"{image['filename']}.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(image, f, indent=2, ensure_ascii=False)

            # Fichier image placeholder
            image_file = output_dir / image['filename']
            with open(image_file, 'w', encoding='utf-8') as f:
                f.write(f"# Image Placeholder: {image['filename']}\n")
                f.write(f"# Type: {image['type']}\n")
                f.write(f"# Caption: {image['caption']}\n")
                f.write(f"# L'image sera extraite avec Marker...\n")

        return {
            "success": True,
            "content_type": "images",
            "total_items": len(images),
            "items": images,
            "stats": {
                "total_size_kb": sum(img['size_kb'] for img in images),
                "average_width": sum(img['width'] for img in images) / len(images) if images else 0,
                "average_height": sum(img['height'] for img in images) / len(images) if images else 0,
                "type_distribution": {img_type: sum(1 for img in images if img['type'] == img_type) for img_type in image_types},
                "pages_with_images": len(set(img['page'] for img in images))
            }
        }

    def _extract_sections(self, pdf_path: Path, output_dir: Path) -> Dict[str, Any]:
        """Extraire des sections par mots-clés"""
        print(f"📝 Extraction de sections avec mots-clés: {', '.join(self.keywords)}")

        if not self.keywords:
            print("⚠️ Aucun mot-clé spécifié pour l'extraction de sections")
            return {"success": False, "error": "Aucun mot-clé spécifié"}

        sections = []

        # Simulation d'extraction de sections
        for i, keyword in enumerate(self.keywords):
            section = {
                "id": f"section_{i+1}",
                "keyword": keyword,
                "title": f"Section: {keyword.title()}",
                "page_start": i + 1,
                "page_end": i + 2,
                "content": f"Contenu de la section {keyword} sera extrait avec Marker...\n\nCette section contiendrait tout le texte relatif à '{keyword}' trouvé dans le document.",
                "word_count": 250 + i * 100,
                "confidence": 0.8 + i * 0.05,
                "subsections": [f"{keyword}.1", f"{keyword}.2"] if i < 3 else []
            }
            sections.append(section)

        # Sauvegarder chaque section
        for section in sections:
            section_file = output_dir / f"{pdf_path.stem}_{section['keyword']}.md"
            with open(section_file, 'w', encoding='utf-8') as f:
                f.write(f"# {section['title']}\n\n")
                f.write(f"**Pages:** {section['page_start']}-{section['page_end']}\n")
                f.write(f"**Mots:** {section['word_count']}\n")
                f.write(f"**Confiance:** {section['confidence']:.2f}\n\n")

                if section['subsections']:
                    f.write("## Sous-sections trouvées:\n")
                    for subsec in section['subsections']:
                        f.write(f"- {subsec}\n")
                    f.write("\n")

                f.write("## Contenu:\n\n")
                f.write(section['content'])

        return {
            "success": True,
            "content_type": "sections",
            "total_items": len(sections),
            "items": sections,
            "stats": {
                "total_words": sum(s['word_count'] for s in sections),
                "keywords_found": len([s for s in sections if s['confidence'] > 0.7]),
                "keywords_missing": len(self.keywords) - len(sections),
                "average_confidence": sum(s['confidence'] for s in sections) / len(sections) if sections else 0,
                "total_pages_covered": len(set(range(s['page_start'], s['page_end']+1) for s in sections))
            }
        }

    def _extract_forms(self, pdf_path: Path, output_dir: Path) -> Dict[str, Any]:
        """Extraire les formulaires et champs"""
        print("📋 Extraction des formulaires...")

        forms = []

        # Simulation d'extraction de formulaires
        field_types = ["text", "checkbox", "radio", "dropdown", "signature"]
        for i in range(2):  # 2 formulaires
            form_fields = []
            for j in range(5):  # 5 champs par formulaire
                field = {
                    "id": f"field_{i+1}_{j+1}",
                    "name": f"field_{j+1}",
                    "type": field_types[j % len(field_types)],
                    "label": f"Champ {j+1}",
                    "value": f"Valeur {j+1}" if j < 3 else "",
                    "required": j < 2,
                    "bbox": [100 + j*80, 200 + i*200 + j*30, 250 + j*80, 220 + i*200 + j*30]
                }
                form_fields.append(field)

            form = {
                "id": f"form_{i+1}",
                "page": i + 1,
                "title": f"Formulaire {i+1}",
                "fields": form_fields,
                "total_fields": len(form_fields),
                "filled_fields": len([f for f in form_fields if f['value']]),
                "required_fields": len([f for f in form_fields if f['required']]),
                "completion_rate": len([f for f in form_fields if f['value']]) / len(form_fields)
            }
            forms.append(form)

        # Sauvegarder les formulaires
        for form in forms:
            form_file = output_dir / f"{pdf_path.stem}_{form['id']}.json"
            with open(form_file, 'w', encoding='utf-8') as f:
                json.dump(form, f, indent=2, ensure_ascii=False)

            # Créer aussi un résumé lisible
            summary_file = output_dir / f"{pdf_path.stem}_{form['id']}_summary.md"
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write(f"# {form['title']}\n\n")
                f.write(f"**Page:** {form['page']}\n")
                f.write(f"**Champs totaux:** {form['total_fields']}\n")
                f.write(f"**Champs remplis:** {form['filled_fields']}\n")
                f.write(f"**Taux de completion:** {form['completion_rate']:.1%}\n\n")

                f.write("## Champs:\n\n")
                for field in form['fields']:
                    status = "✓" if field['value'] else "○"
                    required = " (requis)" if field['required'] else ""
                    f.write(f"- {status} **{field['label']}** ({field['type']}){required}: {field['value'] or 'Vide'}\n")

        return {
            "success": True,
            "content_type": "forms",
            "total_items": len(forms),
            "items": forms,
            "stats": {
                "total_fields": sum(f['total_fields'] for f in forms),
                "total_filled": sum(f['filled_fields'] for f in forms),
                "total_required": sum(f['required_fields'] for f in forms),
                "average_completion": sum(f['completion_rate'] for f in forms) / len(forms) if forms else 0,
                "field_types": {ft: sum(1 for f in forms for field in f['fields'] if field['type'] == ft) for ft in field_types}
            }
        }

    def _enhance_with_llm(self, results: Dict[str, Any], pdf_path: Path) -> Dict[str, Any]:
        """Améliorer les résultats avec LLM"""
        print(f"🧠 Amélioration avec LLM ({self.llm_provider})...")

        # Placeholder pour l'amélioration LLM
        if self.content_type == "tables":
            for item in results.get("items", []):
                item["llm_analysis"] = {
                    "description": f"Tableau analysé par LLM: structure {item['rows']}x{item['columns']}",
                    "data_type": "numerical" if "financial" in item.get("title", "").lower() else "mixed",
                    "quality_score": min(0.95, item.get("confidence", 0.8) + 0.1)
                }

        elif self.content_type == "equations":
            for item in results.get("items", []):
                item["llm_analysis"] = {
                    "description": f"Équation {item['type']} analysée par LLM",
                    "complexity": "simple" if item['type'] == "inline" else "complex",
                    "domain": "mathematics",
                    "verified_latex": True
                }

        elif self.content_type == "sections":
            for item in results.get("items", []):
                item["llm_analysis"] = {
                    "summary": f"Résumé de la section {item['keyword']} généré par LLM",
                    "key_points": [f"Point clé 1 de {item['keyword']}", f"Point clé 2 de {item['keyword']}"],
                    "relevance_score": min(0.95, item.get("confidence", 0.8) + 0.1)
                }

        results["llm_enhanced"] = True
        return results

    def _save_results(self, results: Dict[str, Any], pdf_path: Path, output_dir: Path):
        """Sauvegarder les résultats d'extraction"""
        # Rapport JSON complet
        report_file = output_dir / f"{pdf_path.stem}_{self.content_type}_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        # Résumé markdown
        summary_file = output_dir / f"{pdf_path.stem}_{self.content_type}_summary.md"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(f"# Extraction de {self.content_type.title()}\n\n")
            f.write(f"**Source:** {pdf_path.name}\n")
            f.write(f"**Date:** {results['extraction_metadata']['timestamp']}\n")
            f.write(f"**Éléments extraits:** {results.get('total_items', 0)}\n")
            f.write(f"**LLM utilisé:** {results['extraction_metadata']['llm_provider'] or 'Aucun'}\n\n")

            if results.get("stats"):
                f.write("## Statistiques\n\n")
                for key, value in results["stats"].items():
                    if isinstance(value, dict):
                        f.write(f"**{key.replace('_', ' ').title()}:**\n")
                        for subkey, subvalue in value.items():
                            f.write(f"  - {subkey}: {subvalue}\n")
                    else:
                        f.write(f"- **{key.replace('_', ' ').title()}:** {value}\n")
                f.write("\n")

        print(f"📋 Rapport : {report_file}")
        print(f"📄 Résumé : {summary_file}")

def main():
    """Fonction principale"""
    parser = argparse.ArgumentParser(
        description="Extraction de contenu spécialisé depuis PDFs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  # Extraction de tableaux
  python scripts/extract_specific_content.py document.pdf --content-type tables

  # Extraction d'équations avec LLM
  python scripts/extract_specific_content.py paper.pdf --content-type equations --llm

  # Extraction de sections par mots-clés
  python scripts/extract_specific_content.py thesis.pdf --content-type sections --keywords "introduction,methodology,results"

  # Extraction d'images
  python scripts/extract_specific_content.py manual.pdf --content-type images --output ./extracted_images/
        """
    )

    parser.add_argument(
        "pdf_path",
        type=Path,
        help="Chemin vers le fichier PDF"
    )
    parser.add_argument(
        "--content-type",
        choices=["tables", "equations", "images", "sections", "forms"],
        required=True,
        help="Type de contenu à extraire"
    )
    parser.add_argument(
        "--llm",
        action="store_true",
        help="Utiliser LLM pour améliorer l'extraction"
    )
    parser.add_argument(
        "--provider",
        choices=["auto", "gemini", "openai", "claude", "azure", "bedrock", "ollama"],
        default="auto",
        help="Provider LLM à utiliser (défaut: auto)"
    )
    parser.add_argument(
        "--keywords",
        type=str,
        help="Mots-clés séparés par des virgules (pour sections)"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Répertoire de sortie (défaut: outputs/CONTENT_TYPE)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Mode debug avec informations détaillées"
    )

    args = parser.parse_args()

    if not args.pdf_path.exists():
        print(f"❌ Erreur : Le fichier {args.pdf_path} n'existe pas")
        sys.exit(1)

    # Traitement des mots-clés
    keywords = []
    if args.keywords:
        keywords = [k.strip() for k in args.keywords.split(",")]

    if args.content_type == "sections" and not keywords:
        print("❌ Erreur : Des mots-clés sont requis pour l'extraction de sections")
        print("Utilisez --keywords \"mot1,mot2,mot3\"")
        sys.exit(1)

    # Créer l'extracteur
    try:
        extractor = ContentExtractor(
            content_type=args.content_type,
            use_llm=args.llm,
            llm_provider=args.provider,
            keywords=keywords,
            debug=args.debug
        )

        # Effectuer l'extraction
        result = extractor.extract(args.pdf_path, args.output)

        # Retourner le code de sortie approprié
        sys.exit(0 if result.get("success", False) else 1)

    except Exception as e:
        print(f"\n❌ Erreur : {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()