#!/usr/bin/env python3
"""
Script de conversion PDF vers Markdown optimisé avec Marker
=========================================================

Ce script convertit des PDFs en markdown avec différents niveaux d'optimisation :
- Mode rapide : conversion basique avec Marker
- Mode équilibré : OCR sélectif + formatage avancé
- Mode qualité : LLM + OCR forcé + toutes optimisations

Usage:
    python convert_to_markdown.py input.pdf [--mode MODE] [--output DIR]

Modes disponibles:
    - fast: Conversion rapide (défaut)
    - balanced: Équilibre qualité/vitesse
    - quality: Qualité maximale avec LLM

Auteur: Claude Code
"""

import sys
import os
import argparse
import json
import time
import traceback
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

# Ajout du chemin racine pour importer marker
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(project_root)

try:
    # Import Marker et ses modules
    from marker.converters import PdfConverter
    from marker.models import load_all_models
    from marker.settings import settings
    from marker.logger import configure_logging
    MARKER_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Marker non disponible: {e}")
    MARKER_AVAILABLE = False

# Import du module de configuration LLM
from llm_config import LLMConfig

@dataclass
class ConversionResult:
    """Résultat de conversion"""
    success: bool
    output_file: Optional[str] = None
    metadata_file: Optional[str] = None
    conversion_time: float = 0.0
    pages_processed: int = 0
    images_extracted: int = 0
    tables_found: int = 0
    equations_found: int = 0
    error: Optional[str] = None
    warnings: List[str] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []

class MarkdownConverter:
    """Convertisseur PDF vers Markdown avec différents modes d'optimisation et intégration Marker complète"""

    CONVERSION_MODES = {
        "fast": {
            "description": "Conversion rapide - qualité basique",
            "config": {
                "output_format": "markdown",
                "force_ocr": False,
                "use_llm": False,
                "extract_images": True,
                "strip_existing_ocr": False,
                "debug": False,
                "workers": 1,
            }
        },
        "balanced": {
            "description": "Équilibre qualité/vitesse - OCR sélectif",
            "config": {
                "output_format": "markdown",
                "force_ocr": False,
                "use_llm": False,
                "extract_images": True,
                "strip_existing_ocr": False,
                "redo_inline_math": False,
                "debug": False,
                "workers": 2,
            }
        },
        "quality": {
            "description": "Qualité maximale - LLM + OCR forcé",
            "config": {
                "output_format": "markdown",
                "force_ocr": True,
                "use_llm": True,
                "extract_images": True,
                "strip_existing_ocr": True,
                "redo_inline_math": True,
                "disable_image_extraction": False,
                "debug": True,
                "workers": 1,
            }
        }
    }

    def __init__(self, mode: str = "fast", custom_config: Optional[Dict] = None, llm_provider: str = "auto"):
        """
        Initialise le convertisseur

        Args:
            mode: Mode de conversion ('fast', 'balanced', 'quality')
            custom_config: Configuration personnalisée (optionnel)
            llm_provider: Provider LLM à utiliser si mode quality
        """
        if not MARKER_AVAILABLE:
            raise RuntimeError("Marker n'est pas disponible. Installez-le avec: pip install marker-pdf[full]")

        if mode not in self.CONVERSION_MODES:
            raise ValueError(f"Mode non supporté: {mode}. Modes disponibles: {list(self.CONVERSION_MODES.keys())}")

        self.mode = mode
        self.mode_config = self.CONVERSION_MODES[mode]
        self.llm_provider = llm_provider
        self.models = None

        # Configuration de base
        config = self.mode_config["config"].copy()

        # Application de la configuration personnalisée
        if custom_config:
            config.update(custom_config)

        # Configuration LLM si mode quality
        if config.get("use_llm", False):
            try:
                llm_config = LLMConfig.get_llm_config(llm_provider, config)
                config.update(llm_config)
                print(f"🤖 Mode {mode} avec LLM : {llm_provider}")
            except Exception as e:
                print(f"⚠️ Erreur configuration LLM, passage en mode sans LLM : {e}")
                config["use_llm"] = False

        self.config = config
        print(f"📋 Mode de conversion : {mode} - {self.mode_config['description']}")

    def _load_models(self):
        """Charge les modèles Marker si nécessaire"""
        if self.models is None:
            print("🔄 Chargement des modèles Marker...")
            try:
                # Configuration du device
                if self.config.get("debug"):
                    configure_logging()

                # Charger les modèles
                self.models = load_all_models()
                print("✅ Modèles Marker chargés")
            except Exception as e:
                print(f"❌ Erreur lors du chargement des modèles: {e}")
                raise

    def convert(self, pdf_path: Path, output_dir: Optional[Path] = None, max_pages: Optional[int] = None) -> ConversionResult:
        """
        Convertit un PDF en markdown avec Marker

        Args:
            pdf_path: Chemin vers le PDF à convertir
            output_dir: Dossier de sortie (optionnel)
            max_pages: Nombre maximum de pages à traiter

        Returns:
            ConversionResult avec les résultats détaillés
        """
        if not pdf_path.exists():
            raise FileNotFoundError(f"Fichier PDF non trouvé : {pdf_path}")

        # Configuration du dossier de sortie
        if output_dir is None:
            output_dir = Path("outputs")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Créer des sous-dossiers
        images_dir = output_dir / "images"
        metadata_dir = output_dir / "metadata"
        images_dir.mkdir(exist_ok=True)
        metadata_dir.mkdir(exist_ok=True)

        print(f"\n📄 Conversion de : {pdf_path.name}")
        print(f"📁 Dossier de sortie : {output_dir}")
        print(f"🔧 Mode : {self.mode}")

        start_time = time.time()
        warnings = []

        try:
            # Charger les modèles
            self._load_models()

            # Mise à jour de la configuration
            config = self.config.copy()
            if max_pages:
                config["max_pages"] = max_pages

            # Créer le convertisseur Marker
            converter = PdfConverter(
                config=config,
                artifact_dict=self.models,
                llm_service=config.get("llm_service") if config.get("use_llm") else None
            )

            print("🔄 Début de la conversion...")

            # Conversion avec Marker
            doc_result = converter(pdf_path)

            # Extraire les statistiques
            pages_processed = len(doc_result.pages) if hasattr(doc_result, 'pages') else 0
            images_extracted = 0
            tables_found = 0
            equations_found = 0

            # Compter les éléments dans les blocs
            for page in doc_result.pages:
                for block in page.blocks:
                    if hasattr(block, 'block_type'):
                        if 'table' in str(block.block_type).lower():
                            tables_found += 1
                        elif 'equation' in str(block.block_type).lower():
                            equations_found += 1
                        elif 'figure' in str(block.block_type).lower():
                            images_extracted += 1

            # Générer le markdown
            markdown_content = doc_result.to_markdown()

            # Fichiers de sortie
            output_file = output_dir / f"{pdf_path.stem}_{self.mode}.md"
            metadata_file = metadata_dir / f"{pdf_path.stem}_metadata.json"

            # Sauvegarder le markdown
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(markdown_content)

            # Sauvegarder les métadonnées
            elapsed_time = time.time() - start_time
            metadata = {
                "source_file": str(pdf_path),
                "conversion_mode": self.mode,
                "conversion_time": elapsed_time,
                "pages_processed": pages_processed,
                "images_extracted": images_extracted,
                "tables_found": tables_found,
                "equations_found": equations_found,
                "config_used": config,
                "llm_provider": self.llm_provider if config.get("use_llm") else None,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "warnings": warnings
            }

            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

            print(f"\n✅ Conversion réussie !")
            print(f"📝 Fichier markdown : {output_file}")
            print(f"📊 Pages traitées : {pages_processed}")
            print(f"🖼️  Images extraites : {images_extracted}")
            print(f"📋 Tableaux trouvés : {tables_found}")
            print(f"🧮 Équations trouvées : {equations_found}")
            print(f"⏱️  Temps : {elapsed_time:.2f}s")

            if warnings:
                print(f"⚠️ Avertissements : {len(warnings)}")
                for warning in warnings:
                    print(f"  • {warning}")

            return ConversionResult(
                success=True,
                output_file=str(output_file),
                metadata_file=str(metadata_file),
                conversion_time=elapsed_time,
                pages_processed=pages_processed,
                images_extracted=images_extracted,
                tables_found=tables_found,
                equations_found=equations_found,
                warnings=warnings
            )

        except Exception as e:
            error_msg = str(e)
            print(f"\n❌ Erreur lors de la conversion : {error_msg}")

            if self.config.get("debug"):
                print("🔍 Trace détaillée :")
                traceback.print_exc()

            return ConversionResult(
                success=False,
                error=error_msg,
                conversion_time=time.time() - start_time
            )

    def convert_batch(self, pdf_files: List[Path], output_dir: Optional[Path] = None) -> List[ConversionResult]:
        """
        Convertit plusieurs PDFs en lot

        Args:
            pdf_files: Liste des fichiers PDF à convertir
            output_dir: Dossier de sortie

        Returns:
            Liste des résultats de conversion
        """
        results = []
        total_files = len(pdf_files)

        print(f"\n🔄 Conversion en lot : {total_files} fichiers")

        for i, pdf_file in enumerate(pdf_files, 1):
            print(f"\n📄 [{i}/{total_files}] {pdf_file.name}")

            try:
                result = self.convert(pdf_file, output_dir)
                results.append(result)

                if result.success:
                    print(f"✅ [{i}/{total_files}] Succès")
                else:
                    print(f"❌ [{i}/{total_files}] Échec: {result.error}")

            except Exception as e:
                print(f"❌ [{i}/{total_files}] Erreur critique: {e}")
                results.append(ConversionResult(
                    success=False,
                    error=str(e)
                ))

        # Statistiques finales
        successful = sum(1 for r in results if r.success)
        print(f"\n📊 Résultats : {successful}/{total_files} conversions réussies")

        return results

def main():
    """Fonction principale"""
    parser = argparse.ArgumentParser(
        description="Convertir des PDFs en Markdown avec Marker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  # Conversion rapide
  python convert_to_markdown.py document.pdf

  # Mode qualité avec LLM
  python convert_to_markdown.py document.pdf --mode quality --provider gemini

  # Mode équilibré vers dossier spécifique
  python convert_to_markdown.py document.pdf --mode balanced --output ./results

  # Conversion en lot
  python convert_to_markdown.py *.pdf --batch --output ./batch_results

  # Avec limitation de pages
  python convert_to_markdown.py document.pdf --max-pages 10 --debug
        """
    )

    parser.add_argument(
        "pdf_path",
        type=Path,
        nargs="+",
        help="Chemin(s) vers le(s) fichier(s) PDF à convertir"
    )
    parser.add_argument(
        "--mode",
        choices=["fast", "balanced", "quality"],
        default="fast",
        help="Mode de conversion (défaut: fast)"
    )
    parser.add_argument(
        "--provider",
        choices=["auto", "gemini", "openai", "claude", "azure", "bedrock", "ollama"],
        default="auto",
        help="Provider LLM pour mode quality (défaut: auto)"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Dossier de sortie (défaut: ./outputs)"
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        help="Nombre maximum de pages à traiter"
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Mode traitement en lot pour plusieurs fichiers"
    )
    parser.add_argument(
        "--force-ocr",
        action="store_true",
        help="Forcer l'OCR (surcharge la configuration du mode)"
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Désactiver le LLM même en mode quality"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Activer le mode debug"
    )

    args = parser.parse_args()

    # Vérifier que les fichiers existent
    pdf_files = []
    for pdf_path in args.pdf_path:
        if pdf_path.is_file() and pdf_path.suffix.lower() == '.pdf':
            pdf_files.append(pdf_path)
        elif pdf_path.is_dir():
            # Chercher des PDFs dans le dossier
            pdf_files.extend(pdf_path.glob("*.pdf"))
        else:
            print(f"⚠️ Fichier ignoré (non trouvé ou pas un PDF): {pdf_path}")

    if not pdf_files:
        print("❌ Erreur : Aucun fichier PDF valide trouvé")
        sys.exit(1)

    # Configuration personnalisée
    custom_config = {}
    if args.force_ocr:
        custom_config["force_ocr"] = True
    if args.no_llm:
        custom_config["use_llm"] = False
    if args.debug:
        custom_config["debug"] = True

    # Créer le convertisseur
    try:
        converter = MarkdownConverter(
            mode=args.mode,
            custom_config=custom_config,
            llm_provider=args.provider
        )

        # Effectuer la conversion
        if len(pdf_files) > 1 or args.batch:
            # Conversion en lot
            results = converter.convert_batch(pdf_files, args.output)

            # Statistiques finales
            successful = sum(1 for r in results if r.success)
            total_time = sum(r.conversion_time for r in results)

            print(f"\n📊 Résultats finaux :")
            print(f"   Fichiers traités : {len(pdf_files)}")
            print(f"   Conversions réussies : {successful}")
            print(f"   Temps total : {total_time:.2f}s")

            success = successful == len(pdf_files)
        else:
            # Conversion simple
            result = converter.convert(
                pdf_files[0],
                args.output,
                max_pages=args.max_pages
            )
            success = result.success

        # Retourner le code de sortie approprié
        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        print("\n⚠️ Conversion interrompue par l'utilisateur")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Erreur : {e}")
        if args.debug:
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()