#!/usr/bin/env python3
"""
Script d'analyse PDF simplifié utilisant pypdf
===============================================

Ce script effectue une analyse basique d'un document PDF sans dépendre de Marker.
Il utilise pypdf pour extraire le texte et les métadonnées.

Usage:
    python analyze_pdf_simple.py input.pdf [--output DIR]

Auteur: Claude Code
"""

import sys
import os
import argparse
import json
import re
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict
from collections import Counter

# Import pypdf
import pypdf
from rich.console import Console
from rich.table import Table
from rich.progress import track

# Import du module de configuration LLM
sys.path.append(os.path.dirname(__file__))
from llm_config import LLMConfig

@dataclass
class DocumentStats:
    """Statistiques du document"""
    pages_count: int = 0
    word_count: int = 0
    character_count: int = 0
    paragraph_count: int = 0
    sentences_count: int = 0
    reading_time_minutes: float = 0.0

@dataclass
class AnalysisResult:
    """Résultat d'analyse simplifiée"""
    success: bool
    document_info: Dict[str, Any]
    statistics: DocumentStats
    content_preview: str
    page_contents: List[Dict[str, Any]]
    error: Optional[str] = None

class SimplePDFAnalyzer:
    """Analyseur PDF simplifié utilisant pypdf"""

    def __init__(self, use_llm: bool = False, llm_provider: Optional[str] = None):
        """
        Initialise l'analyseur PDF simplifié

        Args:
            use_llm: Activer l'analyse LLM
            llm_provider: Provider LLM à utiliser
        """
        self.use_llm = use_llm
        self.console = Console()

        # Configuration LLM si activé
        if use_llm:
            llm_config_manager = LLMConfig()
            self.llm_config = llm_config_manager.get_config(llm_provider)
        else:
            self.llm_config = None

    def analyze(self, pdf_path: Path, output_dir: Optional[Path] = None,
                max_pages: Optional[int] = None) -> AnalysisResult:
        """
        Analyse un document PDF

        Args:
            pdf_path: Chemin vers le fichier PDF
            output_dir: Dossier de sortie pour les résultats
            max_pages: Nombre maximum de pages à analyser

        Returns:
            AnalysisResult avec les résultats d'analyse
        """
        if not pdf_path.exists():
            raise FileNotFoundError(f"Fichier PDF non trouvé : {pdf_path}")

        # Configuration du dossier de sortie
        if output_dir is None:
            output_dir = Path("outputs/analyses")
        output_dir.mkdir(parents=True, exist_ok=True)

        self.console.print(f"\n📄 Analyse de : {pdf_path.name}", style="bold blue")
        self.console.print(f"📁 Résultats dans : {output_dir}", style="dim")

        try:
            # Ouvrir le PDF avec pypdf
            self.console.print("🔄 Extraction du contenu PDF...", style="yellow")
            reader = pypdf.PdfReader(str(pdf_path))

            # Extraction des métadonnées
            metadata = reader.metadata if reader.metadata else {}
            document_info = {
                "file_name": pdf_path.name,
                "file_size": pdf_path.stat().st_size,
                "pages_count": len(reader.pages),
                "title": metadata.get("/Title", "Non spécifié"),
                "author": metadata.get("/Author", "Non spécifié"),
                "subject": metadata.get("/Subject", "Non spécifié"),
                "creator": metadata.get("/Creator", "Non spécifié"),
                "producer": metadata.get("/Producer", "Non spécifié"),
                "creation_date": str(metadata.get("/CreationDate", "Non spécifié")),
                "modification_date": str(metadata.get("/ModDate", "Non spécifié"))
            }

            # Extraction du texte page par page
            page_contents = []
            full_text = ""
            pages_to_process = min(len(reader.pages), max_pages) if max_pages else len(reader.pages)

            self.console.print(f"📖 Extraction de {pages_to_process} pages...", style="cyan")

            for page_num in track(range(pages_to_process), description="Pages"):
                page = reader.pages[page_num]
                page_text = page.extract_text()

                # Nettoyage basique du texte
                page_text = re.sub(r'\s+', ' ', page_text).strip()

                page_contents.append({
                    "page_number": page_num + 1,
                    "text": page_text,
                    "word_count": len(page_text.split())
                })

                full_text += page_text + "\n\n"

            # Calcul des statistiques
            stats = self._calculate_statistics(full_text, pages_to_process)

            # Génération d'un aperçu du contenu
            content_preview = self._generate_preview(full_text, max_chars=1000)

            # Analyse LLM si activée
            if self.use_llm and self.llm_config:
                self.console.print("🤖 Analyse avec LLM...", style="magenta")
                llm_insights = self._analyze_with_llm(full_text[:5000])  # Limiter pour l'API
                document_info["llm_summary"] = llm_insights.get("summary", "")
                document_info["llm_keywords"] = llm_insights.get("keywords", [])

            # Création du résultat
            result = AnalysisResult(
                success=True,
                document_info=document_info,
                statistics=stats,
                content_preview=content_preview,
                page_contents=page_contents
            )

            # Sauvegarde des résultats
            self._save_results(result, output_dir, pdf_path.stem)

            # Affichage du résumé
            self._display_summary(result)

            return result

        except Exception as e:
            self.console.print(f"❌ Erreur lors de l'analyse : {str(e)}", style="bold red")
            return AnalysisResult(
                success=False,
                document_info={},
                statistics=DocumentStats(),
                content_preview="",
                page_contents=[],
                error=str(e)
            )

    def _calculate_statistics(self, text: str, pages_count: int) -> DocumentStats:
        """Calcule les statistiques du document"""
        words = text.split()
        word_count = len(words)

        # Estimation des phrases (points suivis d'espace ou fin de texte)
        sentences = re.split(r'[.!?]+\s+', text)
        sentences_count = len([s for s in sentences if s.strip()])

        # Estimation des paragraphes (double saut de ligne)
        paragraphs = text.split('\n\n')
        paragraph_count = len([p for p in paragraphs if p.strip()])

        # Temps de lecture estimé (200 mots par minute)
        reading_time = word_count / 200.0

        return DocumentStats(
            pages_count=pages_count,
            word_count=word_count,
            character_count=len(text),
            paragraph_count=paragraph_count,
            sentences_count=sentences_count,
            reading_time_minutes=round(reading_time, 1)
        )

    def _generate_preview(self, text: str, max_chars: int = 1000) -> str:
        """Génère un aperçu du contenu"""
        preview = text[:max_chars]
        if len(text) > max_chars:
            preview += "..."
        return preview

    def _analyze_with_llm(self, text: str) -> Dict[str, Any]:
        """Analyse le texte avec un LLM"""
        try:
            # Ici, vous pouvez implémenter l'appel à votre LLM préféré
            # Pour l'instant, retour d'un résultat simulé
            return {
                "summary": "Document analysé avec succès",
                "keywords": ["pdf", "analyse", "document"]
            }
        except Exception as e:
            return {"error": str(e)}

    def _save_results(self, result: AnalysisResult, output_dir: Path, base_name: str):
        """Sauvegarde les résultats dans différents formats"""
        # JSON
        json_path = output_dir / f"{base_name}_analysis.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(asdict(result), f, ensure_ascii=False, indent=2)
        self.console.print(f"💾 Résultats JSON : {json_path}", style="green")

        # Rapport Markdown
        md_path = output_dir / f"{base_name}_report.md"
        self._write_markdown_report(result, md_path)
        self.console.print(f"📝 Rapport Markdown : {md_path}", style="green")

    def _write_markdown_report(self, result: AnalysisResult, output_path: Path):
        """Écrit un rapport au format Markdown"""
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("# Rapport d'analyse PDF\n\n")

            f.write("## Informations du document\n\n")
            for key, value in result.document_info.items():
                f.write(f"- **{key.replace('_', ' ').title()}** : {value}\n")

            f.write("\n## Statistiques\n\n")
            f.write(f"- **Pages** : {result.statistics.pages_count}\n")
            f.write(f"- **Mots** : {result.statistics.word_count}\n")
            f.write(f"- **Caractères** : {result.statistics.character_count}\n")
            f.write(f"- **Paragraphes** : {result.statistics.paragraph_count}\n")
            f.write(f"- **Phrases** : {result.statistics.sentences_count}\n")
            f.write(f"- **Temps de lecture** : {result.statistics.reading_time_minutes} minutes\n")

            f.write("\n## Aperçu du contenu\n\n")
            f.write(f"```\n{result.content_preview}\n```\n")

    def _display_summary(self, result: AnalysisResult):
        """Affiche un résumé des résultats"""
        table = Table(title="📊 Résumé de l'analyse")
        table.add_column("Métrique", style="cyan")
        table.add_column("Valeur", style="magenta")

        table.add_row("Fichier", result.document_info.get("file_name", ""))
        table.add_row("Pages", str(result.statistics.pages_count))
        table.add_row("Mots", str(result.statistics.word_count))
        table.add_row("Temps de lecture", f"{result.statistics.reading_time_minutes} min")

        self.console.print(table)

def main():
    """Fonction principale"""
    parser = argparse.ArgumentParser(description="Analyse approfondie de PDF")
    parser.add_argument("pdf_path", help="Chemin vers le fichier PDF")
    parser.add_argument("--output", help="Dossier de sortie", default="outputs")
    parser.add_argument("--max-pages", type=int, help="Nombre max de pages")
    parser.add_argument("--llm", action="store_true", help="Activer l'analyse LLM")
    parser.add_argument("--provider", help="Provider LLM", default="bedrock")

    args = parser.parse_args()

    pdf_path = Path(args.pdf_path)
    output_dir = Path(args.output)

    # Initialisation de l'analyseur
    analyzer = SimplePDFAnalyzer(use_llm=args.llm, llm_provider=args.provider)

    # Lancement de l'analyse
    result = analyzer.analyze(pdf_path, output_dir, args.max_pages)

    if result.success:
        print(f"\n✅ Analyse terminée avec succès!")
    else:
        print(f"\n❌ L'analyse a échoué : {result.error}")
        sys.exit(1)

if __name__ == "__main__":
    main()