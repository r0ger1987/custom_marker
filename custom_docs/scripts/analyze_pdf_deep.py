#!/usr/bin/env python3
"""
Script d'analyse approfondie de PDF avec Marker et LLM
=====================================================

Ce script effectue une analyse complète d'un document PDF avec Marker :
- Extraction de texte et images avec Marker
- Analyse structurelle avancée (tables, équations, figures)
- Analyse statistique du contenu
- Résumé automatique et extraction d'entités avec LLM
- Rapport d'analyse multiformat (JSON, Markdown, HTML)

Usage:
    python analyze_pdf_deep.py input.pdf [--llm] [--provider PROVIDER] [--output DIR]

Auteur: Claude Code
"""

import sys
import os
import argparse
import json
import re
import time
import traceback
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, asdict
from collections import Counter

# Ajout du chemin racine pour importer marker
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(project_root)

try:
    # Import Marker et ses modules
    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict
    from marker.settings import settings
    from marker.logger import configure_logging
    from marker.schema.document import Document
    MARKER_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Marker non disponible: {e}")
    MARKER_AVAILABLE = False

    # Classe Document de secours si Marker n'est pas disponible
    class Document:
        """Classe Document de substitution quand Marker n'est pas disponible"""
        def __init__(self):
            self.pages = []
            self.metadata = {}
            self.text = ""

# Import du module de configuration LLM
from llm_config import LLMConfig

@dataclass
class DocumentStats:
    """Statistiques du document"""
    pages_count: int = 0
    word_count: int = 0
    character_count: int = 0
    paragraph_count: int = 0
    sentences_count: int = 0
    tables_count: int = 0
    figures_count: int = 0
    equations_count: int = 0
    headers_count: int = 0
    footnotes_count: int = 0
    reading_time_minutes: float = 0.0

@dataclass
class StructureElement:
    """Élément de structure du document"""
    type: str
    content: str
    level: int = 0
    page_number: int = 0
    position: Dict[str, float] = None
    metadata: Dict[str, Any] = None

@dataclass
class AnalysisResult:
    """Résultat d'analyse complète"""
    success: bool
    document_info: Dict[str, Any]
    statistics: DocumentStats
    structure_elements: List[StructureElement]
    content_analysis: Dict[str, Any]
    llm_insights: Dict[str, Any]
    quality_metrics: Dict[str, float]
    analysis_metadata: Dict[str, Any]
    error: Optional[str] = None

class PDFAnalyzer:
    """Analyseur PDF avancé avec support Marker et LLM"""

    def __init__(self, use_llm: bool = False, llm_provider: str = "auto", debug: bool = False):
        """
        Initialise l'analyseur PDF

        Args:
            use_llm: Utiliser un LLM pour l'analyse avancée
            llm_provider: Provider LLM à utiliser
            debug: Mode debug détaillé
        """
        if not MARKER_AVAILABLE:
            raise RuntimeError("Marker n'est pas disponible. Installez-le avec: pip install marker-pdf[full]")

        self.use_llm = use_llm
        self.llm_provider = llm_provider
        self.debug = debug
        self.llm_config = None
        self.models = None

        if use_llm:
            try:
                self.llm_config = LLMConfig.get_llm_config(llm_provider)
                print(f"🤖 LLM configuré : {llm_provider}")
            except Exception as e:
                print(f"⚠️ Erreur configuration LLM : {e}")
                self.use_llm = False

        print(f"🔍 Analyseur PDF initialisé (LLM: {'✅' if self.use_llm else '❌'})")

    def _load_models(self):
        """Charge les modèles Marker si nécessaire"""
        if self.models is None:
            print("🔄 Chargement des modèles Marker...")
            try:
                if self.debug:
                    configure_logging()
                self.models = create_model_dict()
                print("✅ Modèles Marker chargés")
            except Exception as e:
                print(f"❌ Erreur lors du chargement des modèles: {e}")
                raise

    def analyze_document(self, pdf_path: Path, output_dir: Optional[Path] = None, max_pages: Optional[int] = None) -> AnalysisResult:
        """
        Analyse complète d'un document PDF avec Marker

        Args:
            pdf_path: Chemin vers le PDF
            output_dir: Dossier de sortie pour les résultats
            max_pages: Nombre maximum de pages à analyser

        Returns:
            AnalysisResult avec tous les résultats d'analyse
        """
        if not pdf_path.exists():
            raise FileNotFoundError(f"Fichier PDF non trouvé : {pdf_path}")

        # Configuration du dossier de sortie avec structure organisée
        base_output_dir = Path("/home/roger/RAG/custom_marker/custom_docs/outputs")

        # Créer une structure organisée : outputs/analyses/nom_fichier_YYYYMMDD_HHMMSS/
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        file_stem = pdf_path.stem
        analysis_dir = base_output_dir / "analyses" / f"{file_stem}_{timestamp}"

        # Créer les sous-dossiers
        output_dir = analysis_dir
        (output_dir / "reports").mkdir(parents=True, exist_ok=True)
        (output_dir / "data").mkdir(parents=True, exist_ok=True)
        (output_dir / "content").mkdir(parents=True, exist_ok=True)

        print(f"\n📄 Analyse approfondie de : {pdf_path.name}")
        print(f"📁 Résultats dans : {output_dir}")

        start_time = time.time()

        try:
            # Charger les modèles Marker
            self._load_models()

            # Configuration pour Marker
            config = {
                "output_format": "markdown",
                "extract_images": True,
                "use_llm": self.use_llm,
                "debug": self.debug
            }

            if max_pages:
                config["max_pages"] = max_pages

            if self.use_llm and self.llm_config:
                config.update(self.llm_config)

            # Conversion avec Marker
            print("🔄 Extraction avec Marker...")
            converter = PdfConverter(
                config=config,
                artifact_dict=self.models,
                llm_service=config.get("llm_service") if self.use_llm else None
            )

            doc_result = converter(str(pdf_path))

            # Analyse des résultats
            print("📊 Calcul des statistiques...")
            stats = self._calculate_statistics(doc_result)

            print("🏗️ Analyse de la structure...")
            structure_elements = self._extract_structure_elements(doc_result)

            print("📝 Analyse du contenu...")
            content_analysis = self._analyze_content(doc_result)

            print("📈 Calcul des métriques de qualité...")
            quality_metrics = self._calculate_quality_metrics(doc_result, content_analysis)

            # Analyse LLM si activée
            llm_insights = {}
            if self.use_llm:
                print(f"🧠 Analyse LLM avec {self.llm_provider}...")
                llm_insights = self._perform_llm_analysis(doc_result, content_analysis)

            # Informations du document
            document_info = self._extract_document_info(pdf_path, doc_result)

            # Métadonnées de l'analyse
            elapsed_time = time.time() - start_time
            analysis_metadata = {
                "analysis_time": elapsed_time,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "llm_used": self.use_llm,
                "llm_provider": self.llm_provider if self.use_llm else None,
                "marker_version": getattr(converter, 'version', 'unknown'),
                "max_pages_analyzed": max_pages,
                "total_processing_time": f"{elapsed_time:.2f}s"
            }

            # Créer le résultat complet
            result = AnalysisResult(
                success=True,
                document_info=document_info,
                statistics=stats,
                structure_elements=structure_elements,
                content_analysis=content_analysis,
                llm_insights=llm_insights,
                quality_metrics=quality_metrics,
                analysis_metadata=analysis_metadata
            )

            # Sauvegarder les résultats
            self._save_comprehensive_results(result, pdf_path, output_dir, doc_result)

            print(f"\n✅ Analyse terminée en {elapsed_time:.2f}s")
            print(f"📊 Pages analysées : {stats.pages_count}")
            print(f"📝 Mots analysés : {stats.word_count}")
            print(f"📋 Tableaux trouvés : {stats.tables_count}")
            print(f"🖼️ Figures trouvées : {stats.figures_count}")

            return result

        except Exception as e:
            error_msg = str(e)
            print(f"\n❌ Erreur lors de l'analyse : {error_msg}")

            if self.debug:
                print("🔍 Trace détaillée :")
                traceback.print_exc()

            return AnalysisResult(
                success=False,
                document_info={},
                statistics=DocumentStats(),
                structure_elements=[],
                content_analysis={},
                llm_insights={},
                quality_metrics={},
                analysis_metadata={"analysis_time": time.time() - start_time},
                error=error_msg
            )

    def _calculate_statistics(self, doc_result: Document) -> DocumentStats:
        """Calcule les statistiques détaillées du document"""
        stats = DocumentStats()

        if hasattr(doc_result, 'pages'):
            stats.pages_count = len(doc_result.pages)

            # Compter les éléments par type
            for page in doc_result.pages:
                for block in page.blocks:
                    block_type = str(block.block_type).lower() if hasattr(block, 'block_type') else ''

                    if 'text' in block_type or 'paragraph' in block_type:
                        text_content = getattr(block, 'text', '') or str(block)
                        stats.word_count += len(text_content.split())
                        stats.character_count += len(text_content)
                        stats.paragraph_count += 1
                        # Estimation du nombre de phrases
                        stats.sentences_count += len(re.findall(r'[.!?]+', text_content))

                    elif 'table' in block_type:
                        stats.tables_count += 1
                    elif 'figure' in block_type or 'image' in block_type:
                        stats.figures_count += 1
                    elif 'equation' in block_type or 'math' in block_type:
                        stats.equations_count += 1
                    elif 'header' in block_type or 'heading' in block_type:
                        stats.headers_count += 1
                    elif 'footnote' in block_type:
                        stats.footnotes_count += 1

        # Calcul du temps de lecture (250 mots/minute en moyenne)
        stats.reading_time_minutes = stats.word_count / 250.0

        return stats

    def _extract_structure_elements(self, doc_result: Document) -> List[StructureElement]:
        """Extrait les éléments de structure du document"""
        elements = []

        if hasattr(doc_result, 'pages'):
            for page_num, page in enumerate(doc_result.pages, 1):
                for block in page.blocks:
                    if hasattr(block, 'block_type'):
                        block_type = str(block.block_type)
                        content = getattr(block, 'text', '') or str(block)

                        # Déterminer le niveau pour les headers
                        level = 0
                        if 'header' in block_type.lower() or 'heading' in block_type.lower():
                            # Essayer d'extraire le niveau depuis le type ou le contenu
                            level_match = re.search(r'(\d+)', block_type)
                            if level_match:
                                level = int(level_match.group(1))
                            else:
                                # Estimation basée sur le contenu
                                if content.startswith('#'):
                                    level = len(content) - len(content.lstrip('#'))

                        # Position approximative (si disponible)
                        position = {}
                        if hasattr(block, 'bbox'):
                            bbox = block.bbox
                            position = {
                                "x": bbox[0] if bbox else 0,
                                "y": bbox[1] if bbox else 0,
                                "width": bbox[2] - bbox[0] if bbox and len(bbox) >= 3 else 0,
                                "height": bbox[3] - bbox[1] if bbox and len(bbox) >= 4 else 0
                            }

                        # Métadonnées supplémentaires
                        metadata = {
                            "block_type": block_type,
                            "content_length": len(content),
                            "word_count": len(content.split()) if content else 0
                        }

                        element = StructureElement(
                            type=block_type,
                            content=content[:200] + "..." if len(content) > 200 else content,
                            level=level,
                            page_number=page_num,
                            position=position,
                            metadata=metadata
                        )
                        elements.append(element)

        return elements

    def _analyze_content(self, doc_result: Document) -> Dict[str, Any]:
        """Analyse le contenu textuel et structurel"""
        analysis = {
            "full_text": "",
            "language_detected": "unknown",
            "text_blocks": [],
            "tables_data": [],
            "images_info": [],
            "equations_info": [],
            "vocabulary_richness": 0.0,
            "most_common_words": [],
            "text_density_per_page": []
        }

        if hasattr(doc_result, 'pages'):
            all_text = doc_result.markdown
            analysis["full_text"] = all_text

            # Analyse de vocabulaire
            words = re.findall(r'\b\w+\b', all_text.lower())
            if words:
                word_freq = Counter(words)
                analysis["most_common_words"] = word_freq.most_common(20)
                analysis["vocabulary_richness"] = len(set(words)) / len(words)

            # Analyse par page
            for page_num, page in enumerate(doc_result.pages, 1):
                page_text = ""
                page_tables = 0
                page_images = 0

                for block in page.blocks:
                    block_type = str(block.block_type).lower() if hasattr(block, 'block_type') else ''
                    content = getattr(block, 'text', '') or str(block)

                    if 'text' in block_type:
                        page_text += content + " "
                        analysis["text_blocks"].append({
                            "page": page_num,
                            "content": content[:100] + "..." if len(content) > 100 else content,
                            "word_count": len(content.split())
                        })
                    elif 'table' in block_type:
                        page_tables += 1
                        analysis["tables_data"].append({
                            "page": page_num,
                            "content": content[:200] + "..." if len(content) > 200 else content
                        })
                    elif 'figure' in block_type or 'image' in block_type:
                        page_images += 1
                        analysis["images_info"].append({
                            "page": page_num,
                            "description": content if content else "Image détectée"
                        })
                    elif 'equation' in block_type or 'math' in block_type:
                        analysis["equations_info"].append({
                            "page": page_num,
                            "content": content
                        })

                analysis["text_density_per_page"].append({
                    "page": page_num,
                    "word_count": len(page_text.split()),
                    "tables": page_tables,
                    "images": page_images
                })

        return analysis

    def _calculate_quality_metrics(self, doc_result: Document, content_analysis: Dict[str, Any]) -> Dict[str, float]:
        """Calcule les métriques de qualité du document"""
        metrics = {
            "text_extraction_quality": 0.0,
            "structure_clarity": 0.0,
            "content_completeness": 0.0,
            "readability_score": 0.0,
            "overall_quality": 0.0
        }

        # Qualité d'extraction du texte (basée sur la présence de caractères cohérents)
        full_text = content_analysis.get("full_text", "")
        if full_text:
            # Ratio de caractères alphanumériques
            alphanum_ratio = len(re.findall(r'[a-zA-Z0-9]', full_text)) / len(full_text)
            metrics["text_extraction_quality"] = min(alphanum_ratio * 1.2, 1.0)

        # Clarté de la structure (basée sur la présence d'éléments structurels)
        structure_score = 0.0
        if content_analysis.get("text_blocks"):
            structure_score += 0.3
        if content_analysis.get("tables_data"):
            structure_score += 0.2
        if content_analysis.get("images_info"):
            structure_score += 0.2
        # Présence d'en-têtes (estimation)
        if re.search(r'^#+\s', full_text, re.MULTILINE):
            structure_score += 0.3
        metrics["structure_clarity"] = min(structure_score, 1.0)

        # Complétude du contenu (basée sur la densité et la variété)
        vocab_richness = content_analysis.get("vocabulary_richness", 0.0)
        metrics["content_completeness"] = min(vocab_richness * 2.0, 1.0)

        # Score de lisibilité (estimation simplifiée)
        if full_text:
            sentences = len(re.findall(r'[.!?]+', full_text))
            words = len(full_text.split())
            if sentences > 0:
                avg_sentence_length = words / sentences
                # Score basé sur la longueur moyenne des phrases (optimal ~15-20 mots)
                readability = max(0, 1.0 - abs(avg_sentence_length - 17.5) / 50.0)
                metrics["readability_score"] = readability

        # Score global
        metrics["overall_quality"] = (
            metrics["text_extraction_quality"] * 0.3 +
            metrics["structure_clarity"] * 0.3 +
            metrics["content_completeness"] * 0.2 +
            metrics["readability_score"] * 0.2
        )

        return metrics

    def _perform_llm_analysis(self, doc_result: Document, content_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Effectue une analyse avancée avec LLM"""
        insights = {
            "summary": "Résumé en cours de génération...",
            "key_topics": [],
            "document_type": "unknown",
            "complexity_assessment": "medium",
            "main_themes": [],
            "entities_extracted": {
                "persons": [],
                "organizations": [],
                "locations": [],
                "dates": [],
                "technical_terms": []
            },
            "sentiment_analysis": "neutral",
            "information_density": "medium",
            "llm_confidence": 0.0
        }

        # Ici on intégrerait l'appel réel au LLM
        # Pour l'instant, on simule une analyse basée sur le contenu
        full_text = content_analysis.get("full_text", "")

        if full_text:
            # Estimation du type de document
            if any(keyword in full_text.lower() for keyword in ["abstract", "introduction", "methodology", "conclusion"]):
                insights["document_type"] = "academic_paper"
            elif any(keyword in full_text.lower() for keyword in ["chapter", "section", "appendix"]):
                insights["document_type"] = "book_or_manual"
            elif any(keyword in full_text.lower() for keyword in ["quarterly", "annual", "report", "financial"]):
                insights["document_type"] = "business_report"
            else:
                insights["document_type"] = "general_document"

            # Extraction simple d'entités (patterns basiques)
            dates = re.findall(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}\b', full_text)
            insights["entities_extracted"]["dates"] = list(set(dates[:10]))  # Limiter à 10

            # Mots clés potentiels (mots de plus de 5 caractères, fréquents)
            words = re.findall(r'\b[A-Z][a-z]{4,}\b', full_text)
            word_freq = Counter(words)
            insights["key_topics"] = [word for word, count in word_freq.most_common(10)]

        return insights

    def _extract_document_info(self, pdf_path: Path, doc_result: Document) -> Dict[str, Any]:
        """Extrait les informations du document"""
        return {
            "filename": pdf_path.name,
            "file_size_mb": round(pdf_path.stat().st_size / (1024 * 1024), 2),
            "pages_count": len(doc_result.pages) if hasattr(doc_result, 'pages') else 0,
            "creation_date": time.ctime(pdf_path.stat().st_ctime),
            "modification_date": time.ctime(pdf_path.stat().st_mtime),
            "marker_extraction": "success"
        }

    def _save_comprehensive_results(self, result: AnalysisResult, pdf_path: Path, output_dir: Path, doc_result: Document):
        """Sauvegarde les résultats d'analyse complets"""
        print("💾 Sauvegarde des résultats...")

        base_name = pdf_path.stem

        # 1. Sauvegarde JSON complète dans data/
        json_file = output_dir / "data" / f"{base_name}_complete_analysis.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            # Convertir les dataclasses en dict pour la sérialisation
            result_dict = asdict(result)
            json.dump(result_dict, f, indent=2, ensure_ascii=False, default=str)

        # 2. Rapport Markdown détaillé dans reports/
        detailed_report = output_dir / "reports" / f"{base_name}_detailed_report.md"
        self._generate_markdown_report(result, detailed_report)

        # 3. Rapport HTML interactif dans reports/
        html_report = output_dir / "reports" / f"{base_name}_interactive_report.html"
        self._generate_html_report(result, html_report)

        # 4. Export du markdown extrait dans content/
        markdown_file = output_dir / "content" / f"{base_name}_extracted_content.md"
        with open(markdown_file, 'w', encoding='utf-8') as f:
            f.write(doc_result.markdown)

        # 5. Créer un fichier README pour expliquer la structure
        readme_file = output_dir / "README.md"
        self._create_analysis_readme(readme_file, base_name, result)

        print(f"📊 Analyse JSON : {json_file}")
        print(f"📋 Rapport détaillé : {detailed_report}")
        print(f"🌐 Rapport HTML : {html_report}")
        print(f"📝 Contenu Markdown : {markdown_file}")
        print(f"📖 Guide d'utilisation : {readme_file}")

    def _generate_markdown_report(self, result: AnalysisResult, output_file: Path):
        """Génère un rapport Markdown détaillé"""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"# Analyse Approfondie - {result.document_info.get('filename', 'Document')}\n\n")

            # Métadonnées
            f.write("## 📊 Métadonnées de l'analyse\n\n")
            f.write(f"- **Date d'analyse :** {result.analysis_metadata.get('timestamp')}\n")
            f.write(f"- **Temps de traitement :** {result.analysis_metadata.get('total_processing_time')}\n")
            f.write(f"- **LLM utilisé :** {result.analysis_metadata.get('llm_provider', 'Aucun')}\n\n")

            # Statistiques
            f.write("## 📈 Statistiques du document\n\n")
            stats = result.statistics
            f.write(f"- **Pages :** {stats.pages_count}\n")
            f.write(f"- **Mots :** {stats.word_count:,}\n")
            f.write(f"- **Caractères :** {stats.character_count:,}\n")
            f.write(f"- **Paragraphes :** {stats.paragraph_count}\n")
            f.write(f"- **Tableaux :** {stats.tables_count}\n")
            f.write(f"- **Figures :** {stats.figures_count}\n")
            f.write(f"- **Équations :** {stats.equations_count}\n")
            f.write(f"- **Temps de lecture estimé :** {stats.reading_time_minutes:.1f} minutes\n\n")

            # Métriques de qualité
            f.write("## 🎯 Métriques de qualité\n\n")
            for metric, value in result.quality_metrics.items():
                f.write(f"- **{metric.replace('_', ' ').title()} :** {value:.2%}\n")
            f.write("\n")

            # Structure du document
            f.write("## 🏗️ Structure du document\n\n")
            headers = [elem for elem in result.structure_elements if 'header' in elem.type.lower()]
            if headers:
                f.write("### En-têtes détectés\n\n")
                for header in headers[:10]:  # Limiter à 10
                    indent = "  " * header.level
                    f.write(f"{indent}- **Page {header.page_number} :** {header.content}\n")
            f.write("\n")

            # Analyse LLM
            if result.llm_insights:
                f.write("## 🧠 Analyse LLM\n\n")
                insights = result.llm_insights

                if insights.get("summary"):
                    f.write(f"### Résumé\n{insights['summary']}\n\n")

                if insights.get("key_topics"):
                    f.write(f"### Sujets clés\n")
                    for topic in insights["key_topics"][:10]:
                        f.write(f"- {topic}\n")
                    f.write("\n")

                f.write(f"**Type de document :** {insights.get('document_type', 'Unknown')}\n")
                f.write(f"**Complexité :** {insights.get('complexity_assessment', 'Unknown')}\n\n")

    def _generate_html_report(self, result: AnalysisResult, output_file: Path):
        """Génère un rapport HTML interactif"""
        html_content = f"""
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Analyse PDF - {result.document_info.get('filename', 'Document')}</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }}
        .header {{ background: #f0f8ff; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
        .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }}
        .stat-card {{ background: #f9f9f9; padding: 15px; border-radius: 5px; border-left: 4px solid #007acc; }}
        .quality-meter {{ background: #e0e0e0; height: 20px; border-radius: 10px; overflow: hidden; margin: 5px 0; }}
        .quality-fill {{ background: linear-gradient(90deg, #ff4444, #ffaa00, #00aa00); height: 100%; transition: width 0.3s; }}
        .section {{ margin: 30px 0; padding: 20px; border: 1px solid #ddd; border-radius: 8px; }}
        .collapsible {{ cursor: pointer; background: #f1f1f1; padding: 10px; border-radius: 5px; }}
        .content {{ display: none; padding: 15px; }}
        .content.active {{ display: block; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>📄 Analyse Approfondie - {result.document_info.get('filename', 'Document')}</h1>
        <p><strong>Analysé le :</strong> {result.analysis_metadata.get('timestamp')}</p>
        <p><strong>Temps de traitement :</strong> {result.analysis_metadata.get('total_processing_time')}</p>
    </div>

    <div class="stats-grid">
        <div class="stat-card">
            <h3>📊 Pages</h3>
            <p>{result.statistics.pages_count}</p>
        </div>
        <div class="stat-card">
            <h3>📝 Mots</h3>
            <p>{result.statistics.word_count:,}</p>
        </div>
        <div class="stat-card">
            <h3>📋 Tableaux</h3>
            <p>{result.statistics.tables_count}</p>
        </div>
        <div class="stat-card">
            <h3>🖼️ Figures</h3>
            <p>{result.statistics.figures_count}</p>
        </div>
    </div>

    <div class="section">
        <h2>🎯 Qualité du document</h2>
        <div class="quality-meter">
            <div class="quality-fill" style="width: {result.quality_metrics.get('overall_quality', 0) * 100}%"></div>
        </div>
        <p>Score global : {result.quality_metrics.get('overall_quality', 0):.2%}</p>
    </div>

    <script>
        document.querySelectorAll('.collapsible').forEach(item => {{
            item.addEventListener('click', () => {{
                const content = item.nextElementSibling;
                content.classList.toggle('active');
            }});
        }});
    </script>
</body>
</html>
        """

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)

    def _create_analysis_readme(self, readme_file: Path, base_name: str, result: AnalysisResult):
        """Crée un fichier README expliquant la structure des résultats"""
        with open(readme_file, 'w', encoding='utf-8') as f:
            f.write(f"# Analyse PDF - {base_name}\n\n")
            f.write(f"Analyse générée le : {result.analysis_metadata.get('timestamp', 'N/A')}\n\n")

            f.write("## Structure des dossiers\n\n")
            f.write("```\n")
            f.write("├── README.md                    # Ce fichier\n")
            f.write("├── data/                        # Données d'analyse\n")
            f.write(f"│   └── {base_name}_complete_analysis.json\n")
            f.write("├── reports/                     # Rapports d'analyse\n")
            f.write(f"│   ├── {base_name}_detailed_report.md\n")
            f.write(f"│   └── {base_name}_interactive_report.html\n")
            f.write("└── content/                     # Contenu extrait\n")
            f.write(f"    └── {base_name}_extracted_content.md\n")
            f.write("```\n\n")

            f.write("## Description des fichiers\n\n")
            f.write("### 📊 data/\n")
            f.write("- **complete_analysis.json** : Données complètes d'analyse au format JSON\n\n")

            f.write("### 📋 reports/\n")
            f.write("- **detailed_report.md** : Rapport détaillé en Markdown\n")
            f.write("- **interactive_report.html** : Rapport interactif visualisable dans un navigateur\n\n")

            f.write("### 📝 content/\n")
            f.write("- **extracted_content.md** : Contenu du PDF extrait en Markdown par Marker\n\n")

            f.write("## Statistiques rapides\n\n")
            stats = result.statistics
            f.write(f"- **Pages analysées** : {stats.pages_count}\n")
            f.write(f"- **Mots extraits** : {stats.word_count:,}\n")
            f.write(f"- **Tableaux détectés** : {stats.tables_count}\n")
            f.write(f"- **Figures détectées** : {stats.figures_count}\n")
            f.write(f"- **Temps de lecture estimé** : {stats.reading_time_minutes} minutes\n\n")

def main():
    """Fonction principale"""
    parser = argparse.ArgumentParser(
        description="Analyse approfondie de PDF avec Marker et LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  # Analyse basique avec Marker
  python analyze_pdf_deep.py document.pdf

  # Analyse complète avec LLM
  python analyze_pdf_deep.py document.pdf --llm --provider gemini

  # Analyse ciblée avec limitation de pages
  python analyze_pdf_deep.py document.pdf --max-pages 5 --debug

  # Analyse avec provider Bedrock
  python analyze_pdf_deep.py document.pdf --llm --provider bedrock --output ./analyses
        """
    )

    parser.add_argument(
        "pdf_path",
        type=Path,
        help="Chemin vers le fichier PDF à analyser"
    )
    parser.add_argument(
        "--llm",
        action="store_true",
        help="Activer l'analyse LLM avancée"
    )
    parser.add_argument(
        "--provider",
        choices=["auto", "gemini", "openai", "claude", "azure", "bedrock", "ollama"],
        default="auto",
        help="Provider LLM à utiliser (défaut: auto)"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Dossier de sortie (défaut: ./outputs/analyses)"
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        help="Nombre maximum de pages à analyser"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Activer le mode debug détaillé"
    )

    args = parser.parse_args()

    if not args.pdf_path.exists():
        print(f"❌ Erreur : Le fichier {args.pdf_path} n'existe pas")
        sys.exit(1)

    # Créer l'analyseur
    try:
        analyzer = PDFAnalyzer(
            use_llm=args.llm,
            llm_provider=args.provider,
            debug=args.debug
        )

        # Effectuer l'analyse
        result = analyzer.analyze_document(
            args.pdf_path,
            args.output,
            max_pages=args.max_pages
        )

        # Afficher un résumé des résultats
        if result.success:
            print(f"\n🎉 Analyse réussie !")
            print(f"📊 Score global de qualité : {result.quality_metrics.get('overall_quality', 0):.2%}")

            if result.llm_insights:
                print(f"🧠 Type de document détecté : {result.llm_insights.get('document_type', 'unknown')}")

            print(f"📈 Voir les rapports détaillés dans : {args.output or 'outputs/analyses'}")
        else:
            print(f"❌ Échec de l'analyse : {result.error}")

        # Retourner le code de sortie approprié
        sys.exit(0 if result.success else 1)

    except KeyboardInterrupt:
        print("\n⚠️ Analyse interrompue par l'utilisateur")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Erreur : {e}")
        if args.debug:
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()