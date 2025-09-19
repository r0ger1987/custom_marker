# Guide d'Utilisation Avancé - Marker PDF

Ce guide vous accompagne dans l'utilisation complète du projet Marker pour comprendre et convertir des PDFs en markdown de manière optimale.

## 🎯 Objectifs et Cas d'Usage

Marker excelle dans plusieurs domaines :

- **📚 Documents Académiques** : Articles, thèses, rapports de recherche avec équations complexes
- **📊 Rapports Financiers** : États financiers, bilans avec tableaux détaillés
- **📋 Documents Administratifs** : Formulaires, contrats, documentation officielle
- **🏭 Manuels Techniques** : Documentation produit, guides d'installation
- **📰 Articles et Publications** : Journaux, magazines, newsletters

## 🚀 Démarrage Rapide

### 1. Configuration Initiale

```bash
# 1. Configurez votre environnement
cd custom_docs/
python scripts/setup_environment.py

# 2. Placez vos PDFs dans le dossier inputs/
cp votre_document.pdf inputs/

# 3. Configurez vos clés API (optionnel mais recommandé)
export GOOGLE_API_KEY="your_gemini_key_here"
# ou
export CLAUDE_API_KEY="your_claude_key_here"
# ou  
export OPENAI_API_KEY="your_openai_key_here"
```

### 2. Première Conversion

```bash
# Analyse préliminaire (recommandée)
python scripts/analyze_pdf_deep.py inputs/document.pdf --output outputs/

# Conversion selon les recommandations de l'analyse
python scripts/convert_to_markdown.py inputs/document.pdf --mode balanced --output outputs/
```

## 🔍 Workflow Complet d'Analyse et Conversion

### Étape 1 : Analyse Approfondie

L'analyse préliminaire vous aide à comprendre votre document et choisir la stratégie optimale :

```bash
# Analyse complète avec sauvegarde
python scripts/analyze_pdf_deep.py inputs/rapport_annuel.pdf \
  --output outputs/analyses/ \
  --debug

# Analyse avec LLM pour documents complexes
python scripts/analyze_pdf_deep.py inputs/article_scientifique.pdf \
  --llm \
  --output outputs/analyses/
```

**Interprétation des résultats :**

```json
{
  "structure_analysis": {
    "reading_complexity": "complex",        // Niveau de complexité
    "total_pages": 45,
    "content_distribution": {
      "Text": 120,                         // Blocs de texte standard
      "Table": 15,                         // Tableaux détectés
      "Equation": 8,                       // Équations mathématiques
      "Figure": 12                         // Images et graphiques
    }
  },
  "quality_analysis": {
    "confidence_score": 0.75,              // Score de confiance (0-1)
    "text_extraction_methods": {
      "pdftext": 35,                       // Pages avec texte extractible
      "ocr": 10                           // Pages nécessitant l'OCR
    }
  },
  "recommendations": [
    "Utiliser --use_llm pour une meilleure précision",
    "Document riche en tableaux - considérer TableConverter"
  ]
}
```

### Étape 2 : Choix de la Stratégie de Conversion

Basé sur l'analyse, choisissez votre approche :

#### Documents Simples (complexity: "simple" ou "moderate")
```bash
# Mode rapide pour tests et prototypage
python scripts/convert_to_markdown.py inputs/document.pdf \
  --mode fast \
  --output outputs/

# Mode équilibré pour usage quotidien  
python scripts/convert_to_markdown.py inputs/document.pdf \
  --mode balanced \
  --output outputs/
```

#### Documents Complexes (complexity: "complex" ou "very_complex")
```bash
# Mode qualité maximale avec LLM
python scripts/convert_to_markdown.py inputs/document.pdf \
  --mode quality \
  --output outputs/

# Conversion avec OCR forcé pour textes de mauvaise qualité
python scripts/convert_to_markdown.py inputs/document_scanné.pdf \
  --mode quality \
  --output outputs/
```

### Étape 3 : Extraction Spécialisée (Optionnel)

Pour des besoins spécifiques, extrayez des éléments particuliers :

#### Extraction de Tables
```bash
# Tables simples
python scripts/extract_specific_content.py inputs/rapport.pdf \
  --content-type tables \
  --output outputs/tables/

# Tables complexes avec LLM
python scripts/extract_specific_content.py inputs/rapport_financier.pdf \
  --content-type tables \
  --llm \
  --output outputs/tables/
```

#### Extraction d'Équations Mathématiques
```bash
# Articles scientifiques avec équations
python scripts/extract_specific_content.py inputs/paper.pdf \
  --content-type equations \
  --llm \
  --output outputs/equations/
```

#### Extraction d'Images et Figures
```bash
# Extraction complète avec métadonnées
python scripts/extract_specific_content.py inputs/manual.pdf \
  --content-type images \
  --output outputs/images/
```

#### Recherche par Sections/Mots-clés
```bash
# Extraction de sections spécifiques
python scripts/extract_specific_content.py inputs/these.pdf \
  --content-type sections \
  --keywords "introduction,methodology,results,conclusion,discussion" \
  --output outputs/sections/
```

## 📊 Traitement en Lot (Batch Processing)

Pour traiter de nombreux documents efficacement :

### Traitement Parallèle Standard
```bash
# Traitement avec détection automatique des workers
python scripts/batch_processor.py inputs/ outputs/ \
  --strategy parallel \
  --format markdown

# Contrôle fin du parallélisme
python scripts/batch_processor.py inputs/ outputs/ \
  --strategy parallel \
  --workers 4 \
  --llm
```

### Traitement Haute Performance
```bash
# Environnement multi-GPU
python scripts/batch_processor.py inputs/ outputs/ \
  --strategy multi_gpu \
  --llm \
  --format json

# Traitement conservateur pour gros documents
python scripts/batch_processor.py inputs/ outputs/ \
  --strategy sequential \
  --llm \
  --force-ocr
```

### Monitoring et Logs
```bash
# Surveillance en temps réel
tail -f batch_processing.log

# Vérification des ressources système
watch -n 2 'ps aux | grep python | grep marker'
```

## 🎛️ Optimisation des Performances

### Configuration Hardware

| Configuration | Recommandation | Workers | Mode |
|---------------|----------------|---------|------|
| **Basique** (CPU 4-core, 8GB RAM) | Documents simples | 2-3 | fast/balanced |
| **Standard** (CPU 8-core, 16GB RAM) | Usage quotidien | 4-6 | balanced/quality |
| **Avancée** (CPU 16-core, 32GB RAM, GPU) | Traitement intensif | 8-12 | quality |
| **Serveur** (CPU 32-core, 64GB RAM, GPU Pro) | Production | 16+ | quality |

### Variables d'Environnement Optimales

```bash
# Configuration GPU (si disponible)
export TORCH_DEVICE=cuda

# Optimisation mémoire
export TOKENIZERS_PARALLELISM=false

# Répertoires de travail
export OUTPUT_DIR=/path/to/your/outputs

# APIs pour mode LLM
export GOOGLE_API_KEY=your_key_here
```

### Monitoring des Ressources

```bash
# Installation d'outils de monitoring
pip install psutil gpustat

# Script de surveillance
python -c "
import psutil
import time
while True:
    cpu = psutil.cpu_percent(interval=1)
    mem = psutil.virtual_memory()
    print(f'CPU: {cpu}% | RAM: {mem.percent}% ({mem.used//1024**3}GB/{mem.total//1024**3}GB)')
    time.sleep(5)
"
```

## 📈 Exemples Concrets par Secteur

### 🏥 Secteur Médical
```bash
# Rapports médicaux avec terminologie complexe
python scripts/convert_to_markdown.py inputs/rapport_medical.pdf \
  --mode quality \
  --output outputs/medical/

# Extraction de tableaux de données cliniques
python scripts/extract_specific_content.py inputs/etude_clinique.pdf \
  --content-type tables \
  --llm \
  --output outputs/medical/tables/
```

### 💼 Finance et Comptabilité
```bash
# États financiers avec nombreux tableaux
python scripts/extract_specific_content.py inputs/bilan.pdf \
  --content-type tables \
  --llm \
  --output outputs/finance/

# Conversion complète de rapports annuels
python scripts/convert_to_markdown.py inputs/rapport_annuel.pdf \
  --mode quality \
  --output outputs/finance/
```

### 🔬 Recherche Scientifique
```bash
# Article avec équations et figures
python scripts/analyze_pdf_deep.py inputs/nature_paper.pdf --llm
python scripts/convert_to_markdown.py inputs/nature_paper.pdf --mode quality
python scripts/extract_specific_content.py inputs/nature_paper.pdf \
  --content-type equations --llm
```

### 🏛️ Juridique
```bash
# Documents contractuels
python scripts/convert_to_markdown.py inputs/contrat.pdf \
  --mode quality \
  --output outputs/legal/

# Extraction de clauses spécifiques
python scripts/extract_specific_content.py inputs/contrat.pdf \
  --content-type sections \
  --keywords "obligations,responsabilité,résiliation,pénalités"
```

## 🔧 Dépannage Avancé

### Problèmes de Mémoire
```bash
# Réduction des workers
python scripts/batch_processor.py inputs/ outputs/ --workers 1

# Mode séquentiel conservateur
python scripts/batch_processor.py inputs/ outputs/ --strategy sequential

# Monitoring mémoire en temps réel
python -c "
import psutil
import time
process = psutil.Process()
while True:
    mem_info = process.memory_info()
    print(f'RSS: {mem_info.rss//1024**2}MB | VMS: {mem_info.vms//1024**2}MB')
    time.sleep(10)
" &
```

### Problèmes GPU
```bash
# Vérification CUDA
nvidia-smi

# Force CPU si problème GPU
export TORCH_DEVICE=cpu
python scripts/convert_to_markdown.py inputs/document.pdf

# Nettoyage cache GPU
python -c "import torch; torch.cuda.empty_cache()"
```

### Problèmes API LLM
```bash
# Test de connectivité
python scripts/setup_environment.py --check-only

# Vérification des limites de rate
curl -H "Authorization: Bearer $OPENAI_API_KEY" \
  https://api.openai.com/v1/models

# Mode sans LLM en cas de problème
python scripts/convert_to_markdown.py inputs/document.pdf --mode balanced
```

### Debug Avancé
```bash
# Mode debug complet
python scripts/analyze_pdf_deep.py inputs/document.pdf \
  --debug \
  --output outputs/debug/

# Inspection des logs détaillés
tail -n 100 batch_processing.log | grep ERROR

# Analyse des performances
python -m cProfile -o profile_output.prof \
  scripts/convert_to_markdown.py inputs/document.pdf

# Visualisation du profile
pip install snakeviz
snakeviz profile_output.prof
```

## 📊 Métriques et Évaluation de Qualité

### Évaluation Automatique
```bash
# Comparaison avant/après traitement
python scripts/analyze_pdf_deep.py inputs/original.pdf --output outputs/analysis/
python scripts/convert_to_markdown.py inputs/original.pdf --mode quality --output outputs/
python scripts/analyze_pdf_deep.py outputs/original.md --output outputs/analysis/converted/
```

### Métriques de Performance
```bash
# Temps de traitement par page
grep "processing_time\|pages_processed" outputs/*/report.json

# Taux de réussite des conversions  
jq '.success_rate' outputs/batch_report_*.json

# Analyse de la distribution des types de contenu
jq '.structure_analysis.content_distribution' outputs/*_analysis.json
```

### Validation Qualitative
```bash
# Génération d'un aperçu HTML pour vérification
python -c "
import markdown
with open('outputs/document.md', 'r') as f:
    content = f.read()
html = markdown.markdown(content, extensions=['tables', 'fenced_code'])
with open('outputs/document.html', 'w') as f:
    f.write(html)
print('Aperçu HTML généré : outputs/document.html')
"
```

## 🔄 Workflows Automatisés

### Script de Traitement Complet
```bash
#!/bin/bash
# workflow_complet.sh

INPUT_DIR="inputs"
OUTPUT_DIR="outputs" 
ANALYSIS_DIR="$OUTPUT_DIR/analyses"

echo "🔍 Phase 1: Analyse des documents"
mkdir -p "$ANALYSIS_DIR"
for pdf in "$INPUT_DIR"/*.pdf; do
    echo "Analyse de $(basename "$pdf")"
    python scripts/analyze_pdf_deep.py "$pdf" \
        --output "$ANALYSIS_DIR" \
        --llm
done

echo "📄 Phase 2: Conversion optimisée"
python scripts/batch_processor.py "$INPUT_DIR" "$OUTPUT_DIR" \
    --strategy parallel \
    --llm \
    --format markdown

echo "📊 Phase 3: Génération du rapport final"
python -c "
import json
import glob
from pathlib import Path

analyses = []
for analysis_file in glob.glob('$ANALYSIS_DIR/*_analysis.json'):
    with open(analysis_file) as f:
        analyses.append(json.load(f))

print(f'📊 Résumé: {len(analyses)} documents analysés')
complexities = [a.get('structure_analysis', {}).get('reading_complexity') for a in analyses]
print(f'Complexités: {dict(zip(*zip(*[[c, complexities.count(c)] for c in set(complexities)])))}')
"

echo "✅ Workflow terminé!"
```

### Automatisation avec Cron
```bash
# Traitement automatique quotidien
# Ajouter dans crontab: crontab -e
# 0 2 * * * cd /path/to/marker/custom_docs && ./workflow_complet.sh >> logs/daily_processing.log 2>&1
```

Ce guide vous donne maintenant tous les outils pour maîtriser Marker et traiter efficacement vos documents PDF selon vos besoins spécifiques.

---

**💡 Conseils Pro :**
- Commencez toujours par une analyse préliminaire
- Utilisez le mode LLM pour les documents importants
- Surveillez vos ressources système lors du traitement en lot
- Gardez des copies de vos configurations optimales
- N'hésitez pas à combiner plusieurs approches selon le contexte