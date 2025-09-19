# Guide des Providers LLM pour Marker

Ce guide explique comment utiliser les différents providers LLM avec les scripts custom_docs de Marker.

## 🔧 Configuration automatique

Tous les scripts utilisent maintenant le fichier `custom_docs/.env` pour charger automatiquement les clés API. **Plus besoin d'exporter manuellement les variables d'environnement !**

### 1. Configurez votre fichier .env

Le fichier `custom_docs/.env` contient vos clés API :

```bash
# Google Gemini API (recommandé)
GOOGLE_API_KEY=AIzaSyXXXXXXXXXXXXXXXXXXXXXXXXX

# Claude API (Anthropic) 
CLAUDE_API_KEY=sk-ant-api03-XXXXXXXXXXXXXXXXXXXXXXX

# OpenAI API
OPENAI_API_KEY=sk-XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

# Azure OpenAI (optionnel)
AZURE_API_KEY=your_azure_api_key
AZURE_ENDPOINT=your_azure_endpoint

# Configuration Marker
TORCH_DEVICE=cuda  # auto, cuda, ou cpu
OUTPUT_DIR=./outputs
```

### 2. Providers disponibles

| Provider | Statut | Installation requise | Qualité | Vitesse |
|----------|--------|---------------------|---------|---------|
| **gemini** | ✅ Recommandé | ✅ Pré-installé | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **bedrock** | ✅ Claude 3.5 | `pip install boto3` | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **claude** | ⚠️ Package requis | `pip install anthropic` | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **openai** | ⚠️ Package requis | `pip install openai` | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **azure** | ⚠️ Package requis | `pip install openai` | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **ollama** | 🔧 Installation locale | Installation Ollama | ⭐⭐⭐ | ⭐⭐ |

## 📋 Utilisation des scripts

### analyze_pdf_deep.py

Analyse approfondie de PDF avec support multi-LLM :

```bash
# Utilisation automatique du meilleur provider disponible
python custom_docs/scripts/analyze_pdf_deep.py document.pdf --llm --output outputs/

# Forcer un provider spécifique
python custom_docs/scripts/analyze_pdf_deep.py document.pdf --llm --provider gemini --output outputs/
python custom_docs/scripts/analyze_pdf_deep.py document.pdf --llm --provider claude --output outputs/
```

**Options :**
- `--llm` : Activer le mode LLM
- `--provider` : Choix du provider (`auto`, `gemini`, `claude`, `openai`, `azure`, `ollama`)
- `--debug` : Mode debug détaillé

### convert_to_markdown.py

Conversion PDF vers Markdown avec LLM :

```bash
# Mode qualité avec LLM automatique
python custom_docs/scripts/convert_to_markdown.py document.pdf --mode quality --output outputs/

# Mode qualité avec provider spécifique
python custom_docs/scripts/convert_to_markdown.py document.pdf --mode quality --provider gemini --output outputs/
```

**Modes :**
- `fast` : Conversion rapide sans LLM
- `balanced` : Équilibré sans LLM
- `quality` : Qualité maximale avec LLM

### extract_specific_content.py

Extraction de contenu spécifique avec LLM :

```bash
# Extraction de tableaux avec LLM
python custom_docs/scripts/extract_specific_content.py document.pdf --content-type tables --llm --provider gemini --output outputs/

# Extraction d'équations avec Claude
python custom_docs/scripts/extract_specific_content.py document.pdf --content-type equations --llm --provider claude --output outputs/
```

## 🎯 Recommandations par cas d'usage

### Documents simples (texte principalement)
```bash
# Mode rapide sans LLM suffit
python custom_docs/scripts/convert_to_markdown.py document.pdf --mode fast
```

### Documents avec tableaux complexes
```bash
# Utiliser Gemini pour les tableaux
python custom_docs/scripts/extract_specific_content.py document.pdf --content-type tables --llm --provider gemini
```

### Documents scientifiques avec équations
```bash
# Claude excelle sur les équations mathématiques
python custom_docs/scripts/analyze_pdf_deep.py document.pdf --llm --provider claude
```

### Documents multilingues
```bash
# Gemini gère bien le multilingue
python custom_docs/scripts/convert_to_markdown.py document.pdf --mode quality --provider gemini
```

## 🛠️ Résolution de problèmes

### Erreur : "Provider non configuré"

```bash
# Vérifiez le statut de vos providers
python custom_docs/scripts/llm_config.py --status
```

### Erreur : "No module named 'anthropic'"

```bash
# Installez le package requis
pip install anthropic
```

### Erreur : "API key not found"

1. Vérifiez votre fichier `.env`
2. Assurez-vous que les clés API sont valides
3. Testez avec un autre provider

### Performance lente

1. Utilisez `gemini` (plus rapide que Claude)
2. Essayez le mode `balanced` au lieu de `quality`
3. Désactivez `--force_ocr` si pas nécessaire

## 📊 Comparaison des providers

### Gemini (Google) ⭐ Recommandé
- **Avantages :** Rapide, pré-installé, excellent sur tableaux
- **Inconvénients :** Limites de débit API gratuite
- **Meilleur pour :** Usage général, tableaux, documents multilingues

### Claude (Anthropic)
- **Avantages :** Excellente compréhension, bon sur les équations
- **Inconvénients :** Plus lent, package supplémentaire requis
- **Meilleur pour :** Documents scientifiques, analyses approfondies

### OpenAI
- **Avantages :** Fiable, bonne qualité générale
- **Inconvénients :** Coût plus élevé, package supplémentaire
- **Meilleur pour :** Usage professionnel, qualité constante

### Azure OpenAI
- **Avantages :** Contrôle entreprise, conformité
- **Inconvénients :** Configuration plus complexe
- **Meilleur pour :** Environnements d'entreprise

### Ollama (Local)
- **Avantages :** Gratuit, privé, pas de limites API
- **Inconvénients :** Installation locale requise, plus lent
- **Meilleur pour :** Documents sensibles, usage hors ligne

## 🔍 Exemples avancés

### Analyse complète avec fallback
```bash
# Essaie Gemini, puis Claude si échec
python custom_docs/scripts/analyze_pdf_deep.py document.pdf --llm --provider gemini || \
python custom_docs/scripts/analyze_pdf_deep.py document.pdf --llm --provider claude
```

### Traitement en lot avec provider optimal
```bash
# Traiter tous les PDFs d'un dossier
for file in inputs/*.pdf; do
    python custom_docs/scripts/convert_to_markdown.py "$file" --mode quality --provider gemini --output outputs/
done
```

### Extraction multi-contenu
```bash
# Extraire tableaux ET équations avec le meilleur provider pour chaque
python custom_docs/scripts/extract_specific_content.py document.pdf --content-type tables --llm --provider gemini --output outputs/
python custom_docs/scripts/extract_specific_content.py document.pdf --content-type equations --llm --provider claude --output outputs/
```

---

**💡 Conseil :** Commencez toujours avec `--provider auto` pour laisser le système choisir le meilleur provider disponible !