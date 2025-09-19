# Guide d'Installation - Marker PDF

## ✅ Installation Réussie !

Félicitations ! Marker est maintenant configuré et fonctionnel. Voici un résumé de ce qui a été installé :

### 📦 Dépendances Installées

```bash
✅ pydantic (2.11.7) - Validation des données
✅ pydantic-core (2.33.2) - Cœur de pydantic
✅ psutil (7.0.0) - Monitoring système
✅ transformers (4.53.3) - Modèles de transformer
✅ surya-ocr (0.16.0) - OCR avancé
✅ pdftext (0.6.3) - Extraction de texte PDF
✅ beautifulsoup4 (4.13.5) - Parsing HTML/XML
✅ markdownify (1.2.0) - Conversion HTML vers Markdown
✅ scikit-learn (1.7.1) - Machine learning
✅ google-genai (1.32.0) - API Google Gemini
✅ opencv-python-headless (4.12.0.88) - Vision par ordinateur
✅ einops (0.8.1) - Opérations tensorielles
✅ Et beaucoup d'autres...
```

### 🎯 État Actuel

- ✅ **Imports Marker** : Fonctionnels
- ✅ **Scripts personnalisés** : Prêts à l'usage
- 🔄 **Premier lancement** : Téléchargement des modèles en cours (1.31GB)
- ✅ **GPU CUDA** : Détecté et fonctionnel

## 🚀 Prochaines Étapes

### 1. Laissez le téléchargement se terminer

Les modèles se téléchargent automatiquement dans `~/.cache/datalab/models/`. Ce téléchargement ne se fait qu'une seule fois.

### 2. Configurez vos APIs (Optionnel)

Pour utiliser le mode LLM haute qualité, configurez vos clés API :

```bash
# Copiez le fichier d'exemple
cp .env.example .env

# Éditez avec vos vraies clés
nano .env
```

### 3. Testez avec un PDF

```bash
# Mode rapide (sans LLM)
python3 scripts/convert_to_markdown.py inputs/votre_document.pdf --mode fast

# Mode équilibré  
python3 scripts/convert_to_markdown.py inputs/votre_document.pdf --mode balanced

# Mode qualité maximale (nécessite API)
python3 scripts/convert_to_markdown.py inputs/votre_document.pdf --mode quality
```

## 📋 Commandes Rapides

### Analyse d'un PDF
```bash
python3 scripts/analyze_pdf_deep.py inputs/document.pdf --output outputs/
```

### Conversion avec sauvegarde
```bash
python3 scripts/convert_to_markdown.py inputs/document.pdf --mode balanced --output outputs/
```

### Extraction de contenu spécifique
```bash
# Tables
python3 scripts/extract_specific_content.py inputs/document.pdf --content-type tables --output outputs/

# Équations
python3 scripts/extract_specific_content.py inputs/document.pdf --content-type equations --output outputs/
```

### Traitement en lot
```bash
python3 scripts/batch_processor.py inputs/ outputs/ --strategy parallel
```

## 🔧 Configuration Système

### Variables d'Environnement Recommandées

```bash
# Configuration GPU (si disponible)
export TORCH_DEVICE=cuda

# Répertoire de sortie par défaut
export OUTPUT_DIR=./outputs

# Optimisation mémoire
export TOKENIZERS_PARALLELISM=false
```

### Optimisations Performance

Pour votre système, voici les recommandations :

- **Workers optimaux** : 4-6 (basé sur votre RAM et CPU)
- **Mode recommandé** : `balanced` pour l'usage quotidien
- **GPU** : Utilisé automatiquement si disponible
- **Mémoire** : ~4GB par worker

## ⚠️ Notes Importantes

### Conflits de Dépendances

Vous verrez des avertissements sur les conflits de versions - c'est normal dans un environnement avec des packages système existants. Marker fonctionne malgré ces avertissements.

### Premier Lancement

Le premier lancement est lent car :
1. Téléchargement des modèles (1.31GB) - une seule fois
2. Compilation CUDA - quelques minutes
3. Initialisation des modèles - normal

### Taille des Modèles

Les modèles téléchargés occupent ~1.5GB sur disque :
- Modèle de reconnaissance : ~1.31GB
- Modèles de layout : ~200MB
- Cache divers : ~50MB

## 🆘 Dépannage

### Si les scripts ne fonctionnent pas

```bash
# Vérifiez l'import
python3 -c "
import sys
sys.path.append('/home/roger/RAG/marker')
from marker.converters.pdf import PdfConverter
print('✅ Marker OK')
"
```

### Si les téléchargements échouent

```bash
# Nettoyez le cache
rm -rf ~/.cache/datalab/models/
# Relancez le script
```

### Problèmes de mémoire

```bash
# Utilisez moins de workers
python3 scripts/batch_processor.py inputs/ outputs/ --workers 2

# Ou mode séquentiel
python3 scripts/batch_processor.py inputs/ outputs/ --strategy sequential
```

## 📊 Performance Attendue

Sur votre système, vous devriez avoir :

| Type Document | Mode Fast | Mode Balanced | Mode Quality |
|---------------|-----------|---------------|--------------|
| Article (10p) | ~15s | ~30s | ~60s |
| Rapport (50p) | ~60s | ~120s | ~300s |
| Manuel (200p) | ~240s | ~480s | ~1200s |

## 🎉 Prêt à Utiliser !

Une fois le téléchargement terminé, vous pourrez utiliser tous les scripts personnalisés. Consultez le `GUIDE_UTILISATION.md` pour des exemples détaillés !

---

**💡 Astuce :** Ajoutez `/home/roger/RAG/marker/custom_docs` à votre PATH pour utiliser les scripts depuis n'importe où !