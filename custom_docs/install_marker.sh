#!/bin/bash
"""
Script d'installation automatique de Marker
===========================================

Ce script installe Marker avec toutes ses dépendances et configure l'environnement.

Usage:
    ./install_marker.sh [--gpu] [--full]

Options:
    --gpu   : Installation avec support GPU (CUDA)
    --full  : Installation complète avec tous les providers LLM

Auteur: Claude Code
"""

set -e  # Arrêter en cas d'erreur

echo "🚀 Installation de Marker PDF"
echo "=============================="

# Couleurs pour l'affichage
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Fonctions d'affichage
print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Vérifier Python
check_python() {
    print_info "Vérification de Python..."

    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
        print_success "Python $PYTHON_VERSION trouvé"

        # Vérifier la version (minimum 3.8)
        if python3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)"; then
            print_success "Version Python compatible"
        else
            print_error "Python 3.8+ requis, version actuelle: $PYTHON_VERSION"
            exit 1
        fi
    else
        print_error "Python 3 non trouvé. Veuillez installer Python 3.8+"
        exit 1
    fi
}

# Créer l'environnement virtuel
create_venv() {
    print_info "Création de l'environnement virtuel..."

    if [ -d "marker_env" ]; then
        print_warning "Environnement virtuel existant trouvé"
        read -p "Voulez-vous le recréer ? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf marker_env
            print_info "Ancien environnement supprimé"
        else
            print_info "Utilisation de l'environnement existant"
            return
        fi
    fi

    python3 -m venv marker_env
    print_success "Environnement virtuel créé"
}

# Activer l'environnement virtuel
activate_venv() {
    print_info "Activation de l'environnement virtuel..."
    source marker_env/bin/activate
    print_success "Environnement virtuel activé"
}

# Installer les dépendances de base
install_base_deps() {
    print_info "Installation des dépendances de base..."

    # Mise à jour pip
    pip install --upgrade pip

    # Installation de Marker
    if [[ "$1" == "--gpu" ]]; then
        print_info "Installation avec support GPU..."
        pip install marker-pdf[full] torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    else
        print_info "Installation CPU standard..."
        pip install marker-pdf[full]
    fi

    print_success "Marker installé"
}

# Installer les providers LLM additionnels
install_llm_providers() {
    print_info "Installation des providers LLM..."

    # Providers principaux
    pip install anthropic openai google-generativeai

    # AWS Bedrock
    pip install boto3

    # Autres providers optionnels
    if [[ "$1" == "--full" ]]; then
        print_info "Installation complète des providers..."
        pip install azure-openai ollama-python
    fi

    # Utilitaires
    pip install python-dotenv

    print_success "Providers LLM installés"
}

# Configurer l'environnement
setup_environment() {
    print_info "Configuration de l'environnement..."

    # Créer le fichier .env s'il n'existe pas
    if [ ! -f ".env" ]; then
        if [ -f ".env.example" ]; then
            cp .env.example .env
            print_success "Fichier .env créé à partir de .env.example"
            print_warning "N'oubliez pas de configurer vos clés API dans .env"
        else
            print_warning "Fichier .env.example non trouvé"
        fi
    else
        print_info "Fichier .env existant conservé"
    fi

    # Créer le dossier outputs
    mkdir -p outputs
    print_success "Dossier outputs créé"
}

# Tester l'installation
test_installation() {
    print_info "Test de l'installation..."

    # Test import marker
    if python -c "import marker; print('Marker version:', marker.__version__)" 2>/dev/null; then
        print_success "Marker fonctionne correctement"
    else
        print_error "Erreur lors du test de Marker"
        return 1
    fi

    # Test des scripts
    if python scripts/llm_config.py --status > /dev/null 2>&1; then
        print_success "Scripts personnalisés fonctionnels"
    else
        print_warning "Problème avec les scripts personnalisés"
    fi
}

# Afficher les instructions finales
show_instructions() {
    echo
    print_success "🎉 Installation terminée !"
    echo
    print_info "Pour utiliser Marker :"
    echo "  1. Activez l'environnement : source marker_env/bin/activate"
    echo "  2. Configurez vos clés API dans le fichier .env"
    echo "  3. Testez avec : python scripts/llm_config.py --status"
    echo
    print_info "Scripts disponibles :"
    echo "  • convert_to_markdown.py  : Conversion PDF → Markdown"
    echo "  • analyze_pdf_deep.py    : Analyse approfondie avec LLM"
    echo "  • bedrock_converter.py   : Conversion avec AWS Bedrock"
    echo "  • batch_processor.py     : Traitement en lot"
    echo
    print_info "Documentation complète dans les fichiers README.md et GUIDE_*.md"
}

# Fonction principale
main() {
    local gpu_support=false
    local full_install=false

    # Parser les arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --gpu)
                gpu_support=true
                shift
                ;;
            --full)
                full_install=true
                shift
                ;;
            -h|--help)
                echo "Usage: $0 [--gpu] [--full]"
                echo "  --gpu   : Installation avec support GPU"
                echo "  --full  : Installation complète"
                exit 0
                ;;
            *)
                print_error "Option inconnue: $1"
                exit 1
                ;;
        esac
    done

    print_info "Démarrage de l'installation..."
    if [[ "$gpu_support" == true ]]; then
        print_info "Mode GPU activé"
    fi
    if [[ "$full_install" == true ]]; then
        print_info "Installation complète activée"
    fi

    # Étapes d'installation
    check_python
    create_venv
    activate_venv

    if [[ "$gpu_support" == true ]]; then
        install_base_deps "--gpu"
    else
        install_base_deps
    fi

    if [[ "$full_install" == true ]]; then
        install_llm_providers "--full"
    else
        install_llm_providers
    fi

    setup_environment

    if test_installation; then
        show_instructions
    else
        print_error "L'installation a échoué lors des tests"
        exit 1
    fi
}

# Point d'entrée
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi