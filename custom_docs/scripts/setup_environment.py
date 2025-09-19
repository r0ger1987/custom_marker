#!/usr/bin/env python3
"""
Script de Configuration d'Environnement pour Marker
==================================================

Ce script configure automatiquement l'environnement Marker avec :
- Vérification des dépendances
- Configuration des clés API
- Test des services LLM
- Optimisation des performances

Usage:
    python scripts/setup_environment.py [--check-only] [--gpu] [--quiet]

Auteur: Claude Code
"""

import sys
import os
import json
import subprocess
import platform
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import argparse

# Ajout du chemin pour importer llm_config
sys.path.append(os.path.dirname(__file__))
from llm_config import LLMConfig

class EnvironmentSetup:
    """Gestionnaire de configuration d'environnement"""

    def __init__(self, quiet: bool = False):
        self.quiet = quiet
        self.checks_passed = 0
        self.checks_total = 0
        self.issues = []
        self.recommendations = []

    def log(self, message: str, level: str = "info"):
        """Log avec gestion du mode quiet"""
        if self.quiet and level == "info":
            return

        icons = {
            "info": "ℹ️ ",
            "success": "✅",
            "warning": "⚠️ ",
            "error": "❌",
            "debug": "🔍"
        }
        print(f"{icons.get(level, '')} {message}")

    def check_python_version(self) -> bool:
        """Vérifier la version Python"""
        self.checks_total += 1
        version = sys.version_info

        if version >= (3, 8):
            self.log(f"Python {version.major}.{version.minor}.{version.micro} - Compatible", "success")
            self.checks_passed += 1
            return True
        else:
            self.log(f"Python {version.major}.{version.minor}.{version.micro} - Version trop ancienne (minimum 3.8)", "error")
            self.issues.append("Mise à jour Python vers version 3.8+")
            return False

    def check_system_resources(self) -> Dict[str, any]:
        """Vérifier les ressources système"""
        self.checks_total += 1

        try:
            import psutil

            # Informations système
            cpu_count = psutil.cpu_count()
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('.')

            system_info = {
                "cpu_cores": cpu_count,
                "ram_gb": round(memory.total / 1024**3, 1),
                "ram_available_gb": round(memory.available / 1024**3, 1),
                "disk_free_gb": round(disk.free / 1024**3, 1),
                "platform": platform.system(),
                "architecture": platform.machine()
            }

            # Évaluation des performances
            if cpu_count >= 4 and memory.total >= 8 * 1024**3:
                self.log(f"Ressources système: {cpu_count} CPU, {system_info['ram_gb']}GB RAM - Excellent", "success")
                self.checks_passed += 1
            elif cpu_count >= 2 and memory.total >= 4 * 1024**3:
                self.log(f"Ressources système: {cpu_count} CPU, {system_info['ram_gb']}GB RAM - Suffisant", "warning")
                self.recommendations.append("Considérer plus de RAM pour les documents complexes")
                self.checks_passed += 1
            else:
                self.log(f"Ressources système: {cpu_count} CPU, {system_info['ram_gb']}GB RAM - Limitées", "warning")
                self.recommendations.append("Utiliser mode --workers 1 pour économiser la mémoire")
                self.checks_passed += 1

            return system_info

        except ImportError:
            self.log("psutil non installé - impossible de vérifier les ressources", "warning")
            self.recommendations.append("Installer psutil: pip install psutil")
            return {}

    def check_gpu_availability(self) -> Dict[str, any]:
        """Vérifier la disponibilité GPU"""
        self.checks_total += 1
        gpu_info = {"available": False, "device": "cpu"}

        try:
            import torch

            if torch.cuda.is_available():
                gpu_info.update({
                    "available": True,
                    "device": "cuda",
                    "gpu_count": torch.cuda.device_count(),
                    "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "Unknown"
                })
                self.log(f"GPU CUDA détecté: {gpu_info['gpu_name']}", "success")
                os.environ["TORCH_DEVICE"] = "cuda"

            elif torch.backends.mps.is_available():
                gpu_info.update({
                    "available": True,
                    "device": "mps",
                    "gpu_count": 1,
                    "gpu_name": "Apple Silicon"
                })
                self.log("GPU MPS (Apple Silicon) détecté", "success")
                os.environ["TORCH_DEVICE"] = "mps"

            else:
                self.log("Aucun GPU détecté - utilisation CPU", "info")
                os.environ["TORCH_DEVICE"] = "cpu"

            self.checks_passed += 1
            return gpu_info

        except ImportError:
            self.log("PyTorch non installé - impossible de vérifier GPU", "warning")
            self.issues.append("Installer PyTorch pour le support GPU")
            return gpu_info

    def check_marker_installation(self) -> bool:
        """Vérifier l'installation de Marker"""
        self.checks_total += 1

        try:
            import marker
            version = getattr(marker, '__version__', 'Unknown')
            self.log(f"Marker PDF installé - version {version}", "success")
            self.checks_passed += 1
            return True

        except ImportError:
            self.log("Marker PDF non installé", "error")
            self.issues.append("Installer Marker: pip install marker-pdf[full]")
            return False

    def check_llm_providers(self) -> Dict[str, Dict]:
        """Vérifier les providers LLM disponibles"""
        self.checks_total += 1

        try:
            # Test des providers via LLMConfig
            available_providers = LLMConfig.get_available_providers()

            provider_details = {}
            configured_count = 0

            for provider, is_available in available_providers.items():
                provider_details[provider] = {"available": is_available}

                if is_available:
                    configured_count += 1
                    if provider == "gemini":
                        provider_details[provider]["note"] = "Recommandé - gratuit avec limites"
                    elif provider == "bedrock":
                        provider_details[provider]["note"] = "AWS - facturé à l'usage"
                    elif provider == "claude":
                        provider_details[provider]["note"] = "Anthropic - haute qualité"
                    elif provider == "openai":
                        provider_details[provider]["note"] = "OpenAI - fiable"

            if configured_count > 0:
                self.log(f"Providers LLM: {configured_count}/{len(available_providers)} configurés", "success")
                for provider, details in provider_details.items():
                    if details["available"]:
                        note = details.get("note", "")
                        self.log(f"  ✓ {provider} {note}", "info")
                self.checks_passed += 1
            else:
                self.log("Aucun provider LLM configuré", "warning")
                self.recommendations.append("Configurer au moins un provider LLM pour la qualité optimale")
                self.checks_passed += 1

            return provider_details

        except Exception as e:
            self.log(f"Erreur lors de la vérification des providers LLM: {e}", "error")
            return {}

    def check_dependencies(self) -> List[str]:
        """Vérifier les dépendances optionnelles"""
        optional_deps = {
            "psutil": "Monitoring des ressources système",
            "pillow": "Traitement d'images",
            "python-dotenv": "Gestion des variables d'environnement",
            "anthropic": "Support Claude",
            "openai": "Support OpenAI/Azure",
            "google-generativeai": "Support Gemini",
            "boto3": "Support AWS Bedrock"
        }

        installed = []
        missing = []

        for package, description in optional_deps.items():
            try:
                __import__(package.replace("-", "_"))
                installed.append(f"{package}: {description}")
            except ImportError:
                missing.append(f"{package}: {description}")

        if installed:
            self.log(f"Dépendances optionnelles installées: {len(installed)}", "info")
            for dep in installed:
                self.log(f"  ✓ {dep}", "debug")

        if missing:
            self.log(f"Dépendances optionnelles manquantes: {len(missing)}", "info")
            for dep in missing:
                self.log(f"  - {dep}", "debug")

        return missing

    def setup_directories(self) -> bool:
        """Créer la structure de répertoires"""
        self.checks_total += 1

        directories = [
            "outputs",
            "outputs/analyses",
            "outputs/tables",
            "outputs/equations",
            "outputs/images",
            "outputs/sections",
            "inputs",
            "logs"
        ]

        created = []
        for directory in directories:
            path = Path(directory)
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)
                created.append(directory)

        if created:
            self.log(f"Répertoires créés: {', '.join(created)}", "success")
        else:
            self.log("Structure de répertoires déjà présente", "info")

        self.checks_passed += 1
        return True

    def create_env_template(self) -> bool:
        """Créer le template .env si nécessaire"""
        env_file = Path(".env")
        env_example = Path(".env.example")

        if not env_file.exists() and env_example.exists():
            import shutil
            shutil.copy(env_example, env_file)
            self.log("Fichier .env créé à partir de .env.example", "success")
            self.recommendations.append("Configurer vos clés API dans le fichier .env")
            return True
        elif not env_file.exists():
            self.log("Aucun fichier .env trouvé", "warning")
            self.recommendations.append("Créer un fichier .env avec vos clés API")
            return False
        else:
            self.log("Fichier .env existant", "info")
            return True

    def optimize_environment(self) -> Dict[str, str]:
        """Optimiser les variables d'environnement"""
        optimizations = {}

        # Optimisation PyTorch
        if not os.getenv("TOKENIZERS_PARALLELISM"):
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            optimizations["TOKENIZERS_PARALLELISM"] = "false"

        # Répertoire de sortie par défaut
        if not os.getenv("OUTPUT_DIR"):
            os.environ["OUTPUT_DIR"] = "./outputs"
            optimizations["OUTPUT_DIR"] = "./outputs"

        if optimizations:
            self.log(f"Variables d'environnement optimisées: {list(optimizations.keys())}", "success")

        return optimizations

    def generate_report(self, system_info: Dict, gpu_info: Dict, llm_providers: Dict) -> Dict:
        """Générer un rapport de configuration"""
        report = {
            "timestamp": __import__("datetime").datetime.now().isoformat(),
            "system": {
                "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                "platform": platform.system(),
                "architecture": platform.machine(),
                **system_info
            },
            "gpu": gpu_info,
            "llm_providers": llm_providers,
            "checks": {
                "passed": self.checks_passed,
                "total": self.checks_total,
                "success_rate": round(self.checks_passed / max(self.checks_total, 1) * 100, 1)
            },
            "issues": self.issues,
            "recommendations": self.recommendations
        }

        # Sauvegarder le rapport
        report_file = Path("logs/environment_report.json")
        report_file.parent.mkdir(exist_ok=True)

        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        self.log(f"Rapport sauvegardé: {report_file}", "info")
        return report

    def run_full_setup(self, check_only: bool = False) -> bool:
        """Exécuter la configuration complète"""
        self.log("🚀 Configuration de l'environnement Marker", "info")
        self.log("=" * 50, "info")

        # Vérifications de base
        self.check_python_version()
        system_info = self.check_system_resources()
        gpu_info = self.check_gpu_availability()

        marker_ok = self.check_marker_installation()
        if not marker_ok and check_only:
            self.log("Mode vérification uniquement - Marker non installé", "error")

        llm_providers = self.check_llm_providers()
        missing_deps = self.check_dependencies()

        if not check_only:
            self.setup_directories()
            self.create_env_template()
            self.optimize_environment()

        # Rapport final
        report = self.generate_report(system_info, gpu_info, llm_providers)

        self.log("=" * 50, "info")
        self.log(f"Configuration terminée: {self.checks_passed}/{self.checks_total} vérifications réussies",
                "success" if self.checks_passed == self.checks_total else "warning")

        if self.issues:
            self.log(f"⚠️  {len(self.issues)} problèmes détectés:", "warning")
            for issue in self.issues:
                self.log(f"  • {issue}", "warning")

        if self.recommendations:
            self.log(f"💡 {len(self.recommendations)} recommandations:", "info")
            for rec in self.recommendations:
                self.log(f"  • {rec}", "info")

        return len(self.issues) == 0

def main():
    """Fonction principale"""
    parser = argparse.ArgumentParser(
        description="Configuration automatique de l'environnement Marker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
  # Configuration complète
  python scripts/setup_environment.py

  # Vérification uniquement
  python scripts/setup_environment.py --check-only

  # Configuration silencieuse
  python scripts/setup_environment.py --quiet
        """
    )

    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Vérifier uniquement sans créer de fichiers"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Mode silencieux (erreurs et warnings uniquement)"
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Forcer la configuration GPU"
    )

    args = parser.parse_args()

    # Configuration
    setup = EnvironmentSetup(quiet=args.quiet)

    try:
        success = setup.run_full_setup(check_only=args.check_only)

        if not success:
            setup.log("Configuration incomplète - consultez les recommandations ci-dessus", "warning")

        # Instructions finales
        if not args.check_only:
            setup.log("\n🎯 Prochaines étapes:", "info")
            setup.log("1. Configurez vos clés API dans .env", "info")
            setup.log("2. Testez avec: python scripts/llm_config.py --status", "info")
            setup.log("3. Première conversion: python scripts/convert_to_markdown.py inputs/test.pdf", "info")

        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        setup.log("Configuration interrompue par l'utilisateur", "warning")
        sys.exit(130)
    except Exception as e:
        setup.log(f"Erreur inattendue: {e}", "error")
        sys.exit(1)

if __name__ == "__main__":
    main()