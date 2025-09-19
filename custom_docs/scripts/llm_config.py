#!/usr/bin/env python3
"""
Module de configuration LLM pour les scripts custom_docs
========================================================

Ce module gère automatiquement :
- Le chargement des variables depuis .env
- La sélection du provider LLM (Gemini, OpenAI, Claude, Bedrock, etc.)
- La configuration appropriée pour Marker

Auteur: Claude Code
"""

import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from dotenv import load_dotenv

# Charger le fichier .env depuis custom_docs
env_path = Path(__file__).parent.parent / '.env'
if env_path.exists():
    load_dotenv(env_path, override=True)
    print(f"✅ Chargement des variables depuis : {env_path}")
else:
    print(f"⚠️  Fichier .env non trouvé : {env_path}")

class LLMConfig:
    """Gestionnaire de configuration LLM pour Marker"""

    # Mapping des providers vers les services Marker (chemins complets de classes)
    PROVIDER_MAPPING = {
        "gemini": "marker.services.gemini.GoogleGeminiService",
        "openai": "marker.services.openai.OpenAIService",
        "claude": "marker.services.claude.AnthropicService",
        "anthropic": "marker.services.claude.AnthropicService",
        "ollama": "marker.services.ollama.OllamaService",
        "azure": "marker.services.azure_openai.AzureOpenAIService",
        "vertex": "marker.services.vertex.VertexAIService",
        "bedrock": "marker.services.bedrock.BedrockService"
    }

    @staticmethod
    def get_available_providers() -> Dict[str, bool]:
        """Retourne les providers disponibles selon les clés API configurées"""
        return {
            "gemini": bool(os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")),
            "openai": bool(os.getenv("OPENAI_API_KEY")),
            "claude": bool(os.getenv("CLAUDE_API_KEY") or os.getenv("ANTHROPIC_API_KEY")),
            "azure": bool(os.getenv("AZURE_API_KEY") and os.getenv("AZURE_ENDPOINT")),
            "bedrock": bool(os.getenv("AWS_ACCESS_KEY_ID") or os.getenv("AWS_PROFILE") or os.path.exists(os.path.expanduser("~/.aws/credentials"))),
            "ollama": True,  # Toujours disponible si installé localement
        }

    @staticmethod
    def setup_environment(provider: str = "auto") -> str:
        """
        Configure l'environnement pour le provider LLM spécifié

        Args:
            provider: Provider à utiliser ("auto", "gemini", "openai", "claude", etc.)

        Returns:
            Le nom du provider configuré
        """
        available = LLMConfig.get_available_providers()

        # Mode auto : sélectionner le premier provider disponible
        if provider == "auto":
            priority_order = ["gemini", "claude", "bedrock", "openai", "azure", "ollama"]
            for p in priority_order:
                if available.get(p, False):
                    provider = p
                    print(f"🤖 Provider auto-sélectionné : {provider}")
                    break
            else:
                raise ValueError("Aucun provider LLM configuré. Vérifiez votre fichier .env")

        # Vérifier que le provider demandé est disponible
        if provider not in LLMConfig.PROVIDER_MAPPING:
            raise ValueError(f"Provider non supporté : {provider}. Options : {list(LLMConfig.PROVIDER_MAPPING.keys())}")

        if not available.get(provider, False) and provider != "ollama":
            raise ValueError(f"Provider {provider} non configuré. Vérifiez les clés API dans .env")

        # Configurer les variables d'environnement spécifiques
        if provider == "gemini":
            # Marker cherche GOOGLE_API_KEY
            if not os.getenv("GOOGLE_API_KEY") and os.getenv("GEMINI_API_KEY"):
                os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

        elif provider == "claude" or provider == "anthropic":
            # Marker cherche ANTHROPIC_API_KEY
            if not os.getenv("ANTHROPIC_API_KEY") and os.getenv("CLAUDE_API_KEY"):
                os.environ["ANTHROPIC_API_KEY"] = os.getenv("CLAUDE_API_KEY")

        print(f"✅ Provider LLM configuré : {provider}")
        return provider

    @staticmethod
    def get_llm_config(provider: str = "auto", custom_config: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Retourne la configuration pour le provider LLM

        Args:
            provider: Provider à utiliser
            custom_config: Configuration personnalisée additionnelle

        Returns:
            Configuration pour ConfigParser de Marker
        """
        # Setup du provider
        actual_provider = LLMConfig.setup_environment(provider)

        # Configuration de base
        config = {
            "use_llm": True,
            "llm_service": LLMConfig.PROVIDER_MAPPING[actual_provider]
        }

        # Configuration spécifique par provider
        if actual_provider == "gemini":
            config.update({
                "gemini_model": os.getenv("GEMINI_MODEL", "gemini-1.5-flash"),
                "gemini_api_key": os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"),
            })

        elif actual_provider == "openai":
            config.update({
                "openai_model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                "openai_api_key": os.getenv("OPENAI_API_KEY"),
            })

        elif actual_provider in ["claude", "anthropic"]:
            config.update({
                "anthropic_model": os.getenv("CLAUDE_MODEL", "claude-3-haiku-20240307"),
                "anthropic_api_key": os.getenv("ANTHROPIC_API_KEY") or os.getenv("CLAUDE_API_KEY"),
            })

        elif actual_provider == "azure":
            config.update({
                "azure_openai_api_key": os.getenv("AZURE_API_KEY"),
                "azure_openai_endpoint": os.getenv("AZURE_ENDPOINT"),
                "azure_openai_deployment": os.getenv("AZURE_DEPLOYMENT", "gpt-4"),
            })

        elif actual_provider == "ollama":
            config.update({
                "ollama_model": os.getenv("OLLAMA_MODEL", "llama3.2"),
                "ollama_url": os.getenv("OLLAMA_URL", "http://localhost:11434"),
            })

        elif actual_provider == "bedrock":
            config.update({
                "bedrock_model_id": os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-5-sonnet-20241022-v2:0"),
                "aws_region": os.getenv("AWS_REGION", "us-east-1"),
                "aws_access_key_id": os.getenv("AWS_ACCESS_KEY_ID"),
                "aws_secret_access_key": os.getenv("AWS_SECRET_ACCESS_KEY"),
                "aws_session_token": os.getenv("AWS_SESSION_TOKEN"),
            })

        # Appliquer la configuration personnalisée
        if custom_config:
            config.update(custom_config)

        return config

    @staticmethod
    def print_status():
        """Affiche le statut de configuration des providers"""
        print("\n📊 Statut des providers LLM :")
        print("-" * 40)

        available = LLMConfig.get_available_providers()
        for provider, is_available in available.items():
            status = "✅ Configuré" if is_available else "❌ Non configuré"
            print(f"  {provider:10} : {status}")

        # Afficher les variables d'environnement Marker
        print("\n🔧 Configuration Marker :")
        print("-" * 40)
        device = os.getenv("TORCH_DEVICE", "auto")
        output_dir = os.getenv("OUTPUT_DIR", "./outputs")
        print(f"  TORCH_DEVICE : {device}")
        print(f"  OUTPUT_DIR   : {output_dir}")
        print()

if __name__ == "__main__":
    """Test du module"""
    import argparse

    parser = argparse.ArgumentParser(description="Test de configuration LLM")
    parser.add_argument("--provider", default="auto",
                       choices=["auto", "gemini", "openai", "claude", "azure", "bedrock", "ollama"],
                       help="Provider LLM à utiliser")
    parser.add_argument("--status", action="store_true",
                       help="Afficher le statut de configuration")

    args = parser.parse_args()

    if args.status:
        LLMConfig.print_status()
    else:
        try:
            provider = LLMConfig.setup_environment(args.provider)
            config = LLMConfig.get_llm_config(args.provider)
            print(f"\n✅ Configuration réussie pour {provider}")
            print(f"Configuration : {config}")
        except Exception as e:
            print(f"❌ Erreur : {e}")
            sys.exit(1)