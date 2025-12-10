from __future__ import annotations

import argparse
from typing import Any, Dict, Optional

from src.config.defaults import ProviderDefaults, RuntimeConfig, make_runtime_config
from src.models import model_catalog

CLIResolvedConfig = RuntimeConfig


class CLIConfig:
    """Command-line configuration helper for the agent CLI."""

    provider_defaults = ProviderDefaults()

    @staticmethod
    def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
        parser = argparse.ArgumentParser(description="MCP Agent CLI Client")
        parser.add_argument("--interactive", action="store_true", help="Launch the interactive configuration wizard.")
        parser.add_argument("--provider", help="Provider identifier (e.g. ollama).")
        parser.add_argument("--model", help="Model identifier to use.")
        return parser.parse_args(argv)

    @classmethod
    def resolve(cls, args: Optional[argparse.Namespace] = None) -> CLIResolvedConfig:
        if args is None:
            args = cls.parse_args()

        provider = (args.provider or "").strip().lower() or None
        model_id = (args.model or "").strip() or None

        if args.interactive or not provider or not model_id:
            provider, model_id = cls._run_wizard(provider, model_id)

        return make_runtime_config(provider, model_id)

    @classmethod
    def _run_wizard(cls, provider: Optional[str], model_id: Optional[str]) -> tuple[str, str]:
        print("\n=== MCP Agent CLI Configuration ===\n")
        provider_choice = cls._prompt_provider(provider)
        model_choice = cls._prompt_model(provider_choice, model_id)
        return provider_choice, model_choice

    @classmethod
    def _prompt_provider(cls, provider: Optional[str]) -> str:
        providers = model_catalog.providers()
        if not providers:
            providers = [cls.provider_defaults.provider]

        default_provider = (provider or cls.provider_defaults.provider).lower()

        while True:
            print("Available providers:")
            for idx, entry in enumerate(providers, start=1):
                marker = "*" if entry == default_provider else " "
                print(f"  {idx}) [{marker}] {entry}")

            choice = input(f"Choose provider [{default_provider}]: ").strip()
            if not choice:
                return default_provider

            if choice.isdigit():
                index = int(choice) - 1
                if 0 <= index < len(providers):
                    return providers[index]

            normalized = choice.lower()
            if normalized in providers:
                return normalized

            print("Please select a provider by index or name.\n")

    @classmethod
    def _prompt_model(cls, provider: str, model_id: Optional[str]) -> str:
        models = model_catalog.list_models(provider)
        default_id = model_id or cls._default_model_for(provider)

        if not models:
            while True:
                suffix = f" [{default_id}]" if default_id else ""
                manual = input(f"Enter model identifier for '{provider}'{suffix}: ").strip()
                if manual:
                    return manual
                if default_id:
                    return default_id
                print("A model identifier is required when no catalog entry exists.")

        print(f"\nAvailable models for '{provider}':")
        for idx, model in enumerate(models, start=1):
            caps = cls._format_capabilities(model.capabilities)
            print(f"  {idx}) {model.display_name} [{model.model_id}] - {caps}")

        default_label = default_id or models[0].model_id
        while True:
            choice = input(f"Choose model [{default_label}]: ").strip()
            if not choice:
                return default_label
            if choice.isdigit():
                index = int(choice) - 1
                if 0 <= index < len(models):
                    return models[index].model_id
            else:
                for model in models:
                    if model.model_id == choice:
                        return model.model_id
            print("Please select by index or provide a valid model identifier.")

    @staticmethod
    def _format_capabilities(capabilities: Dict[str, bool]) -> str:
        enabled = [name for name, value in capabilities.items() if value]
        return ", ".join(enabled) if enabled else "standard"

    @classmethod
    def _default_model_for(cls, provider: str) -> str:
        models = model_catalog.list_models(provider)
        if models:
            return models[0].model_id
        return ""

    @staticmethod
    def _resolve_default_model(
        candidates: list[model_catalog.ModelInfo],
        default_model_id: str,
    ) -> model_catalog.ModelInfo:
        if not candidates:
            raise ValueError("No candidate models available.")
        for model in candidates:
            if model.model_id == default_model_id:
                return model
        return candidates[0]
