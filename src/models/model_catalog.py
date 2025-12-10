from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class ModelInfo:
    provider: str
    model_id: str
    display_name: str
    capabilities: Dict[str, bool]


_CATALOG: List[ModelInfo] = [
    ModelInfo(
        provider="ollama",
        model_id="qwen3-coder:30b",
        display_name="Qwen3 Coder 30B",
        capabilities={"thinking": False, "streaming": True, "vision": False},
    ),
        ModelInfo(
        provider="ollama",
        model_id="qwen3:8b",
        display_name="Qwen 3 8B",
        capabilities={"thinking": True, "streaming": True, "vision": False},
    ),
    ModelInfo(
        provider="ollama",
        model_id="llama3.2:3b",
        display_name="Llama 3.2 3B",
        capabilities={"thinking": False, "streaming": True, "vision": False},
    ),
]

def providers() -> List[str]:
    """Return sorted list of providers present in the catalog."""
    return sorted({entry.provider for entry in _CATALOG})


def list_models(provider: str) -> List[ModelInfo]:
    """Return all models for a provider."""
    provider = provider.lower()
    return [entry for entry in _CATALOG if entry.provider == provider]

