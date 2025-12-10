from __future__ import annotations

from typing import Any, Callable, Dict, Tuple

from src.adapters.adapter import MCPAdapter
from src.agent.base import AgentConfig, BaseAgent

ProviderFactory = Callable[[AgentConfig, Dict[str, Any]], Tuple[BaseAgent, MCPAdapter]]

_REGISTRY: Dict[str, ProviderFactory] = {}


def register_provider(name: str, factory: ProviderFactory) -> None:
    """Registers a provider factory under the given name."""
    _REGISTRY[name.lower()] = factory


def create_provider_bundle(
    provider: str,
    agent_config: AgentConfig,
    *,
    options: Dict[str, Any] | None = None,
) -> Tuple[BaseAgent, MCPAdapter]:
    """Creates a `(BaseAgent, MCPAdapter)` pair for the requested provider."""
    normalized = provider.lower()
    if normalized not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys())) or "<none>"
        raise ValueError(f"Unknown provider '{provider}'. Available: {available}")
    factory = _REGISTRY[normalized]
    return factory(agent_config, dict(options or {}))


def _register_default_providers() -> None:
    def _ollama_factory(agent_config: AgentConfig, options: Dict[str, Any]) -> Tuple[BaseAgent, MCPAdapter]:
        from src.agent.ollama.ollama_agent import OllamaAgent
        from src.adapters.ollama.ollama_adapter import OllamaAdapter

        host = options.get("host") or "http://localhost:11434"
        agent_options = options.get("options")
        agent = OllamaAgent(config=agent_config, host=host, options=agent_options)
        adapter = OllamaAdapter(agent_config)
        return agent, adapter

    register_provider("ollama", _ollama_factory)


_register_default_providers()
