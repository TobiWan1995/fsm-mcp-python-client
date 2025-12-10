from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from src.agent.base import AgentConfig
from src.mcp.client import MCPClientConfig
from src.models import model_catalog

SYSTEM_PROMPT_PATH = "src/prompts/system.md"


@dataclass(frozen=True)
class ProviderDefaults:
    """Default provider specific overrides for the CLI."""

    provider: str = "ollama"
    host: str = "http://ai-gpu:11434"
    options: Dict[str, Any] = field(
        default_factory=lambda: {
            "temperature": 0.1,
            "top_p": 0.8,
            "top_k": 10,
            "num_ctx": 50000,
        }
    )


@dataclass(frozen=True)
class MCPDefaults:
    """Default MCP connectivity settings for the CLI."""

    url: str = "http://127.0.0.1:8000/sse"
    timeout: float = 300.0
    session_name: str = "cli_mcp"
    transport: str = "sse"
    auth_token: Optional[str] = None
    sse_read_timeout: float = 60.0 * 5


@dataclass
class RuntimeConfig:
    agent_config: AgentConfig
    mcp_config: MCPClientConfig
    provider: str
    provider_options: Dict[str, Any]
    model_info: Optional[model_catalog.ModelInfo] = None


def make_runtime_config(provider: Optional[str], model_id: Optional[str]) -> RuntimeConfig:
    """
    Build the runtime configuration that the CLI uses to create a session.
    Provider/model selection happens at the CLI level; everything else is derived
    from the defaults defined in this module.
    """
    provider_defaults = ProviderDefaults()
    provider_name = (provider or provider_defaults.provider).lower()

    model_info = _resolve_model(provider_name, model_id)
    resolved_model_id = model_id or (model_info.model_id if model_info else None)
    if not resolved_model_id:
        raise ValueError(f"No models registered for provider '{provider_name}'.")

    capabilities = model_info.capabilities if model_info else {}
    agent_config = AgentConfig(
        model=resolved_model_id,
        system_prompt_path=SYSTEM_PROMPT_PATH,
        thinking_enabled=capabilities.get("thinking", False),
        stream_enabled=capabilities.get("streaming", False),
        supports_vision=capabilities.get("vision", False),
    )

    mcp_defaults = MCPDefaults()
    mcp_config = MCPClientConfig(
        name=mcp_defaults.session_name,
        transport=mcp_defaults.transport,
        url=mcp_defaults.url,
        auth_token=mcp_defaults.auth_token,
        timeout=mcp_defaults.timeout,
        sse_read_timeout=mcp_defaults.sse_read_timeout,
    )

    provider_options = _build_provider_options(provider_name, provider_defaults)

    return RuntimeConfig(
        agent_config=agent_config,
        mcp_config=mcp_config,
        provider=provider_name,
        provider_options=provider_options,
        model_info=model_info,
    )


def _resolve_model(provider: str, model_id: Optional[str]) -> Optional[model_catalog.ModelInfo]:
    if model_id:
        for entry in model_catalog.list_models(provider):
            if entry.model_id == model_id:
                return entry
        return None

    models = model_catalog.list_models(provider)
    if models:
        return models[0]
    return None


def _build_provider_options(provider: str, defaults: ProviderDefaults) -> Dict[str, Any]:
    if provider == "ollama":
        return {"host": defaults.host, "options": dict(defaults.options)}
    return {}
