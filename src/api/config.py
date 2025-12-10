from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from src.agent.base import AgentConfig
from src.config.defaults import RuntimeConfig, make_runtime_config
from src.mcp.client import MCPClientConfig


class StreamRequest(BaseModel):
    """Request payload for the streaming API."""

    user_id: str
    chat_id: str
    messages: List[List[str]] = Field(default_factory=list)

    # MCP configuration
    mcp_transport: Optional[str] = None
    mcp_command: Optional[str] = None
    mcp_args: Optional[List[str]] = None
    mcp_env: Optional[Dict[str, str]] = None
    mcp_cwd: Optional[str] = None
    mcp_url: Optional[str] = None
    mcp_auth_token: Optional[str] = None

    # Provider/model overrides
    provider: Optional[str] = None
    provider_options: Optional[Dict[str, Any]] = None
    ollama_host: Optional[str] = None
    model: Optional[str] = None

    # Agent behavior overrides
    thinking_enabled: Optional[bool] = None
    stream_enabled: Optional[bool] = None
    system_prompt_path: Optional[str] = None


@dataclass
class StreamState:
    """State tracking for stream completion detection."""

    has_content: bool = False
    last_was_tool_call: bool = False


def resolve_default_runtime(provider: Optional[str] = None, model_id: Optional[str] = None) -> RuntimeConfig:
    """Return the shared runtime configuration for the API server."""
    return make_runtime_config(provider, model_id)


def resolve_request_runtime(request: StreamRequest) -> RuntimeConfig:
    """Return a runtime config adjusted for the incoming request overrides."""
    resolved = make_runtime_config(request.provider, request.model)

    agent_config = _build_agent_config(resolved.agent_config, request)
    provider_options = _build_provider_options(resolved.provider_options, request)
    mcp_config = _build_mcp_config(resolved.mcp_config, request)

    return RuntimeConfig(
        agent_config=agent_config,
        mcp_config=mcp_config,
        provider=resolved.provider,
        provider_options=provider_options,
        model_info=resolved.model_info,
    )


def _build_agent_config(base: AgentConfig, request: StreamRequest) -> AgentConfig:
    agent_config = replace(base)

    if request.system_prompt_path:
        agent_config.system_prompt_path = request.system_prompt_path
    if request.thinking_enabled is not None:
        agent_config.thinking_enabled = request.thinking_enabled
    if request.stream_enabled is not None:
        agent_config.stream_enabled = request.stream_enabled

    return agent_config


def _build_provider_options(base: Dict[str, Any], request: StreamRequest) -> Dict[str, Any]:
    options = dict(base)
    if request.provider_options:
        options.update(request.provider_options)
    if request.ollama_host:
        options["host"] = request.ollama_host
    return options


def _build_mcp_config(base: MCPClientConfig, request: StreamRequest) -> MCPClientConfig:
    data = base.model_dump()
    overrides = {
        "transport": request.mcp_transport,
        "command": request.mcp_command,
        "args": request.mcp_args,
        "env": request.mcp_env,
        "cwd": request.mcp_cwd,
        "url": request.mcp_url,
        "auth_token": request.mcp_auth_token,
    }
    for key, value in overrides.items():
        if value is not None:
            data[key] = value

    data["name"] = f"{request.user_id}_{request.chat_id}"
    return MCPClientConfig(**data)
