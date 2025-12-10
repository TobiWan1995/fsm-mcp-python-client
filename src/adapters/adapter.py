from __future__ import annotations

import logging
from typing import Iterable, Sequence, Any, Dict, List, Optional, Tuple

import mcp.types as types

from src.adapters.call_translator import MCPCallTranslator
from src.adapters.content_mapper import MCPContentMapper
from src.adapters.tool_mapper import MCPToolMapper
from src.agent.base import BaseAgent

MCPBlock = types.ContentBlock | types.PromptMessage | types.TextResourceContents | types.BlobResourceContents


logger = logging.getLogger(__name__)


class MCPAdapter:
    """
    Generic MCP adapter that coordinates capability updates, tool specification mapping,
    tool call translation, and MCP content mapping for provider specific agents.
    """

    def __init__(
        self,
        *,
        content_mapper: MCPContentMapper,
        call_translator: MCPCallTranslator,
        tool_mapper: MCPToolMapper,
    ) -> None:
        self._tools: list[types.Tool] = []
        self._prompts: list[types.Prompt] = []
        self._resources: list[types.Resource] = []
        self._content_mapper = content_mapper
        self._call_translator = call_translator
        self._tool_mapper = tool_mapper

    # ----------------------------
    # Capability lifecycle
    # ----------------------------

    def update_capabilities(
        self,
        tools: Iterable[types.Tool] | None,
        prompts: Iterable[types.Prompt] | None,
        resources: Iterable[types.Resource] | None,
    ) -> Optional[str]:
        """
        Refresh cached MCP capabilities and propagate them to the mapper/translator helpers.
        """
        self._tools = list(tools or [])
        self._prompts = list(prompts or [])
        self._resources = list(resources or [])
        summary = self._tool_mapper.update(self._tools, self._prompts, self._resources)
        self._call_translator.update_capabilities(self._tools, self._prompts, self._resources)
        self._after_capability_update()
        return summary

    def _after_capability_update(self) -> None:
        """Hook for subclasses when capability caches changed."""
        return None

    # ----------------------------
    # Provider integration hooks
    # ----------------------------

    def adapt_model_call_to_mcp(self, payload: Any) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """
        Translate a provider specific tool call payload into one or many MCP JSON-RPC
        requests ready to be forwarded to the MCP client.
        """
        tool_calls = self._call_translator.extract_tool_calls(payload)
        requests: list[Dict[str, Any]] = []
        failures: list[str] = []
        for idx, call in enumerate(tool_calls, start=1):
            try:
                requests.append(self._call_translator.to_json_rpc(call, rpc_id=idx))
            except Exception as exc:
                logger.error("Failed to translate provider tool_call %s: %s", call, exc)
                failures.append(f"{self._describe_tool_call(call)} -> {exc}")
        error_message = self._format_tool_mapping_failure(failures) if failures else None
        return requests, error_message

    def to_backend_tools(self) -> List[Dict[str, Any]]:
        """
        Return provider specific tool specifications that must be supplied to
        the underlying model runtime on each call.
        """
        return list(self._tool_mapper.get_provider_tools())

    def build_provider_messages(
        self,
        agent: BaseAgent,
        payloads: Sequence[Any],
        role: Optional[str],
    ) -> Tuple[List[Any], List[Dict[str, Any]]]:
        """
        Build provider-specific message objects (ready to be appended to the agent history)
        from queued payloads. Uses the configured content mapper for non-user payloads.
        """
        if role == "user":
            return self._build_user_messages(agent, payloads), []

        processed, artifacts = self._content_mapper.map_items(payloads or [])
        if not processed:
            return [], artifacts
        messages = self._content_mapper.build_provider_messages(agent, processed)
        return messages, artifacts

    # ----------------------------
    # Introspection helpers
    # ----------------------------

    @property
    def tools(self) -> Sequence[types.Tool]:
        return tuple(self._tools)

    @property
    def prompts(self) -> Sequence[types.Prompt]:
        return tuple(self._prompts)

    @property
    def resources(self) -> Sequence[types.Resource]:
        return tuple(self._resources)

    # ----------------------------
    # Helpers
    # ----------------------------

    def _build_user_messages(self, agent: BaseAgent, payloads: Sequence[Any]) -> List[Any]:
        messages: List[Any] = []
        for item in payloads or []:
            text = item if isinstance(item, str) else str(item)
            messages.append(agent.make_user_message(text))
        return messages

    def _describe_tool_call(self, call: Any) -> str:
        if isinstance(call, dict):
            function = call.get("function")
            if isinstance(function, dict):
                name = function.get("name") or "<unnamed>"
                return f"function:{name}"
        return str(call)

    def _format_tool_mapping_failure(self, failures: Sequence[str]) -> str:
        if not failures:
            return ""
        tool_specs = self.to_backend_tools()
        tool_names = sorted(
            {
                spec.get("function", {}).get("name")
                or spec.get("name")
                or ""
                for spec in tool_specs
                if isinstance(spec, dict)
            }
        )
        available = ", ".join(name for name in tool_names if name)
        suffix = f" Available tools: {available}" if available else ""
        details = " ; ".join(failures)
        return (
            "Requested tool or resource could not be mapped. "
            "Check the currently available tools; availability can change during the session."
            f"{suffix} | Details: {details}"
        )
