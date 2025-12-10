from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Sequence

import mcp.types as types


@dataclass
class CapabilityChange:
    added: list[Any]
    removed: list[Any]
    unchanged: list[Any]


class MCPToolMapper(ABC):
    """
    Base class for building provider specific tool specifications from MCP capabilities.
    Implementations expose the tool payloads that should be forwarded to the provider runtime.
    """

    def __init__(self) -> None:
        self._tools_by_name: dict[str, types.Tool] = {}
        self._prompts_by_name: dict[str, types.Prompt] = {}
        self._resources_by_uri: dict[str, types.Resource] = {}
        self._provider_tools: list[Dict[str, Any]] = []

    def update(
        self,
        tools: Iterable[types.Tool] | None,
        prompts: Iterable[types.Prompt] | None,
        resources: Iterable[types.Resource] | None,
    ) -> str | None:
        """Refresh internal caches using the latest MCP capabilities."""
        tool_changes = self._merge_entries(
            current=self._tools_by_name,
            incoming=tools,
            key_fn=lambda item: item.name or "",
        )
        prompt_changes = self._merge_entries(
            current=self._prompts_by_name,
            incoming=prompts,
            key_fn=lambda item: item.name or "",
        )
        resource_changes = self._merge_entries(
            current=self._resources_by_uri,
            incoming=resources,
            key_fn=lambda item: str(item.uri or ""),
        )

        self._provider_tools = self._build_provider_tools()
        return self._format_capability_update(tool_changes, prompt_changes, resource_changes)

    def get_provider_tools(self) -> Sequence[Dict[str, Any]]:
        """Return the provider-specific tool specifications ready to pass into the agent runtime."""
        return list(self._provider_tools)

    def _merge_entries(
        self,
        current: dict[str, Any],
        incoming: Iterable[Any] | None,
        key_fn,
    ) -> CapabilityChange:
        previous = current.copy()
        current.clear()

        for item in incoming or []:
            key = key_fn(item)
            if key:
                current[key] = item

        added = [current[name] for name in current.keys() if name not in previous]
        removed = [previous[name] for name in previous.keys() if name not in current]
        unchanged = [current[name] for name in current.keys() if name in previous]
        return CapabilityChange(added=added, removed=removed, unchanged=unchanged)

    @abstractmethod
    def _build_provider_tools(self) -> list[Dict[str, Any]]:
        """Build the provider-specific tool definitions from cached capabilities."""

    @abstractmethod
    def _format_capability_update(
        self,
        tool_change: CapabilityChange,
        prompt_change: CapabilityChange,
        resource_change: CapabilityChange,
    ) -> str | None:
        """Format a provider specific message describing capability changes."""
