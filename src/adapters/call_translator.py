from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, Sequence

import mcp.types as types


class MCPCallTranslator(ABC):
    """
    Base class for converting provider specific tool call payloads into MCP JSON-RPC requests.
    Implementations typically keep their own capability caches to resolve provider identifiers.
    """

    def update_capabilities(
        self,
        tools: Iterable[types.Tool] | None,
        prompts: Iterable[types.Prompt] | None,
        resources: Iterable[types.Resource] | None,
    ) -> None:
        """Refresh capability caches required for translating provider tool calls."""
        return None

    @abstractmethod
    def extract_tool_calls(self, payload: Any) -> Sequence[Any]:
        """
        Normalize the provider specific payload into a flat sequence of tool call entries.
        Implementations may return dictionaries, dataclasses, or SDK specific objects that
        can later be consumed by `to_json_rpc`.
        """

    @abstractmethod
    def to_json_rpc(self, tool_call: Any, *, rpc_id: Any) -> Dict[str, Any]:
        """Translate a single provider tool call entry into an MCP compliant JSON-RPC request."""
