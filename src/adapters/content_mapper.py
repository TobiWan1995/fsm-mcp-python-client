from __future__ import annotations

from abc import ABC, abstractmethod
import base64
from typing import Any, Generic, Iterable, List, Optional, Sequence, Tuple, TypeVar

import mcp.types as types
from mcp.types import (
    AudioContent,
    BlobResourceContents,
    ContentBlock,
    EmbeddedResource,
    ImageContent,
    ListPromptsResult,
    ListResourcesResult,
    ListToolsResult,
    PromptMessage,
    ResourceLink,
    TextContent,
    TextResourceContents,
)

from src.agent.base import BaseAgent

MC = TypeVar("MC")


class MCPContentMapper(ABC, Generic[MC]):
    """
    Base mapper for MCP results.
    Handles the three result shapes returned by MCP:
    - CallToolResult.content (list[ContentBlock])
    - GetPromptResult.messages (list[PromptMessage])
    - ReadResourceResult.contents (list[TextResourceContents | BlobResourceContents])
    Implementations provide concrete handling for blocks, text resources, and blob resources.
    """

    def map_items(self, items: Sequence[Any]) -> Tuple[List[MC], List[dict]]:
        agent_messages: List[MC] = []
        artifacts: List[dict] = []

        for item in self._iter_items(items):
            if isinstance(item, PromptMessage):
                self.handle_prompt_message(item, agent_messages)
            elif isinstance(item, ListToolsResult):
                self.handle_list_method("tool", item, agent_messages)
            elif isinstance(item, ListPromptsResult):
                self.handle_list_method("prompt", item, agent_messages)
            elif isinstance(item, ListResourcesResult):
                self.handle_list_method("resource", item, agent_messages)
            elif isinstance(item, TextResourceContents):
                self.handle_text_resource(item, agent_messages)
            elif isinstance(item, BlobResourceContents):
                self.handle_blob_resource(item, agent_messages, artifacts)
            else:
                self.handle_content_block(item, agent_messages, artifacts)

        return agent_messages, artifacts

    def handle_prompt_message(self, msg: PromptMessage, agent_messages: List[MC]) -> None:
        self.handle_content_block(msg.content, agent_messages, [], role=getattr(msg, "role", None))

    @abstractmethod
    def handle_content_block(
        self,
        block: Any,
        agent_messages: List[MC],
        artifacts: List[dict],
        *,
        role: str | None = None,
    ) -> None:
        """Map a single ContentBlock-like payload."""

    @abstractmethod
    def handle_text_resource(
        self,
        resource: TextResourceContents,
        agent_messages: List[MC],
    ) -> None:
        """Map a text resource payload."""

    @abstractmethod
    def handle_blob_resource(
        self,
        resource: BlobResourceContents,
        agent_messages: List[MC],
        artifacts: List[dict],
    ) -> None:
        """Map a blob resource payload."""

    @abstractmethod
    def handle_list_method(
        self,
        kind: str,
        result: ListToolsResult | ListPromptsResult | ListResourcesResult,
        agent_messages: List[MC],
    ) -> None:
        """Map list_* responses to provider specific messages."""

    @abstractmethod
    def build_provider_messages(self, agent: "BaseAgent", contents: Sequence[MC]) -> List[Any]:
        """Convert mapped content entries into provider-native messages."""

    def _iter_items(self, items: Sequence[Any] | None) -> Iterable[Any]:
        for item in items or []:
            yield from self._coerce_entry(item)

    def _coerce_entry(self, entry: Any) -> Iterable[Any]:
        if entry is None:
            return
        if isinstance(entry, types.ServerResult):
            root = entry.root
            if root is not None:
                yield from self._coerce_entry(root)
            return
        if isinstance(entry, (list, tuple)):
            for sub in entry:
                yield from self._coerce_entry(sub)
            return
        if isinstance(entry, types.CallToolResult):
            yield from self._coerce_entry(entry.content or [])
            return
        if isinstance(entry, types.GetPromptResult):
            for msg in entry.messages or []:
                yield msg
            return
        if isinstance(entry, types.ReadResourceResult):
            yield from self._coerce_entry(entry.contents or [])
            return
        if isinstance(entry, str):
            yield entry
            return
        yield entry

    # Shared helpers for MCP content structures
    def _text_of(self, block: ContentBlock | Any) -> Optional[str]:
        if isinstance(block, TextContent):
            return block.text or ""
        if isinstance(block, (str, int, float, bool)):
            return str(block)
        if isinstance(block, dict) and block.get("type") == "text":
            return str(block.get("text", ""))
        return None

    def _image_data(self, block: ContentBlock | Any) -> Optional[str]:
        if isinstance(block, ImageContent):
            return block.data
        if isinstance(block, dict) and block.get("type") == "image":
            return block.get("data")
        return None

    def _resource_link_text(self, block: ContentBlock | Any) -> Optional[str]:
        if isinstance(block, ResourceLink):
            name = getattr(block, "name", "") or "Resource"
            uri = getattr(block, "uri", "") or ""
            return f"{name}: {uri}".strip()
        if isinstance(block, dict) and block.get("type") == "resource_link":
            name = block.get("name") or "Resource"
            uri = block.get("uri") or ""
            return f"{name}: {uri}".strip()
        return None

    def _embedded_blob(self, block: ContentBlock | Any) -> Optional[dict]:
        if isinstance(block, EmbeddedResource) and isinstance(block.resource, BlobResourceContents):
            mime = getattr(block.resource, "mimeType", None) or getattr(block, "mimeType", None) or "application/octet-stream"
            meta = block.meta if isinstance(getattr(block, "meta", None), dict) else {}
            name = (meta or {}).get("name", "")
            return {"mime": mime, "name": name, "blob_b64": block.resource.blob, "meta": meta}

        if isinstance(block, dict) and block.get("type") == "resource":
            resource_payload = block.get("resource", {})
            if isinstance(resource_payload, dict) and "blob" in resource_payload:
                mime = resource_payload.get("mimeType") or block.get("mimeType") or "application/octet-stream"
                meta = block.get("meta") if isinstance(block.get("meta"), dict) else {}
                name = (meta or {}).get("name", "")
                return {"mime": mime, "name": name, "blob_b64": resource_payload["blob"], "meta": meta}
        return None

    def _is_audio(self, block: ContentBlock | Any) -> bool:
        return isinstance(block, AudioContent) or (isinstance(block, dict) and block.get("type") == "audio")

    def _estimate_blob_size(self, blob_b64: Optional[str]) -> Optional[int]:
        if not blob_b64:
            return None
        padding = blob_b64.count("=")
        size = (len(blob_b64) * 3) // 4 - padding
        return max(size, 0)

    def _decode_blob_text(self, blob_b64: str, mime: str) -> Optional[str]:
        try:
            raw = base64.b64decode(blob_b64)
        except Exception:
            return None
        try:
            if mime.startswith("text/") or mime in {"application/json", "application/xml"}:
                return raw.decode("utf-8")
        except Exception:
            return None
        return None
