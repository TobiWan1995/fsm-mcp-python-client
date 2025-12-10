from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

from mcp.types import (
    BlobResourceContents,
    ContentBlock,
    ListPromptsResult,
    ListResourcesResult,
    ListToolsResult,
    PromptArgument,
    TextResourceContents,
)
from ollama import Message as OllamaMessage

from src.agent.base import AgentConfig, BaseAgent
from src.adapters.content_mapper import MCPContentMapper


MCPBlock = ContentBlock | str


@dataclass
class OllamaMappedContent:
    text: str
    images: Optional[List[str]] = None


class OllamaContentMapper(MCPContentMapper[OllamaMappedContent]):
    """
    Routes MCP content blocks to agent messages and UI artifacts.
    - Text blocks become agent messages.
    - Images are added to the agent when the model supports vision, otherwise emitted as artifacts.
    - Prompt messages expand to individual agent messages.
    - Resource links are mirrored as text.
    - Blobs become artifacts unless explicitly whitelisted for inline rendering.
    - Audio and unknown blocks turn into artifacts.
    """

    def __init__(
        self,
        cfg: AgentConfig,
        *,
        inline_blob_mime_types: Optional[Iterable[str]] = None,
        max_inline_blob_size: int = 512_000,
    ) -> None:
        self._cfg = cfg
        self._inline_blob_mime_types = {m.lower() for m in (inline_blob_mime_types or [])}
        self._max_inline_blob_size = max_inline_blob_size


    def handle_content_block(
        self,
        block: MCPBlock,
        agent_messages: List[OllamaMappedContent],
        artifacts: List[dict],
        *,
        role: str | None = None,
    ) -> None:
        prefix = f"[{role}]: " if role else ""

        text = self._text_of(block)
        if text is not None and text.strip():
            agent_messages.append(OllamaMappedContent(text=f"{prefix}{text}"))
            return

        image_data = self._image_data(block)
        if image_data is not None:
            if self._cfg.supports_vision:
                agent_messages.append(OllamaMappedContent(text=prefix, images=[image_data]))
            else:
                artifacts.append({"kind": "image", "data": image_data, "note": "vision_not_supported"})
            return

        link_text = self._resource_link_text(block)
        if link_text is not None:
            agent_messages.append(OllamaMappedContent(text=f"{prefix}- {link_text}"))
            return

        blob_info = self._embedded_blob(block)
        if blob_info is not None:
            self._handle_blob_info(blob_info, agent_messages, artifacts, prefix=prefix)
            return

        if self._is_audio(block):
            artifacts.append({"kind": "audio"})
            return

        artifacts.append({"kind": "other"})


    def handle_text_resource(
        self,
        resource: TextResourceContents,
        agent_messages: List[OllamaMappedContent],
    ) -> None:
        agent_messages.append(OllamaMappedContent(text=resource.text or ""))


    def handle_blob_resource(
        self,
        resource: BlobResourceContents,
        agent_messages: List[OllamaMappedContent],
        artifacts: List[dict],
    ) -> None:
        blob_info = {
            "mime": (resource.mimeType or "application/octet-stream").lower(),
            "name": resource.uri or "",
            "blob_b64": resource.blob,
            "meta": resource.meta or {},
        }
        self._handle_blob_info(blob_info, agent_messages, artifacts, prefix="")


    def handle_list_method(
        self,
        kind: str,
        result: ListToolsResult | ListPromptsResult | ListResourcesResult,
        agent_messages: List[OllamaMappedContent],
    ) -> None:
        entries: list[str] = []
        if isinstance(result, ListToolsResult):
            entries = [
                self._format_entry(tool.name or "<unnamed>", tool.description, tool.inputSchema or {})
                for tool in result.tools
            ]
        elif isinstance(result, ListPromptsResult):
            entries = [
                self._format_entry(prompt.name or "<unnamed>", prompt.description, self._schema_from_prompt(prompt.arguments or []))
                for prompt in result.prompts
            ]
        elif isinstance(result, ListResourcesResult):
            entries = [
                self._format_entry(str(resource.uri), resource.description, self._schema_from_resource())
                for resource in result.resources
            ]

        if not entries:
            return

        header = "The following callable entries are available:\n"
        body = "\n\n".join(entries)
        agent_messages.append(OllamaMappedContent(text=header + body))


    def _handle_blob_info(
        self,
        blob_info: dict,
        agent_messages: List[OllamaMappedContent],
        artifacts: List[dict],
        *,
        prefix: str,
    ) -> None:
        mime = (blob_info.get("mime") or "").lower()
        size_bytes = self._estimate_blob_size(blob_info.get("blob_b64"))
        if (
            mime in self._inline_blob_mime_types
            and size_bytes is not None
            and size_bytes <= self._max_inline_blob_size
        ):
            text_payload = self._decode_blob_text(blob_info["blob_b64"], mime)
            if text_payload is not None:
                agent_messages.append(OllamaMappedContent(text=f"{prefix}{text_payload}"))
                return
        artifact_payload = {"kind": "blob", **blob_info}
        if size_bytes is not None:
            artifact_payload["size_bytes"] = size_bytes
        artifacts.append(artifact_payload)


    def build_provider_messages(self, agent: BaseAgent, contents: Sequence[OllamaMappedContent]) -> List[OllamaMessage]:
        messages: List[OllamaMessage] = []
        for content in contents:
            message = agent.make_tool_message(content.text, images=content.images)
            assert isinstance(message, OllamaMessage)
            messages.append(message)
        return messages


    def _format_entry(self, name: str, description: Optional[str], schema: dict) -> str:
        schema_text = json.dumps(schema, indent=2)
        desc = description or "No description provided."
        return f"Name: {name}\nDescription: {desc}\nSchema:\n{schema_text}"


    def _schema_from_prompt(self, arguments: List[PromptArgument]) -> dict:
        properties: dict[str, dict[str, str]] = {}
        required: list[str] = []
        for arg in arguments:
            entry: dict[str, str] = {"type": "string"}
            if arg.description:
                entry["description"] = arg.description
            if arg.name:
                properties[arg.name] = entry
                if arg.required:
                    required.append(arg.name)
        return {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        }
    

    def _schema_from_resource(self) -> dict:
        return {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        }
