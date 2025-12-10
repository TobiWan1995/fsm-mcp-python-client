from __future__ import annotations

import json
import logging
from typing import Any, AsyncGenerator, Dict, List, Optional, Sequence, Set

from ollama import (
    AsyncClient,
    ChatResponse,
    Client,
    Message as OllamaMessage,
    Options,
)

from src.agent.base import AgentConfig, BaseAgent

logger = logging.getLogger(__name__)


def _stable_call_key(call: Dict[str, object]) -> str:
    """Return a deterministic key for deduplicating tool calls."""
    return json.dumps(call, sort_keys=True, separators=(",", ":"))


class OllamaAgent(BaseAgent):
    """
    Ollama specific agent implementation that works with structured tool calls.
    """

    def __init__(
        self,
        config: AgentConfig,
        host: str = "http://localhost:11434",
        options: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(config)
        self.host = host
        self.client = AsyncClient(host=host)
        self.sync_client = Client(host=host)
        self.options = self._init_options(options)
        self.config.options = self.options


    # ----------------------------
    # BaseAgent hooks
    # ----------------------------

    def make_user_message(
        self,
        content: str,
        *,
        images: Optional[List[str]] = None,
    ) -> OllamaMessage:
        message = OllamaMessage(role="user", content=content or "")
        if images:
            message.images = list(images)
        return message

    def make_system_message(self, content: str) -> OllamaMessage:
        return OllamaMessage(role="system", content=content or "")

    def make_tool_message(
        self,
        content: str,
        *,
        name: Optional[str] = None,
        images: Optional[List[str]] = None,
    ) -> OllamaMessage:
        message = OllamaMessage(role="tool", content=content or "")
        if name:
            message.tool_name = name
        if images:
            message.images = list(images)
        return message

    def make_assistant_message(
        self,
        content: str,
        *,
        thinking: Optional[str] = None,
        tool_calls: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> OllamaMessage:
        message = OllamaMessage(role="assistant", content=content or "")
        if thinking and self.config.thinking_enabled:
            message.thinking = thinking
        if tool_calls:
            message.tool_calls = [self._to_tool_call(call) for call in tool_calls]
        return message

    def _is_system_message(self, message: OllamaMessage) -> bool:
        return isinstance(message, OllamaMessage) and message.role == "system"

    # ----------------------------
    # Response generation
    # ----------------------------

    async def generate_response(
        self,
        new_messages: Sequence[OllamaMessage],
    ) -> AsyncGenerator[tuple[Optional[str], str, Optional[List[Dict[str, Any]]]], None]:
        for message in new_messages:
            self.add_message(message)

        ollama_messages = self._convert_messages_for_ollama()
        stream = await self.client.chat(
            model=self.config.model,
            messages=ollama_messages,
            options=self.options,
            stream=True,
            think=self.config.thinking_enabled,
            tools=self.active_tools or None,
        )

        accumulated_content = ""
        accumulated_thinking = ""
        seen_call_keys: Set[str] = set()
        collected_calls: List[Dict[str, Any]] = []

        async for response in stream:
            chat_response = self._ensure_chat_response(response)
            msg = chat_response.message
            chunk_content = msg.content or ""
            chunk_thinking = msg.thinking or ""
            tool_calls = self._normalize_tool_calls(msg.tool_calls or [])

            if chunk_content:
                accumulated_content += chunk_content

            if chunk_thinking:
                accumulated_thinking += chunk_thinking

            new_calls: List[Dict[str, Any]] = []
            for call in tool_calls:
                key = _stable_call_key(call)
                if key not in seen_call_keys:
                    seen_call_keys.add(key)
                    collected_calls.append(call)
                    new_calls.append(call)

            if not chunk_content and not chunk_thinking and not new_calls:
                continue

            yield (
                accumulated_thinking if chunk_thinking else None,
                chunk_content,
                new_calls or None,
            )

        if accumulated_content or accumulated_thinking or collected_calls:
            assistant_message = self.make_assistant_message(
                accumulated_content,
                thinking=accumulated_thinking or None,
                tool_calls=collected_calls or None,
            )
            self.add_message(assistant_message)

    async def generate_response_sync(
        self,
        new_messages: Sequence[OllamaMessage],
    ) -> tuple[Optional[str], str, List[Dict[str, Any]]]:
        for message in new_messages:
            self.add_message(message)

        ollama_messages = self._convert_messages_for_ollama()

        response = await self.client.chat(
            model=self.config.model,
            messages=ollama_messages,
            options=self.options,
            stream=False,
            think=self.config.thinking_enabled,
            tools=self.active_tools or None,
        )

        chat_response = self._ensure_chat_response(response)
        content = chat_response.message.content or ""
        thinking = chat_response.message.thinking if self.config.thinking_enabled else None
        tool_calls = self._normalize_tool_calls(chat_response.message.tool_calls or [])

        assistant_message = self.make_assistant_message(
            content,
            thinking=thinking,
            tool_calls=tool_calls or None,
        )
        self.add_message(assistant_message)

        return thinking, content, tool_calls

    # ----------------------------
    # Helpers
    # ----------------------------

    def _convert_messages_for_ollama(self) -> List[OllamaMessage]:
        """Return the current history as Ollama message objects."""
        return [msg for msg in self.history if isinstance(msg, OllamaMessage)]

    @staticmethod
    def _normalize_tool_calls(tool_calls: Optional[Sequence[Any]]) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        if not tool_calls:
            return normalized

        for call in tool_calls:
            if hasattr(call, "model_dump"):
                call_dict = call.model_dump()
            elif isinstance(call, dict):
                call_dict = call
            else:
                raise ValueError(f"Unsupported tool_call entry: {call!r}")
            normalized.append(call_dict)
        return normalized

    @staticmethod
    def _ensure_chat_response(response: Any) -> ChatResponse:
        if isinstance(response, ChatResponse):
            return response
        if isinstance(response, dict):
            return ChatResponse.model_validate(response)
        raise TypeError(f"Unexpected response type from Ollama client: {type(response).__name__}")

    @staticmethod
    def _init_options(options: Any) -> Options:
        if options is None:
            return Options(temperature=0.1, top_p=0.8, top_k=10, num_ctx=50000)
        if isinstance(options, Options):
            return options
        if isinstance(options, dict):
            return Options(**options)
        raise TypeError(f"Unsupported options type for OllamaAgent: {type(options).__name__}")

    @staticmethod
    def _to_tool_call(call: Any) -> OllamaMessage.ToolCall:
        if isinstance(call, OllamaMessage.ToolCall):
            return call
        if isinstance(call, dict):
            return OllamaMessage.ToolCall.model_validate(call)
        raise TypeError(f"Unsupported tool_call payload for history: {type(call).__name__}")
