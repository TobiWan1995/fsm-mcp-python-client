from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Optional, Sequence

from prompt_toolkit import print_formatted_text
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.styles import Style

if TYPE_CHECKING:
    from src.cli.main import CLIClient
    from src.cli.writer import ConversationWriter


class CLICallbacks:
    """Aggregate callbacks used by CLIClient."""

    def __init__(self, client: "CLIClient", style: Style) -> None:
        self.client = client
        self.style = style
        self.writer: ConversationWriter | None = None

    async def on_agent_response(self, user_id: str, chat_id: str, content: str) -> None:
        client = self.client
        if client.stream_buffer and client.agent_config.stream_enabled:
            self._stream_response_delta(user_id, chat_id, content)
        else:
            if client.is_thinking:
                print_formatted_text(
                    FormattedText([("class:thinking", "]")]),
                    style=self.style,
                )
                client.is_thinking = False

            print_formatted_text(
                FormattedText([("class:assistant", f"\nAssistant: {content}")]),
                style=self.style,
            )
        if self.writer:
            self.writer.record_agent(content)

    async def on_agent_thinking(self, user_id: str, chat_id: str, thinking: str) -> None:
        client = self.client
        if not client.agent_config.thinking_enabled:
            return

        if client.stream_buffer and client.agent_config.stream_enabled:
            self._stream_thinking_delta(user_id, chat_id, thinking)
        else:
            print_formatted_text(
                FormattedText([("class:thinking", f"\n[Thinking: {thinking}]")]),
                style=self.style,
            )
        if self.writer:
            self.writer.record_thinking(thinking)

    async def on_agent_tool_call(self, user_id: str, chat_id: str, method: str, params: dict) -> None:
        client = self.client
        if client.is_streaming:
            if client.is_thinking:
                print_formatted_text(
                    FormattedText([("class:thinking", "]")]),
                    style=self.style,
                )
                client.is_thinking = False
            print()
            client.is_streaming = False

        print_formatted_text(
            FormattedText([("class:tool", f"\n[JSON-RPC Call: {method}({json.dumps(params, indent=2)})]")]),
            style=self.style,
        )
        if self.writer:
            self.writer.record_tool_call(method, params)

    async def on_tool_response(self, user_id: str, chat_id: str, content: str) -> None:
        if not content.strip():
            return
        print_formatted_text(
            FormattedText([("class:tool_response", f"\n[Tool]: {content}")]),
            style=self.style,
        )
        if self.writer:
            self.writer.record_tool_response(content)

    async def on_agent_completion(
        self,
        user_id: str,
        chat_id: str,
        thinking: str | None,
        content: str | None,
        json_rpc_calls: Optional[Sequence[dict]] = None,
    ) -> None:
        client = self.client
        if client.stream_buffer and client.agent_config.stream_enabled:
            if content:
                self._stream_response_delta(user_id, chat_id, content)
            if thinking and client.agent_config.thinking_enabled:
                self._stream_thinking_delta(user_id, chat_id, thinking)
        await asyncio.sleep(0.05)
        if client.is_streaming:
            client.finalize_streaming()
        if client.stream_buffer:
            client.stream_buffer.clear(user_id, chat_id)

    def _stream_response_delta(self, user_id: str, chat_id: str, content: str) -> None:
        client = self.client
        if not client.stream_buffer:
            return

        delta, is_first = client.stream_buffer.get_delta(
            user_id=user_id,
            chat_id=chat_id,
            content=content,
            buffer_type="response",
        )
        if not delta:
            return

        if is_first:
            if client.is_thinking:
                print_formatted_text(
                    FormattedText([("class:thinking", "]")]),
                    style=self.style,
                )
                print()
                client.is_thinking = False

            print_formatted_text(
                FormattedText([("class:assistant", "\n[Assistant]: ")]),
                style=self.style,
                end="",
            )

        print_formatted_text(
            FormattedText([("class:assistant", delta)]),
            style=self.style,
            end="",
        )
        client.flush_stdout()
        client.is_streaming = True

    def _stream_thinking_delta(self, user_id: str, chat_id: str, thinking: str) -> None:
        client = self.client
        if not client.stream_buffer:
            return

        delta, is_first = client.stream_buffer.get_delta(
            user_id=user_id,
            chat_id=chat_id,
            content=thinking,
            buffer_type="thinking",
        )
        if not delta:
            return

        if is_first:
            print_formatted_text(
                FormattedText([("class:thinking", "\n[Thinking: ")]),
                style=self.style,
                end="",
            )
            client.is_thinking = True

        print_formatted_text(
            FormattedText([("class:thinking", delta)]),
            style=self.style,
            end="",
        )
        client.flush_stdout()
        client.is_streaming = True
