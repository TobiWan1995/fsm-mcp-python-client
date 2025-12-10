from __future__ import annotations

import json
from typing import Any, Dict, Optional

from src.api.config import StreamState


class APICallbacks:
    """Async callbacks wired into AgentManager for the API server."""

    def __init__(
        self,
        message_queues: Dict[str, Any],
        stream_states: Dict[str, StreamState],
        logger,
    ) -> None:
        self.message_queues = message_queues
        self.stream_states = stream_states
        self.logger = logger

    async def on_agent_response(self, user_id: str, chat_id: str, content: str) -> None:
        await self._enqueue(user_id, chat_id, {"type": "response", "content": content})

    async def on_agent_thinking(self, user_id: str, chat_id: str, thinking: str) -> None:
        await self._enqueue(user_id, chat_id, {"type": "thinking", "content": thinking})

    async def on_agent_tool_call(self, user_id: str, chat_id: str, method: str, params: dict) -> None:
        self.logger.debug("\n[JSON-RPC Call: %s(%s)]", method, json.dumps(params, indent=2))
        await self._enqueue(
            user_id,
            chat_id,
            {"type": "tool_call", "tool": method, "params": params},
        )

    async def on_agent_completion(
        self,
        user_id: str,
        chat_id: str,
        thinking: Optional[str],
        content: Optional[str],
        json_rpc_calls: Optional[list[dict]],
    ) -> None:
        queue_key = self._queue_key(user_id, chat_id)
        state = self.stream_states.setdefault(queue_key, StreamState())
        state.has_content = bool(thinking) or bool(content)
        state.last_was_tool_call = bool(json_rpc_calls)
        self.logger.debug(
            "Streaming state for %s updated to [has_content=%s, last_was_tool_call=%s]",
            queue_key,
            state.has_content,
            state.last_was_tool_call,
        )

    async def _enqueue(self, user_id: str, chat_id: str, payload: dict) -> None:
        queue = self.message_queues.get(self._queue_key(user_id, chat_id))
        if queue:
            await queue.put(payload)

    @staticmethod
    def _queue_key(user_id: str, chat_id: str) -> str:
        return f"{user_id}:{chat_id}"
