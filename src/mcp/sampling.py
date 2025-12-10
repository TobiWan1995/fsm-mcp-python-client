from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Optional, TYPE_CHECKING

import mcp.types as types
from mcp.client.session import ClientSession
from mcp.shared.context import RequestContext
from ollama import Message as OllamaMessage

if TYPE_CHECKING:
    from agent.manager import AgentManager, AgentSession

logger = logging.getLogger(__name__)

# TODO: This is heavily provider specific and needs to be changed:
# - Maybe create an new Agent?
# - Maybe create a sampling interface on base agent
# - how to handle options?

class SessionAwareSamplingHandler:
    """
    Global sampling throttle across ALL MCP clients/sessions.
    - Enforces a single concurrency cap (Semaphore) for outbound model calls
    - Resolves session -> AgentSession via AgentManager
    - Handles timeouts, cancellation, basic payload validation
    """

    def __init__(
        self,
        agent_manager: AgentManager,
        *,
        max_concurrency: int = 10,
        request_timeout_s: float = 60.0,
    ) -> None:
        self._manager = agent_manager
        self._global_sem = asyncio.Semaphore(max_concurrency)
        self._timeout_s = request_timeout_s

        self._session_locks: dict[str, asyncio.Lock] = {}

        self._inflight = 0
        self._completed = 0
        self._rejected = 0

    def _lock_for(self, session_key: str) -> asyncio.Lock:
        if session_key not in self._session_locks:
            self._session_locks[session_key] = asyncio.Lock()
        return self._session_locks[session_key]

    def _get_session(self, session_key: str) -> Optional[AgentSession]:
        return self._manager.sessions.get(session_key)

    async def sample(
        self,
        session_key: str,
        ctx: RequestContext[ClientSession, Any],
        params: types.CreateMessageRequestParams,
    ) -> types.CreateMessageResult | types.ErrorData:
        session = self._get_session(session_key)
        if not session or not session.active:
            self._rejected += 1
            return types.ErrorData(
                code=types.INTERNAL_ERROR,
                message=f"Sampling failed: unknown or inactive session '{session_key}'",
            )

        if session.provider != "ollama":
            self._rejected += 1
            return types.ErrorData(
                code=types.INTERNAL_ERROR,
                message=f"Sampling not supported for provider '{session.provider}'",
            )

        try:
            provider_messages = self._to_ollama_messages(params)
        except ValueError as exc:
            self._rejected += 1
            return types.ErrorData(code=types.INVALID_REQUEST, message=str(exc))

        agent = session.agent
        if not hasattr(agent, "client"):
            self._rejected += 1
            return types.ErrorData(code=types.INTERNAL_ERROR, message="Sampling failed: agent has no client")

        start = time.perf_counter()

        async with self._global_sem:
            self._inflight += 1
            try:
                logger.debug("START: Sampling for %s", session_key)
                response = await asyncio.wait_for(
                    agent.client.chat(
                        model=agent.config.model,
                        messages=provider_messages,
                        options=agent.options,
                        stream=False,
                        think=agent.config.thinking_enabled,
                        tools=agent.active_tools or None,
                    ),
                    timeout=self._timeout_s,
                )
                logger.debug("FINISH: Sampling for %s", session_key)
            except asyncio.TimeoutError:
                self._rejected += 1
                logger.warning("Sampling timeout | session=%s", session_key)
                return types.ErrorData(code=types.INTERNAL_ERROR, message="Sampling timed out")
            except asyncio.CancelledError:
                self._rejected += 1
                logger.debug("Sampling cancelled | session=%s", session_key)
                raise
            except Exception as exc:  # pragma: no cover - defensive
                self._rejected += 1
                logger.exception("Sampling failed | session=%s", session_key)
                return types.ErrorData(code=types.INTERNAL_ERROR, message=f"Sampling failed: {exc}")
            finally:
                self._inflight -= 1
                elapsed = (time.perf_counter() - start) * 1000
                self._completed += 1
                logger.debug(
                    "Sampling done | session=%s inflight=%d completed=%d rejected=%d ms=%.1f",
                    session_key,
                    self._inflight,
                    self._completed,
                    self._rejected,
                    elapsed,
                )

        content: str = (response.message.content or "").strip()
        return types.CreateMessageResult(
            role="assistant",
            content=types.TextContent(text=content, type="text"),
            model=agent.config.model,
            stopReason=None,
        )

    @staticmethod
    def _to_ollama_messages(params: types.CreateMessageRequestParams) -> list[OllamaMessage]:
        messages: list[OllamaMessage] = []

        system_prompt = getattr(params, "systemPrompt", None)
        if system_prompt:
            messages.append(OllamaMessage(role="system", content=system_prompt))

        sampling_msgs: list[types.SamplingMessage] = list(params.messages or [])
        if not sampling_msgs:
            raise ValueError("Sampling expects at least one message")

        for msg in sampling_msgs:
            if not isinstance(msg.content, types.TextContent):
                raise ValueError("Sampling expects TextContent only")
            role = msg.role if isinstance(msg.role, str) else getattr(msg.role, "value", "user")
            messages.append(OllamaMessage(role=role, content=msg.content.text))

        return messages
