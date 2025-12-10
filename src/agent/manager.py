import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import mcp.types as types

from src.adapters.adapter import MCPAdapter
from src.adapters.provider_registry import create_provider_bundle
from src.agent.base import AgentConfig, BaseAgent
from src.mcp.client import MCPClient, MCPClientConfig
from src.mcp.sampling import SessionAwareSamplingHandler
from src.util.file.file_handler import BaseFileHandler
from src.util.file.file_handler_openwebui import OpenWebUIMarkdownFileHandler

QueuedEntry = Tuple[types.ServerResult | str, Optional[str]]
QueuedTurn = list[QueuedEntry]

logger = logging.getLogger(__name__)


@dataclass
class AgentSession:
    """Represents a single agent session."""

    session_id: str
    user_id: str
    chat_id: str
    provider: str
    agent: BaseAgent
    adapter: MCPAdapter
    mcp_client: MCPClient
    task: Optional[asyncio.Task] = None
    message_queue: asyncio.Queue[QueuedTurn] = field(default_factory=asyncio.Queue)
    pending_turn: QueuedTurn = field(default_factory=list)
    active: bool = True


class AgentManager:
    """Manages multiple agent sessions while staying provider agnostic."""

    def __init__(
        self,
        *,
        default_provider: str = "ollama",
        default_model: str = "llama3.2:3b",
        system_prompt_path: Optional[str] = None,
        provider_defaults: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        self.sessions: Dict[str, AgentSession] = {}
        self.default_provider = default_provider
        self.default_model = default_model
        self.system_prompt_path = system_prompt_path
        self.system_prompt_template = self._load_system_prompt()

        self.provider_defaults: Dict[str, Dict[str, Any]] = provider_defaults or {}
        self.provider_defaults.setdefault("ollama", {"host": "http://localhost:11434"})

        self.sampling_handler = SessionAwareSamplingHandler(self)

        self.on_agent_response: Optional[Callable[[str, str, str], Any]] = None
        self.on_agent_thinking: Optional[Callable[[str, str, str], Any]] = None
        self.on_agent_tool_call: Optional[Callable[[str, str, str, Dict[str, Any]], Any]] = None
        self.on_tool_response: Optional[Callable[[str, str, str], Any]] = None
        self.on_agent_completion: Optional[Callable[[str, str, Optional[str], Optional[str], Optional[Sequence[dict]]], Any]] = None

        self.file_handler: BaseFileHandler | None = OpenWebUIMarkdownFileHandler()

    # ----------------------------
    # Session lifecycle
    # ----------------------------

    def attach_file_handler(self, handler: BaseFileHandler) -> None:
        self.file_handler = handler

    def _load_system_prompt(self) -> str:
        if self.system_prompt_path and Path(self.system_prompt_path).exists():
            return Path(self.system_prompt_path).read_text(encoding="utf-8")
        return ""

    def get_session_key(self, user_id: str, chat_id: str) -> str:
        return f"{user_id}:{chat_id}"

    async def create_session(
        self,
        user_id: str,
        chat_id: str,
        mcp_config: MCPClientConfig,
        *,
        agent_config: Optional[AgentConfig] = None,
        provider: Optional[str] = None,
        provider_options: Optional[Dict[str, Any]] = None,
    ) -> AgentSession:
        session_key = self.get_session_key(user_id, chat_id)
        if session_key in self.sessions:
            logger.debug("Session already exists for %s", session_key)
            return self.sessions[session_key]

        provider_name = (provider or self.default_provider).lower()
        cfg = agent_config or AgentConfig(model=self.default_model)

        defaults = dict(self.provider_defaults.get(provider_name, {}))
        if provider_options:
            defaults.update(provider_options)

        agent, adapter = create_provider_bundle(provider=provider_name, agent_config=cfg, options=defaults)
        agent.set_system_prompt(self.system_prompt_template)

        mcp_client = MCPClient(
            session_key=session_key,
            config=mcp_config,
            sampling_handler=self.sampling_handler
        )

        session = AgentSession(
            session_id=str(uuid.uuid4()),
            user_id=user_id,
            chat_id=chat_id,
            provider=provider_name,
            agent=agent,
            adapter=adapter,
            mcp_client=mcp_client,
        )

        async def capability_handler(tools, prompts, resources):
            summary = adapter.update_capabilities(tools, prompts, resources)
            agent.set_active_tools(adapter.to_backend_tools())
            if summary:
                session.pending_turn.append((summary, "tool"))
                logger.debug("Capabilities updated for session %s:\n%s", session_key, summary)

        mcp_client.on_capabilities_changed = capability_handler
        initialized = await mcp_client.initialize()
        if not initialized:
            raise RuntimeError(f"Failed to initialize MCP client for session {session_key}")

        agent.set_active_tools(adapter.to_backend_tools())
        session.task = asyncio.create_task(self._run_agent_loop(session))
        self.sessions[session_key] = session

        logger.debug("Created session %s (%s)", session.session_id, session_key)
        return session

    # ----------------------------
    # Core loop
    # ----------------------------

    async def _run_agent_loop(self, session: AgentSession) -> None:
        logger.debug("Starting agent loop for session %s", session.session_id)
        while session.active:
            try:
                turn = await asyncio.wait_for(session.message_queue.get(), timeout=3.0)
                session.pending_turn.clear()
                await self._process_turn(session, turn)
            except asyncio.TimeoutError:
                continue
            except Exception as exc:  # pragma: no cover - defensive
                logger.error("Agent loop error for session %s: %s", session.session_id, exc, exc_info=True)
                await asyncio.sleep(0.1)
        logger.debug("Agent loop closed for session %s", session.session_id)

    async def _process_turn(self, session: AgentSession, turn: QueuedTurn) -> None:
        if not turn:
            return

        provider_messages = await self._prepare_turn_messages(session, turn)
        if not provider_messages:
            return

        agent = session.agent

        logger.debug("Processing turn for session %s with %d entries", session.session_id, len(turn))

        try:
            if agent.config.stream_enabled:
                await self._handle_streaming_response(session, provider_messages)
            else:
                await self._handle_sync_response(session, provider_messages)
        except Exception as exc:
            logger.error(
                "Error while processing turn for session %s: %s",
                session.session_id,
                exc,
                exc_info=True,
            )
        finally:
            await self._enqueue_turn(session)
                

    async def _handle_streaming_response(self, session: AgentSession, new_messages: Sequence[Any]) -> None:
        agent = session.agent
        adapter = session.adapter

        last_thinking: Optional[str] = None
        thinking_buffer: str = ""
        content_buffer: str = ""
        last_request: Optional[dict[str, Any]] = None

        async for thinking, content, tool_payload in agent.generate_response(new_messages):
            if thinking:
                thinking_buffer += thinking
                last_thinking = thinking_buffer
                if self.on_agent_thinking:
                    await self.on_agent_thinking(session.user_id, session.chat_id, thinking_buffer)

            if content and content.strip():
                content_buffer += content
                if self.on_agent_response:
                    await self.on_agent_response(session.user_id, session.chat_id, content_buffer)

            if tool_payload:
                requests, mapping_error = adapter.adapt_model_call_to_mcp(tool_payload)
                if requests:
                    last_request = requests[-1]
                if mapping_error:
                    session.pending_turn.append((mapping_error, "tool"))
                for request in requests:
                    session.pending_turn.append((await self._execute_single_request(session, request), "tool"))

        if self.on_agent_completion:
            final_content = content_buffer.strip() or None
            await self.on_agent_completion(
                session.user_id,
                session.chat_id,
                last_thinking,
                final_content,
                [last_request] if last_request else None,
            )

    async def _handle_sync_response(self, session: AgentSession, new_messages: Sequence[Any]) -> None:
        agent = session.agent
        adapter = session.adapter

        thinking, content, tool_payload = await agent.generate_response_sync(new_messages)

        if thinking and self.on_agent_thinking:
            await self.on_agent_thinking(session.user_id, session.chat_id, thinking)

        if content.strip() and self.on_agent_response:
            await self.on_agent_response(session.user_id, session.chat_id, content)

        requests: list[dict[str, Any]] = []
        if tool_payload:
            requests, mapping_error = adapter.adapt_model_call_to_mcp(tool_payload)
            if mapping_error:
                session.pending_turn.append((mapping_error, "tool"))

        for request in requests:
            session.pending_turn.append((await self._execute_single_request(session, request), "tool"))

        if self.on_agent_completion:
            await self.on_agent_completion(
                session.user_id,
                session.chat_id,
                thinking,
                content.strip() or None,
                requests,
            )

    async def _execute_single_request(self, session: AgentSession, request: dict[str, Any]) -> types.ServerResult | str:
        method = request.get("method", "unknown")
        params = request.get("params", {})

        logger.debug("Executing MCP call via %s with params %s", method, params)

        if self.on_agent_tool_call:
            await self.on_agent_tool_call(session.user_id, session.chat_id, method, params)

        try:
            return await session.mcp_client.execute_json_rpc(request)
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Error during MCP request %s: %s", method, exc, exc_info=True)
            error_msg = f"Error while executing {method}: {exc}"
            return error_msg

    async def _prepare_turn_messages(self, session: AgentSession, turn: Sequence[QueuedEntry]) -> List[Any]:
        provider_messages: List[Any] = []
        for payload, role in turn:
            payload_seq: Sequence[Any]
            if isinstance(payload, (list, tuple)):
                payload_seq = list(payload)
            else:
                payload_seq = [payload]
            messages, artifacts = session.adapter.build_provider_messages(session.agent, payload_seq, role)
            provider_messages.extend(messages)

            if self.on_tool_response and role == "tool":
                for message in messages:
                    content = getattr(message, "content", None)
                    if isinstance(content, str) and content.strip():
                        await self.on_tool_response(session.user_id, session.chat_id, content)

            if self.file_handler and self.on_agent_response:
                for artifact in artifacts:
                    if artifact.get("kind") == "blob":
                        rendered = await self.file_handler.stringify_if_supported(
                            mime_type=artifact.get("mime"),
                            blob_b64=artifact.get("blob_b64"),
                            name=artifact.get("name"),
                            meta=artifact.get("meta"),
                        )
                        if rendered:
                            await self.on_agent_response(session.user_id, session.chat_id, rendered)

        return provider_messages

    async def _enqueue_turn(self, session: AgentSession) -> None:
        if not session.pending_turn:
            return
        try:
            entries = list(session.pending_turn)
            session.pending_turn.clear()
            await session.message_queue.put(entries)
        except Exception as queue_exc:  # pragma: no cover - defensive
            logger.error(
                "Failed to enqueue turn for session %s: %s", session.session_id, queue_exc, exc_info=True
            )

    # ----------------------------
    # Public API
    # ----------------------------

    async def send_message(self, user_id: str, chat_id: str, message: str) -> None:
        session_key = self.get_session_key(user_id, chat_id)
        if session_key not in self.sessions:
            raise ValueError(f"No active session for {session_key}")
        session = self.sessions.get(session_key)
        session.pending_turn.append((message, "user"))
        await self._enqueue_turn(session)

    async def end_session(self, user_id: str, chat_id: str) -> None:
        session_key = self.get_session_key(user_id, chat_id)
        session = self.sessions.get(session_key)
        if not session:
            return

        session.active = False
        if session.task:
            await session.task

        await session.mcp_client.teardown()
        self.sessions.pop(session_key, None)
        logger.debug("Ended session %s (%s)", session.session_id, session_key)

    async def shutdown(self) -> None:
        logger.debug("Shutting down all sessions...")
        await asyncio.gather(*(self.end_session(*key.split(":", 1)) for key in list(self.sessions.keys())), return_exceptions=True)
        logger.debug("Shutdown complete")
