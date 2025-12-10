from __future__ import annotations

import asyncio
import json
import logging
from collections import defaultdict
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from src.api.callbacks import APICallbacks
from src.api.config import (
    StreamRequest,
    StreamState,
    resolve_default_runtime,
    resolve_request_runtime,
)
from src.agent.manager import AgentManager
from src.config.defaults import RuntimeConfig
from src.util.stream_buffer import StreamBuffer

logger = logging.getLogger(__name__)

agent_manager: Optional[AgentManager] = None
default_runtime_config: Optional[RuntimeConfig] = None
api_callbacks: Optional[APICallbacks] = None

stream_buffer = StreamBuffer()
message_queues: Dict[str, asyncio.Queue] = defaultdict(asyncio.Queue)
stream_states: Dict[str, StreamState] = {}


# ----------------------------
# Helper functions
# ----------------------------


async def ensure_session_exists(request: StreamRequest) -> RuntimeConfig:
    """Ensure an agent session exists for the given request and return the runtime config."""
    if not agent_manager:
        raise RuntimeError("Agent manager is not initialized")

    runtime = resolve_request_runtime(request)
    session_key = agent_manager.get_session_key(request.user_id, request.chat_id)

    if session_key not in agent_manager.sessions:
        logger.debug("Creating new session for %s:%s", request.user_id, request.chat_id)
        await agent_manager.create_session(
            user_id=request.user_id,
            chat_id=request.chat_id,
            mcp_config=runtime.mcp_config,
            agent_config=runtime.agent_config,
            provider=runtime.provider,
            provider_options=runtime.provider_options,
        )
    else:
        logger.debug("Using existing session for %s:%s", request.user_id, request.chat_id)

    return runtime


async def send_user_message(request: StreamRequest) -> None:
    """Send the latest user message to the agent if present."""
    if not agent_manager:
        raise RuntimeError("Agent manager is not initialized")
    if not request.messages:
        return

    latest_message = request.messages[-1]
    if latest_message[0] == "user":
        await agent_manager.send_message(
            request.user_id,
            request.chat_id,
            latest_message[1],
        )


def format_sse_message(message_type: str, content: Any) -> str:
    """Format a server-sent event payload."""
    if message_type == "start":
        data = {"choices": [{"delta": {}, "finish_reason": None}]}
    elif message_type == "thinking":
        data = {"choices": [{"delta": {"reasoning_content": content}, "finish_reason": None}]}
    elif message_type == "response":
        data = {"choices": [{"delta": {"content": content}, "finish_reason": None}]}
    elif message_type == "tool_call":
        data = {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [
                            {
                                "function": {
                                    "name": content["tool"],
                                    "arguments": json.dumps(content["params"]),
                                }
                            }
                        ]
                    },
                    "finish_reason": None,
                }
            ]
        }
    elif message_type == "end":
        data = {"choices": [{"delta": {}, "finish_reason": "stop"}]}
    elif message_type == "error":
        data = {"error": {"message": content, "type": "stream_error"}}
    else:
        return ""

    return f"data: {json.dumps(data)}\n\n"


async def process_message(
    message: dict,
    *,
    user_id: str,
    chat_id: str,
    stream_enabled: bool,
    thinking_enabled: bool,
) -> Optional[str]:
    """Convert a queued callback message into SSE output."""
    message_type = message["type"]

    if message_type == "thinking":
        if not thinking_enabled:
            return None
        if stream_enabled:
            delta, _ = stream_buffer.get_delta(user_id, chat_id, message["content"], "thinking")
            if delta:
                return format_sse_message("thinking", delta)
        else:
            return format_sse_message("thinking", message["content"])

    elif message_type == "response":
        if stream_enabled:
            delta, _ = stream_buffer.get_delta(user_id, chat_id, message["content"], "response")
            if delta:
                return format_sse_message("response", delta)
        else:
            return format_sse_message("response", message["content"])

    elif message_type == "tool_call":
        return format_sse_message("tool_call", message)

    return None


def should_complete_stream(state: StreamState, idle_time: float) -> bool:
    """
    Determine if the stream should be completed.
    Completion happens when:
    1. We have content AND last message wasn't a tool call
    2. OR we hit the safety timeout (120 seconds)
    """
    if state.has_content and not state.last_was_tool_call and idle_time > 1.0:
        return True
    if idle_time >= 120.0:
        logger.warning("Stream safety timeout reached")
        return True
    return False


# ----------------------------
# Lifespan
# ----------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    global agent_manager, default_runtime_config, api_callbacks

    logger.debug("Starting MCP Streaming API server...")

    default_runtime_config = resolve_default_runtime()
    provider_defaults = {
        default_runtime_config.provider: dict(default_runtime_config.provider_options)
    }
    agent_manager = AgentManager(
        default_provider=default_runtime_config.provider,
        default_model=default_runtime_config.agent_config.model,
        system_prompt_path=default_runtime_config.agent_config.system_prompt_path,
        provider_defaults=provider_defaults,
    )

    api_callbacks = APICallbacks(message_queues, stream_states, logger)
    agent_manager.on_agent_response = api_callbacks.on_agent_response
    agent_manager.on_agent_thinking = api_callbacks.on_agent_thinking
    agent_manager.on_agent_tool_call = api_callbacks.on_agent_tool_call
    agent_manager.on_agent_completion = api_callbacks.on_agent_completion

    logger.debug(
        "Defaults: provider=%s options=%s model=%s prompt=%s",
        default_runtime_config.provider,
        provider_defaults.get(default_runtime_config.provider),
        default_runtime_config.agent_config.model,
        default_runtime_config.agent_config.system_prompt_path,
    )

    yield

    logger.debug("Shutting down API server...")
    if agent_manager:
        await agent_manager.shutdown()


app = FastAPI(lifespan=lifespan, title="MCP Streaming API")


# ----------------------------
# Main Endpoints
# ----------------------------


@app.post("/stream")
async def stream_agent_response(request: StreamRequest):
    """Stream agent responses with thinking support."""
    global agent_manager, stream_buffer, message_queues

    if not agent_manager:
        raise HTTPException(status_code=500, detail="Agent manager not initialized")

    async def event_stream():
        queue_key = f"{request.user_id}:{request.chat_id}"

        # Clear buffers for this session at start
        stream_buffer.clear(request.user_id, request.chat_id)

        # Get queue for this session
        response_queue = message_queues[queue_key]

        # Setup stream state
        state = StreamState()
        stream_states[queue_key] = state

        try:
            # Send stream start
            yield format_sse_message("start", None)

            # Ensure session exists and resolve runtime capabilities
            runtime_config = await ensure_session_exists(request)
            thinking_enabled = (
                request.thinking_enabled
                if request.thinking_enabled is not None
                else runtime_config.agent_config.thinking_enabled
            )
            stream_enabled = (
                request.stream_enabled
                if request.stream_enabled is not None
                else runtime_config.agent_config.stream_enabled
            )

            # Send user message to agent
            await send_user_message(request)

            # Timing + stop signal used by both tasks
            loop = asyncio.get_event_loop()
            last_message_time = loop.time()
            stop = asyncio.Event()

            async def completion_watcher():
                """Periodically check whether we should complete the stream."""
                nonlocal last_message_time
                while not stop.is_set():
                    await asyncio.sleep(0.5)
                    idle_time = loop.time() - last_message_time
                    if should_complete_stream(state, idle_time):
                        logger.debug("Stream completed for %s:%s", request.user_id, request.chat_id)
                        stop.set()

            async def message_processor():
                """Yield SSE outputs as messages arrive."""
                nonlocal last_message_time
                while not stop.is_set():
                    get_task = asyncio.create_task(response_queue.get())
                    stop_task = asyncio.create_task(stop.wait())
                    done, pending = await asyncio.wait(
                        {get_task, stop_task},
                        return_when=asyncio.FIRST_COMPLETED,
                    )

                    for task in pending:
                        task.cancel()

                    if stop.is_set() or stop_task in done:
                        if not get_task.done():
                            get_task.cancel()
                        break

                    message = get_task.result()
                    last_message_time = loop.time()

                    sse_output = await process_message(
                        message,
                        user_id=request.user_id,
                        chat_id=request.chat_id,
                        stream_enabled=stream_enabled,
                        thinking_enabled=thinking_enabled,
                    )
                    if sse_output:
                        yield sse_output

            watcher = asyncio.create_task(completion_watcher())
            try:
                async for output in message_processor():
                    yield output
            finally:
                watcher.cancel()
                try:
                    await watcher
                except asyncio.CancelledError:
                    pass

            # Clear buffers after stream completes
            stream_buffer.clear(request.user_id, request.chat_id)

            # Send stream end
            yield format_sse_message("end", None)

        except Exception as exc:
            logger.error("Error in stream: %s", exc, exc_info=True)
            yield format_sse_message("error", str(exc))

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@app.delete("/session/{user_id}/{chat_id}")
async def terminate_session(user_id: str, chat_id: str):
    """Terminate a specific session."""
    global agent_manager, message_queues

    if not agent_manager:
        raise HTTPException(status_code=500, detail="Agent manager not initialized")

    try:
        await agent_manager.end_session(user_id, chat_id)
        queue_key = f"{user_id}:{chat_id}"
        message_queues.pop(queue_key, None)
        stream_states.pop(queue_key, None)
        return {"status": "success", "message": "Session terminated"}
    except Exception as exc:
        logger.error("Error terminating session: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if not default_runtime_config:
        raise HTTPException(status_code=500, detail="Defaults not initialized")

    return {
        "status": "healthy",
        "defaults": {
            "provider": default_runtime_config.provider,
            "provider_options": default_runtime_config.provider_options,
            "model": default_runtime_config.agent_config.model,
            "system_prompt_path": default_runtime_config.agent_config.system_prompt_path,
        },
    }


@app.get("/sessions")
async def list_sessions():
    """List all active sessions."""
    global agent_manager

    if not agent_manager:
        raise HTTPException(status_code=500, detail="Agent manager not initialized")

    sessions = []
    for key, session in agent_manager.sessions.items():
        sessions.append(
            {
                "session_id": session.session_id,
                "user_id": session.user_id,
                "chat_id": session.chat_id,
                "active": session.active,
            }
        )

    return {"sessions": sessions}


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    uvicorn.run(app, host="127.0.0.1", port=8080)
