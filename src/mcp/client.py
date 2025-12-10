import asyncio
import logging
from typing import Any, Callable, Optional, Awaitable

import mcp.types as types
from pydantic import BaseModel

from mcp.client.session import ClientSession
from mcp.client.sse import sse_client
from mcp.shared.context import RequestContext
from mcp.shared.session import RequestResponder

from src.mcp.sampling import SessionAwareSamplingHandler

logger = logging.getLogger(__name__)

JsonRpcMethod = Callable[[dict[str, Any]], Awaitable[types.ServerResult | str]]

class MCPClientConfig(BaseModel):
    """Configuration for MCP client initialisation."""

    name: str
    transport: str = "sse"
    command: Optional[str] = None
    args: Optional[list[str]] = None
    env: Optional[dict[str, str]] = None
    cwd: Optional[str] = None
    url: Optional[str] = None
    auth_token: Optional[str] = None
    timeout: float = 5.0
    sse_read_timeout: float = 60.0 * 5


class MCPClient:
    """Thin wrapper around the MCP client session."""

    def __init__(
        self,
        session_key: str,
        config: MCPClientConfig,
        sampling_handler: SessionAwareSamplingHandler
    ) -> None:
        self.session_key = session_key
        self.name = f"mcp_{session_key}"
        self.config = config
        self.sampling_handler = sampling_handler

        self.closed = asyncio.Event()
        self.done = asyncio.Event()
        self._connected = asyncio.Event()
        self._task: Optional[asyncio.Task] = None

        self.session: Optional[ClientSession] = None
        self.on_capabilities_changed = None # set from outside

        self._tools_cache: list[types.Tool] = []
        self._prompts_cache: list[types.Prompt] = []
        self._resources_cache: list[types.Resource] = []

        # Update changes if needed (True for initial retrieval)
        self._tool_list_changed = True
        self._prompt_list_changed = True
        self._resource_list_changed = True

    # ----------------------------
    # Lifecycle
    # ----------------------------

    async def initialize(self) -> bool:
        transport = (self.config.transport or "sse").lower()
        if transport == "stdio":
            message = (
                "STDIO transport is not implemented yet. "
                "Please use the SSE transport until support is available."
            )
            logger.error(message)
            raise NotImplementedError(message)

        if transport not in ("sse", "streamable_http"):
            logger.error("Unknown MCP transport: %s", self.config.transport)
            return False

        self._task = asyncio.create_task(self._run_client_loop())
        try:
            await asyncio.wait_for(self._connected.wait(), timeout=10)
        except asyncio.TimeoutError:
            logger.error("Timeout while initialising MCP client %s", self.name)
            return False

        return True

    async def teardown(self) -> None:
        self.done.set()
        if self._task:
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self.session = None
        self.closed.set()

    # ----------------------------
    # Global SessionAware Handler
    # ----------------------------

    async def _sampling_callback(self, ctx: RequestContext["ClientSession", Any], params: types.CreateMessageRequestParams):
        return await self.sampling_handler.sample(self.session_key, ctx, params)
    
    # ----------------------------
    # Local Handlers
    # ----------------------------

    async def _message_handler(
        self,
        message: RequestResponder[types.ServerRequest, types.ClientResult]
                | types.ServerNotification
                | Exception,
    ) -> None:
        if isinstance(message.root, types.ToolListChangedNotification):
            logger.info("Notification received: ToolListChanged")
            self._tool_list_changed = True

        elif isinstance(message.root, types.ResourceListChangedNotification):
            logger.info("Notification received: ResourceListChanged")
            self._resource_list_changed = True

        elif isinstance(message.root, types.PromptListChangedNotification):
            logger.info("Notification received: PromptListChanged")
            self._prompt_list_changed = True

    # ----------------------------
    # Main loop
    # ----------------------------

    async def _run_client_loop(self) -> None:
        try:
            transport = (self.config.transport or "sse").lower()
            if transport == "sse":
                async with sse_client(
                    url=self.config.url,
                    timeout=self.config.timeout,
                    sse_read_timeout=self.config.sse_read_timeout,
                ) as (read_stream, write_stream):
                    async with ClientSession(
                        read_stream,
                        write_stream,
                        sampling_callback=self._sampling_callback,
                        message_handler=self._message_handler
                    ) as session:
                        self.session = session
                        await session.initialize()
                        self._connected.set()
                        await self._refresh_capabilities()
                        await self.done.wait()
            else:
                raise ValueError(f"Unsupported transport {self.config.transport}")
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("MCP client loop failed for %s: %s", self.name, exc)
        finally:
            self.closed.set()

    # ----------------------------
    # Capabilities
    # ----------------------------

    async def _refresh_capabilities(self) -> None:
        if not self.session:
            return

        try:
            if self._tool_list_changed: 
                tools_result: types.ListToolsResult = await self.session.list_tools()
                self._tools_cache = tools_result.tools
                self._tool_list_changed = False

            if self._resource_list_changed:
                resources_result: types.ListResourcesResult = await self.session.list_resources()
                self._resources_cache = resources_result.resources
                self._resource_list_changed = False

            if self._prompt_list_changed:
                prompts_result: types.ListPromptsResult = await self.session.list_prompts()
                self._prompts_cache = prompts_result.prompts
                self._prompt_list_changed = False

            if self.on_capabilities_changed:
                await self.on_capabilities_changed(
                    self._tools_cache,
                    self._prompts_cache,
                    self._resources_cache,
                )
        except Exception as exc:
            logger.error("Error while refreshing capabilities: %s", exc, exc_info=True)

    # ----------------------------
    # JSON-RPC routing
    # ----------------------------

    async def execute_json_rpc(self, json_rpc: dict[str, Any]) -> types.ServerResult | str:
        if not self.session:
            return f"Client {self.name} not initialized"

        method_name = json_rpc.get("method")
        if not method_name:
            return "Missing 'method' in JSON-RPC request"

        params: dict[str, Any] = json_rpc.get("params") or {}

        methods: dict[str, JsonRpcMethod] = {
            "tools/call":      self.call_tool,
            "prompts/get":     self.get_prompt,
            "resources/read":  self.read_resource,
            "tools/list":      lambda _params: self.list_tools(),
            "prompts/list":    lambda _params: self.list_prompts(),
            "resources/list":  lambda _params: self.list_resources(),
        }

        method = methods.get(method_name)
        if method is None:
            return f"Unknown MCP method: {method_name}"

        try:
            result = await method(params)
            await self._refresh_capabilities()
            return result
        except Exception as exc:
            logger.exception("JSON-RPC error")
            return f"JSON-RPC error: {exc}"


    # ----------------------------
    # MCP operations
    # ----------------------------

    async def call_tool(self, params: dict[str, Any]) -> types.CallToolResult | str:
        if not self.session:
            raise RuntimeError(f"Client {self.name} not initialized")

        name = params.get("name")
        if not name:
            raise ValueError("Missing parameter 'name' for tools/call")

        arguments = params.get("arguments") or {}

        try:
            result: types.CallToolResult = await self.session.call_tool(name, arguments)
            return result
        except Exception as exc:
            logger.error("Error during tool call %s: %s", name, exc)
            return f"Tool error {name}: {exc}"


    async def get_prompt(self, params: dict[str, Any]) -> types.GetPromptResult | str:
        if not self.session:
            raise RuntimeError(f"Client {self.name} not initialized")

        name = params.get("name")
        if not name:
            raise ValueError("Missing parameter 'name' for prompts/get")

        arguments = params.get("arguments") or {}

        try:
            result: types.GetPromptResult = await self.session.get_prompt(name, arguments)
            return result
        except Exception as exc:
            logger.error("Error in prompts/get %s: %s", name, exc)
            return f"Prompt error {name}: {exc}"


    async def read_resource(self, params: dict[str, Any]) -> types.ReadResourceResult | str:
        if not self.session:
            raise RuntimeError(f"Client {self.name} not initialized")

        uri = params.get("uri")
        if not uri:
            raise ValueError("Missing parameter 'uri' for resources/read")

        try:
            result: types.ReadResourceResult = await self.session.read_resource(uri)
            return result
        except Exception as exc:
            logger.error("Error in resources/read %s: %s", uri, exc)
            return f"Read error {uri}: {exc}"
    

    async def list_tools(self) -> types.ListToolsResult | str:
        if not self.session:
            raise RuntimeError(f"Client {self.name} not initialized")
        try:
            result: types.ListToolsResult = await self.session.list_tools()
            self._tools_cache = result.tools
            return result
        except Exception as exc:
            logger.error("Error in list_tools: %s", exc)
            return f"Error in list_tools: {exc}"


    async def list_prompts(self) -> types.ListPromptsResult | str:
        if not self.session:
            raise RuntimeError(f"Client {self.name} not initialized")
        try:
            result: types.ListPromptsResult = await self.session.list_prompts()
            self._prompts_cache = result.prompts
            return result
        except Exception as exc:
            logger.error("Error in list_prompts: %s", exc)
            return f"Error in list_prompts: {exc}"


    async def list_resources(self) -> types.ListResourcesResult | str:
        if not self.session:
            raise RuntimeError(f"Client {self.name} not initialized")
        try:
            result: types.ListResourcesResult = await self.session.list_resources()
            self._resources_cache = result.resources
            return result
        except Exception as exc:
            logger.error("Error in list_resources: %s", exc)
            return f"Error in list_resources: {exc}"

