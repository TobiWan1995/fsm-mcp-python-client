from __future__ import annotations

import difflib
import json
from typing import Any, Dict, Iterable, Tuple, List

from mcp.types import Prompt as MCPPrompt
from mcp.types import Resource as MCPResource
from mcp.types import Tool as MCPTool

from src.adapters.call_translator import MCPCallTranslator


class OllamaCallTranslator(MCPCallTranslator):
    """
    Translates Ollama tool_call payloads into MCP JSON-RPC requests.
    Uses canonical identifiers: tool name and resource URI.
    """

    def __init__(
        self,
    ) -> None:
        self._tools_by_name: Dict[str, MCPTool] = {}
        self._prompts_by_name: Dict[str, MCPPrompt] = {}
        self._resources_by_uri: Dict[str, MCPResource] = {}
        self._ollama_name_index: Dict[str, Tuple[str, str]] = {}

    # ----------------------------
    # Cache management
    # ----------------------------

    def update_capabilities(
        self,
        tools: Iterable[MCPTool] | None,
        prompts: Iterable[MCPPrompt] | None,
        resources: Iterable[MCPResource] | None,
    ) -> None:
        super().update_capabilities(tools, prompts, resources)
        self._tools_by_name.clear()
        self._prompts_by_name.clear()
        self._resources_by_uri.clear()
        self._ollama_name_index.clear()

        for tool in tools or []:
            name = tool.name
            if not name:
                continue
            self._tools_by_name[name] = tool
            self._ollama_name_index[name] = ("tool", name)

        for prompt in prompts or []:
            name = prompt.name
            if not name:
                continue
            self._prompts_by_name[name] = prompt

        for resource in resources or []:
            uri = str(resource.uri or "")
            if not uri:
                continue
            self._resources_by_uri[uri] = resource
            self._ollama_name_index[uri] = ("resource", uri)

    # ----------------------------
    # Translation
    # ----------------------------

    def extract_tool_calls(self, payload: Any) -> List[Dict[str, Any]]:
        """
        Accept multiple shapes produced by Ollama:
        - single tool_call dict
        - list of tool_call dicts
        - response message containing tool_calls (streaming or final)
        """
        if payload is None:
            return []

        raw_calls: List[Any] = []

        if isinstance(payload, dict):
            if payload.get("type") == "function" and "function" in payload:
                raw_calls = [payload]
            elif "function" in payload:
                raw_calls = [payload]
            else:
                candidate = payload.get("message") if "message" in payload else payload
                if isinstance(candidate, dict):
                    tc = candidate.get("tool_calls")
                    if isinstance(tc, list):
                        raw_calls = tc
                tc = payload.get("tool_calls")
                if isinstance(tc, list):
                    raw_calls = tc

        elif hasattr(payload, "tool_calls"):
            raw_calls = list(getattr(payload, "tool_calls") or [])

        elif isinstance(payload, list):
            raw_calls = list(payload)

        normalized: List[Dict[str, Any]] = []
        for call in raw_calls:
            if hasattr(call, "model_dump"):
                call_dict = call.model_dump()
            elif isinstance(call, dict):
                call_dict = call
            else:
                raise ValueError(f"Unsupported tool_call entry: {call!r}")
            normalized.append(call_dict)

        return normalized

    def to_json_rpc(self, tool_call: Dict[str, Any], *, rpc_id: Any = 1) -> Dict[str, Any]:
        fn_payload = (tool_call or {}).get("function", {}) or {}
        name = fn_payload.get("name")
        if not name:
            raise ValueError("Invalid tool_call payload: missing function.name")

        arguments = self._ensure_arguments_dict(fn_payload.get("arguments"))

        hit = self._ollama_name_index.get(name)
        if hit:
            kind, key = hit
            return self._make_rpc(kind, key, arguments, rpc_id)

        if name in self._tools_by_name:
            return self._make_rpc("tool", name, arguments, rpc_id)
        if name in self._resources_by_uri:
            return self._make_rpc("resource", name, arguments, rpc_id)

        if isinstance(arguments, dict) and "uri" in arguments:
            uri = str(arguments["uri"])
            if uri in self._resources_by_uri:
                return self._make_rpc("resource", uri, arguments, rpc_id)

        raise self._no_match_error(name)

    # ----------------------------
    # Helpers
    # ----------------------------

    def _make_rpc(self, kind: str, key: str, arguments: Dict[str, Any], rpc_id: Any) -> Dict[str, Any]:
        if kind == "tool":
            return {"jsonrpc": "2.0", "id": rpc_id, "method": "tools/call", "params": {"name": key, "arguments": arguments}}
        if kind == "resource":
            return {"jsonrpc": "2.0", "id": rpc_id, "method": "resources/read", "params": {"uri": key}}
        raise ValueError(f"Unknown capability kind: {kind}")

    @staticmethod
    def _ensure_arguments_dict(arguments: Any) -> Dict[str, Any]:
        if isinstance(arguments, dict):
            return arguments
        if arguments is None or arguments == "":
            return {}
        if isinstance(arguments, (list, int, float, bool)):
            return {"_": arguments}
        if isinstance(arguments, str):
            try:
                parsed = json.loads(arguments)
                return parsed if isinstance(parsed, dict) else {"_": parsed}
            except Exception:
                return {"_raw": arguments}
        try:
            return dict(arguments)  # type: ignore[arg-type]
        except Exception:
            return {"_raw": str(arguments)}

    def _no_match_error(self, name: str) -> ValueError:
        candidates = list(self._ollama_name_index.keys()) + list(self._tools_by_name.keys()) + list(self._resources_by_uri.keys())
        suggestions = difflib.get_close_matches(name, candidates, n=3, cutoff=0.6)
        hint = f" (did you mean: {', '.join(suggestions)})" if suggestions else ""
        return ValueError(f"Ollama tool_call '{name}' could not be mapped to an MCP capability{hint}.")
