from __future__ import annotations

from typing import Any, Dict, List, Mapping, Tuple

from mcp.types import Prompt as MCPPrompt

from src.adapters.tool_mapper import CapabilityChange, MCPToolMapper


class OllamaToolMapper(MCPToolMapper):
    """
    Builds Ollama function tool definitions from MCP capabilities.
    - Tools keep their original name.
    - Prompts use their canonical name.
    - Resources use their URI as the function name.

    Provides:
      - `get_ollama_tools()` -> list usable as Ollama `tools=[...]`.
      - `get_reverse_index()` -> {ollama_name: (kind, key)} for reverse lookups.
    """

    def __init__(self) -> None:
        super().__init__()
        self._reverse_index: Dict[str, Tuple[str, str]] = {}

    def get_ollama_tools(self) -> List[Dict[str, Any]]:
        """Backward compatibility wrapper."""
        return self.get_provider_tools()

    def get_reverse_index(self) -> Dict[str, Tuple[str, str]]:
        return dict(self._reverse_index)

    # ----------------------------
    # Builders
    # ----------------------------

    def _build_provider_tools(self) -> List[Dict[str, Any]]:
        tools_out: List[Dict[str, Any]] = []
        reverse: Dict[str, Tuple[str, str]] = {}

        for name, tool in self._tools_by_name.items():
            spec = {
                "type": "function",
                "function": {
                    "name": name,
                    "description": tool.description,
                    "parameters": self._normalize_root_schema(tool.inputSchema or {}),
                },
            }
            tools_out.append(spec)
            reverse[name] = ("tool", name)

        for uri, resource in self._resources_by_uri.items():
            title = resource.title or ""
            description = resource.description or ""
            description_parts = [part for part in [title or str(uri), description] if part]
            merged_description = " - ".join(description_parts) if description_parts else str(uri)

            spec = {
                "type": "function",
                "function": {
                    "name": str(uri),
                    "description": merged_description,
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": [],
                        "additionalProperties": False,
                    },
                },
            }
            tools_out.append(spec)
            reverse[str(uri)] = ("resource", uri)

        self._reverse_index = reverse
        return tools_out

    def _format_capability_update(
        self,
        tool_change: CapabilityChange,
        prompt_change: CapabilityChange,
        resource_change: CapabilityChange,
    ) -> str | None:
        combined_added = tool_change.added + resource_change.added
        combined_removed = tool_change.removed + resource_change.removed

        if not combined_added and not combined_removed:
            return None

        lines: list[str] = ["The list of available tools has been updated.", ""]
        current_tools = (
            list(self._tools_by_name.values())
            + list(self._resources_by_uri.values())
        )

        lines.append("The following Tools are available:")
        if current_tools:
            for idx, tool in enumerate(current_tools, start=1):
                name = getattr(tool, "name", None) or getattr(tool, "uri", None) or "<unknown>"
                desc = getattr(tool, "description", None) or ""
                lines.append(f"{idx}. {name}: {desc}")
        else:
            lines.append("None")

        if combined_removed:
            lines.append("")
            lines.append("The following tools have been removed:")
            for idx, item in enumerate(combined_removed, start=1):
                name = getattr(item, "name", None) or getattr(item, "uri", None) or "<unknown>"
                desc = getattr(item, "description", None) or ""
                lines.append(f"{idx}. {name}: {desc}")

        return "\n".join(lines)

    # ----------------------------
    # Helpers
    # ----------------------------

    def _prompt_to_schema(self, prompt: MCPPrompt) -> Dict[str, Any]:
        args = prompt.arguments or []
        properties: Dict[str, Any] = {}
        required: List[str] = []
        for arg in args:
            name = arg.name
            if not name:
                continue
            desc = arg.description or ""
            is_required = arg.required
            prop: Dict[str, Any] = {"type": "string"}
            if desc:
                prop["description"] = desc
            properties[name] = prop
            if is_required:
                required.append(name)
        return {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        }

    def _normalize_root_schema(self, schema: Mapping[str, Any] | None) -> Dict[str, Any]:
        sch = dict(schema or {})
        sch.pop("$schema", None)
        if sch.get("type") != "object":
            sch = {
                "type": "object",
                "properties": {"payload": schema or {}},
                "required": ["payload"],
                "additionalProperties": False,
            }
        return sch
