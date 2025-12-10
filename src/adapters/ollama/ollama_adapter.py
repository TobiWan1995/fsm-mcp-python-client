from __future__ import annotations

from src.adapters.adapter import MCPAdapter
from src.agent.base import AgentConfig
from src.adapters.ollama.ollama_call_translator import OllamaCallTranslator
from src.adapters.ollama.ollama_content_mapper import OllamaContentMapper
from src.adapters.ollama.ollama_tool_mapper import OllamaToolMapper


class OllamaAdapter(MCPAdapter):
    """
    Concrete MCP adapter for the Ollama runtime. Bridges between MCP capabilities
    and Ollama specific payloads (function tools, content blocks, etc.).
    """

    def __init__(self, agent_config: AgentConfig) -> None:
        super().__init__(
            content_mapper=OllamaContentMapper(agent_config),
            call_translator=OllamaCallTranslator(),
            tool_mapper=OllamaToolMapper(),
        )
