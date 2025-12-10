from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Dict, List, Optional, Sequence, Tuple


@dataclass
class AgentConfig:
    """
    Shared agent configuration used across providers.
    - model: target model identifier
    - thinking_enabled: whether reasoning/thinking output should be requested
    - stream_enabled: whether streaming responses are enabled
    - system_prompt_path: optional path to a system prompt template
    - supports_vision: capability flag for image handling
    - options: provider specific options object (e.g. Ollama Options)
    """
    model: str = "llama3.2:3b"
    thinking_enabled: bool = False
    stream_enabled: bool = False
    system_prompt_path: Optional[str] = "src/prompts/system.md"
    supports_vision: bool = False
    options: Any = None


class BaseAgent(ABC):
    """
    Abstract base class for chat agents.
    Keeps provider specific history entries, manages the system prompt
    template, and stores negotiated tool specifications.
    """

    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self.history: List[Any] = []
        self.system_prompt_template: str = ""
        self._active_tools: List[Dict[str, Any]] = []

    # ----------------------------
    # History management
    # ----------------------------

    def reset(self) -> None:
        """Reset the conversation but keep the system message intact."""
        self.history = [msg for msg in self.history if self._is_system_message(msg)]

    def add_message(self, message: Any) -> None:
        """Append a provider-specific message object to the history."""
        self.history.append(message)

    def set_active_tools(self, tools: Sequence[Dict[str, Any]] | None) -> None:
        """Store provider specific tool specifications for use on the next call."""
        self._active_tools = list(tools or [])

    @property
    def active_tools(self) -> List[Dict[str, Any]]:
        """Return a shallow copy of the currently registered tool specs."""
        return list(self._active_tools)

    # ----------------------------
    # System prompt management
    # ----------------------------

    def set_system_prompt(self, template: str) -> None:
        """
        Persist the base template and upsert the system message.
        If the template is empty, leave the history unchanged.
        """
        self.system_prompt_template = template or ""

        if not self.system_prompt_template:
            return

        self._upsert_system_message(self.system_prompt_template)

    def _upsert_system_message(self, content: str) -> None:
        """Create or replace the provider-specific system message."""
        system_message = self.make_system_message(content)
        if self.history and self._is_system_message(self.history[0]):
            self.history[0] = system_message
        else:
            self.history.insert(0, system_message)

    # ----------------------------
    # Provider specific factories
    # ----------------------------

    @abstractmethod
    def make_user_message(self, content: str, *, images: Optional[List[str]] = None) -> Any:
        """Create a provider specific user message."""

    @abstractmethod
    def make_system_message(self, content: str) -> Any:
        """Create a provider specific system message."""

    @abstractmethod
    def make_tool_message(
        self,
        content: str,
        *,
        name: Optional[str] = None,
        images: Optional[List[str]] = None,
    ) -> Any:
        """Create a provider specific tool message."""

    @abstractmethod
    def make_assistant_message(
        self,
        content: str,
        *,
        thinking: Optional[str] = None,
        tool_calls: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> Any:
        """Create a provider specific assistant message."""

    @abstractmethod
    def _is_system_message(self, message: Any) -> bool:
        """Return True if the given history entry represents a system message."""

    # ----------------------------
    # Response generation
    # ----------------------------

    @abstractmethod
    async def generate_response(
        self,
        new_messages: Sequence[Any],
    ) -> AsyncGenerator[Tuple[Optional[str], str, Optional[List[Dict[str, Any]]]], None]:
        """
        Produce a streaming response after ingesting the provided provider-specific
        messages. `new_messages` should already be in the provider's native format
        (e.g. OllamaMessage instances) and will typically be appended to the history
        before generating the response.
        """

    @abstractmethod
    async def generate_response_sync(
        self,
        new_messages: Sequence[Any],
    ) -> Tuple[Optional[str], str, List[Dict[str, Any]]]:
        """
        Produce a non-streaming response after ingesting the provided provider-specific
        messages.
        Returns (thinking, full content, tool payloads).
        """
