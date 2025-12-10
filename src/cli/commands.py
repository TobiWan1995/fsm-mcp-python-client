from __future__ import annotations

from typing import TYPE_CHECKING

from prompt_toolkit import print_formatted_text
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.styles import Style

if TYPE_CHECKING:
    from src.cli.main import CLIClient


class CLICommands:
    """Command helpers for the CLI."""

    def __init__(self, client: "CLIClient", style: Style) -> None:
        self.client = client
        self.style = style

    async def handle(self, raw_input: str) -> bool:
        command = raw_input.strip().lower()
        if command == "/help":
            self._print_help()
            return True
        if command == "/clear":
            await self._clear_conversation()
            return True
        if command == "/tools":
            await self._list_tools()
            return True
        if command == "/prompts":
            await self._list_prompts()
            return True
        if command == "/resources":
            await self._list_resources()
            return True
        return False

    def _print_help(self) -> None:
        help_text = """
            Available commands:
            /help      - Show this help message
            /clear     - Clear conversation history
            /tools     - List available tools
            /prompts   - List available prompts
            /resources - List available resources
            /quit      - Exit the CLI
        """
        print_formatted_text(FormattedText([("class:info", help_text)]), style=self.style)

    async def _clear_conversation(self) -> None:
        session = self.client.session
        if session:
            session.agent.reset()
        if self.client.stream_buffer:
            self.client.stream_buffer.clear("cli_user", "cli_chat")
        print_formatted_text(
            FormattedText([("class:info", "Conversation cleared.\n")]),
            style=self.style,
        )

    async def _list_tools(self) -> None:
        session = self.client.session
        if not session:
            return
        tools = await session.mcp_client.list_tools()
        print_formatted_text(
            FormattedText([("class:info", f"\nAvailable tools ({len(tools)}):\n")]),
            style=self.style,
        )
        for tool in tools:
            name = tool.get("name", tool.get("function", {}).get("name", "Unknown"))
            desc = tool.get("description", tool.get("function", {}).get("description", ""))
            print_formatted_text(
                FormattedText([("class:tool", f"  - {name}: {desc}")]),
                style=self.style,
            )
        print()

    async def _list_prompts(self) -> None:
        session = self.client.session
        if not session:
            return
        prompts = await session.mcp_client.list_prompts()
        print_formatted_text(
            FormattedText([("class:info", f"\nAvailable prompts ({len(prompts)}):\n")]),
            style=self.style,
        )
        for prompt in prompts:
            name = prompt.get("name", "Unknown")
            desc = prompt.get("description", "")
            print_formatted_text(
                FormattedText([("class:tool", f"  - {name}: {desc}")]),
                style=self.style,
            )
        print()

    async def _list_resources(self) -> None:
        session = self.client.session
        if not session:
            return
        resources = await session.mcp_client.list_resources()
        print_formatted_text(
            FormattedText([("class:info", f"\nAvailable resources ({len(resources)}):\n")]),
            style=self.style,
        )
        for resource in resources:
            uri = resource.get("uri", "Unknown")
            name = resource.get("name", "")
            print_formatted_text(
                FormattedText([("class:tool", f"  - {uri}: {name}")]),
                style=self.style,
            )
        print()
