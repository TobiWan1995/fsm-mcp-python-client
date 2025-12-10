from __future__ import annotations

from typing import TYPE_CHECKING, Any

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
        stripped = raw_input.strip()
        command = stripped.lower()
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
        if command == "/prompt":
            await self._run_prompt_interactive()
            return True
        if stripped.lower().startswith("/promptmock"):
            await self._run_prompt_mock(stripped)
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
        result = await session.mcp_client.list_prompts()
        if isinstance(result, str):
            print_formatted_text(
                FormattedText([("class:error", f"Error listing prompts: {result}\n")]),
                style=self.style,
            )
            return
        prompts = result.prompts or []
        print_formatted_text(
            FormattedText([("class:info", f"\nAvailable prompts ({len(prompts)}):\n")]),
            style=self.style,
        )
        for prompt in prompts:
            name = prompt.name or "Unknown"
            desc = prompt.description or ""
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

    async def _run_prompt_interactive(self) -> None:
        session = self.client.session
        agent_manager = getattr(self.client, "agent_manager", None)
        if not session or not agent_manager:
            return

        result = await session.mcp_client.list_prompts()
        if isinstance(result, str):
            print_formatted_text(
                FormattedText([("class:error", f"Error listing prompts: {result}\n")]),
                style=self.style,
            )
            return

        prompts = result.prompts or []
        if not prompts:
            print_formatted_text(
                FormattedText([("class:info", "No prompts available.\n")]),
                style=self.style,
            )
            return

        print_formatted_text(
            FormattedText([("class:info", f"\nAvailable prompts ({len(prompts)}):\n")]),
            style=self.style,
        )
        for idx, prompt in enumerate(prompts, start=1):
            name = prompt.name or "Unknown"
            desc = prompt.description or ""
            line = f"  {idx}) {name}: {desc}"
            print_formatted_text(FormattedText([("class:tool", line)]), style=self.style)
        print()

        from prompt_toolkit import prompt as pt_prompt

        choice = pt_prompt("Select prompt (number or name): ").strip()
        selected = None
        if choice.isdigit():
            pos = int(choice) - 1
            if 0 <= pos < len(prompts):
                selected = prompts[pos]
        else:
            for prompt in prompts:
                if prompt.name == choice:
                    selected = prompt
                    break

        if not selected:
            print_formatted_text(
                FormattedText([("class:error", "Invalid selection.\n")]),
                style=self.style,
            )
            return

        args: dict[str, Any] = {}
        for argument in selected.arguments or []:
            arg_name = argument.name
            if not arg_name:
                continue
            desc = argument.description or ""
            required = bool(argument.required)

            label = arg_name
            if desc:
                label += f" ({desc})"
            if required:
                label += " [required]"
            label += ": "

            while True:
                value = pt_prompt(label).strip()
                if not value and required:
                    print_formatted_text(
                        FormattedText([("class:error", "This field is required.\n")]),
                        style=self.style,
                    )
                    continue
                if value:
                    args[arg_name] = value
                break

        prompt_name = selected.name
        if not prompt_name:
            print_formatted_text(
                FormattedText([("class:error", "Selected prompt is missing a name.\n")]),
                style=self.style,
            )
            return

        json_rpc = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "prompts/get",
            "params": {
                "name": prompt_name,
                "arguments": args,
            },
        }

        await agent_manager.run_prompt(
            user_id="cli_user",
            chat_id="cli_chat",
            json_rpc=json_rpc,
            enqueue=True,
        )

        print_formatted_text(
            FormattedText([("class:info", "Prompt enqueued.\n")]),
            style=self.style,
        )

    async def _run_prompt_mock(self, raw_command: str) -> None:
        parts = raw_command.split(maxsplit=1)
        if len(parts) != 2:
            print_formatted_text(
                FormattedText([("class:error", "Usage: /promptmock <key>\n")]),
                style=self.style,
            )
            return

        key = parts[1].strip()
        if not key:
            print_formatted_text(
                FormattedText([("class:error", "Usage: /promptmock <key>\n")]),
                style=self.style,
            )
            return

        try:
            from mock_prompts import MOCK_PROMPT_CALLS
        except ImportError:
            print_formatted_text(
                FormattedText([("class:error", "mock_prompts.py not found.\n")]),
                style=self.style,
            )
            return

        json_rpc = MOCK_PROMPT_CALLS.get(key)
        if not json_rpc:
            print_formatted_text(
                FormattedText([("class:error", f"No mock prompt '{key}'\n")]),
                style=self.style,
            )
            return

        session = self.client.session
        agent_manager = getattr(self.client, "agent_manager", None)
        if not session or not agent_manager:
            return

        await agent_manager.run_prompt(
            user_id="cli_user",
            chat_id="cli_chat",
            json_rpc=json_rpc,
            enqueue=True,
        )

        print_formatted_text(
            FormattedText([("class:info", "Mock prompt enqueued.\n")]),
            style=self.style,
        )
