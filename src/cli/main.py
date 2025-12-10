import asyncio, logging, sys

from typing import Any, Dict, Optional
from prompt_toolkit import PromptSession, print_formatted_text
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.styles import Style
from src.cli.config import CLIConfig
from src.cli.callbacks import CLICallbacks
from src.cli.commands import CLICommands

from src.agent.manager import AgentManager
from src.mcp.client import MCPClientConfig
from src.agent.base import AgentConfig
from src.util.stream_buffer import StreamBuffer 
from src.cli.writer import ConversationWriter

# Silence httpx + its transport library
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# (optional) stop them bubbling up to the root logger
logging.getLogger("httpx").propagate = False
logging.getLogger("httpcore").propagate = False

logger = logging.getLogger(__name__)

# Define style for the CLI
style = Style.from_dict({
    'user': '#00aa00 bold',
    'assistant': '#0088ff',
    'thinking': '#888888 italic',
    'tool': '#ffaa00',
    'tool_response': '#00aa00',
    'error': '#ff0000 bold',
    'info': '#aaaaaa',
})


class CLIClient:
    """Command-line interface for testing the agent system"""
    
    def __init__(
        self,
        mcp_config: MCPClientConfig,
        agent_config: AgentConfig,
        provider: str = "ollama",
        provider_options: Optional[Dict[str, Any]] = None,
    ):
        self.mcp_config = mcp_config
        self.agent_config = agent_config
        self.provider = provider
        self.provider_options: Dict[str, Any] = dict(provider_options or {})
        self.agent_manager: Optional[AgentManager] = None
        self.session = None
        self.prompt_session = PromptSession()
        self.running = True
        
        # Initialize StreamBuffer for streaming mode (no style needed - we handle printing)
        self.stream_buffer = StreamBuffer() if agent_config.stream_enabled else None
        
        # Track if we're currently streaming
        self.is_streaming = False
        # Track if we're in thinking mode to close bracket later
        self.is_thinking = False
        self.callbacks = CLICallbacks(self, style)
        self.commands = CLICommands(self, style)
    
    async def initialize(self):
        """Initialize the agent manager and session"""
        # Create agent manager
        provider_defaults = {self.provider: dict(self.provider_options)}
        self.agent_manager = AgentManager(
            default_provider=self.provider,
            default_model=self.agent_config.model,
            system_prompt_path=self.agent_config.system_prompt_path,
            provider_defaults=provider_defaults,
        )
        # Set up callbacks
        self.callbacks.writer = ConversationWriter()
        self.agent_manager.on_agent_response = self.callbacks.on_agent_response
        self.agent_manager.on_agent_thinking = self.callbacks.on_agent_thinking
        self.agent_manager.on_agent_tool_call = self.callbacks.on_agent_tool_call
        self.agent_manager.on_tool_response = self.callbacks.on_tool_response
        self.agent_manager.on_agent_completion = self.callbacks.on_agent_completion
        
        # Create session
        self.session = await self.agent_manager.create_session(
            user_id="cli_user",
            chat_id="cli_chat",
            mcp_config=self.mcp_config,
            agent_config=self.agent_config,
            provider=self.provider,
            provider_options=self.provider_options,
        )
        
        print_formatted_text(
            FormattedText([('class:info', f"Session started: {self.session.session_id}\n")]),
            style=style
        )
    
    
    
    def finalize_streaming(self):
        """Helper to finalize streaming output"""
        if self.is_streaming:
            # Close thinking bracket if needed
            if self.is_thinking:
                print_formatted_text(
                    FormattedText([('class:thinking', ']')]),
                    style=style
                )
                self.is_thinking = False
            
            # Add final newline
            print()
            sys.stdout.flush()
            self.is_streaming = False

    def flush_stdout(self) -> None:
        """Flush stdout for streaming callbacks."""
        sys.stdout.flush()
    
    async def user_input_loop(self):
        """Handle user input"""
        with patch_stdout():
            while self.running:
                try:
                    # Finalize any streaming output before prompting
                    if self.is_streaming:
                        self.finalize_streaming()
                    
                    # Get user input
                    user_input = await asyncio.to_thread(
                        self.prompt_session.prompt,
                        FormattedText([('class:user', '[You]: ')]),
                        style=style
                    )
                    
                    stripped_input = user_input.strip()

                    # Handle special commands
                    lowered = stripped_input.lower()
                    if lowered in ['exit', 'quit', '/quit']:
                        self.running = False
                        break
                    handled = await self.commands.handle(stripped_input)
                    if handled:
                        continue
                    
                    # Clear buffers for new message if streaming
                    if self.stream_buffer:
                        self.stream_buffer.clear("cli_user", "cli_chat")
                    
                    # Send message to agent
                    if stripped_input:
                        if self.callbacks.writer:
                            self.callbacks.writer.record_user(stripped_input)
                        await self.agent_manager.send_message(
                            "cli_user",
                            "cli_chat",
                            stripped_input
                        )
                
                except KeyboardInterrupt:
                    self.running = False
                    break
                except Exception as e:
                    print_formatted_text(
                        FormattedText([('class:error', f"\nError: {e}\n")]),
                        style=style
                    )
    
    async def run(self):
        """Run the CLI client"""
        try:
            await self.initialize()
            await self.user_input_loop()
        finally:
            # Finalize any ongoing streaming
            if self.is_streaming:
                self.finalize_streaming()
            
            if self.agent_manager:
                await self.agent_manager.shutdown()
            
            print_formatted_text(
                FormattedText([('class:info', "\nGoodbye!\n")]),
                style=style
            )


async def main():
    """Main entry point"""
    args = CLIConfig.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    resolved = CLIConfig.resolve(args)

    cli = CLIClient(
        mcp_config=resolved.mcp_config,
        agent_config=resolved.agent_config,
        provider=resolved.provider,
        provider_options=resolved.provider_options,
    )
    
    await cli.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(0)
