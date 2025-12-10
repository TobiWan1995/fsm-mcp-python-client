# FSM MCP Client

> **Note:** This repository contains the reference client implementation (Agent Layer) for the **FSM MCP Python SDK**. It is designed as a server-centric, adapter-based architecture to connect Large Language Models (LLMs) with state-aware MCP servers.

## About this Project

This project implements the **Agent Layer** described in the associated Master's thesis (December 2025). It addresses two specific challenges in the MCP ecosystem:

1.  **Provider Independence:** It utilizes an adapter architecture to support local LLM providers. Currently, **Ollama** is the supported provider for function calling.
2.  **Server-Centric Logic:** It ensures that all business logic and process constraints remain on the MCP server. The client acts solely as a manager for the LLM, the session, and user interaction, without replicating the server's state machine.

## Architecture

The architecture relies on the **Adapter Pattern** to decouple the specific requirements of an LLM provider from the standardized MCP protocol.

### 1\. The Adapter Layer

The core of this repository is the `MCPAdapter` class. It acts as the bridge between the proprietary API of the LLM (wrapped in a `BaseAgent`) and the standard `MCPClient`. It encapsulates three specialized mappers:

  * **CallTranslator:** Extracts tool calls from the LLM's raw payload and translates them into MCP-compliant JSON-RPC requests.
  * **ToolMapper:** Converts standard MCP tool signatures into the specific schema required by the LLM provider.
  * **ContentMapper:** Transforms MCP `ServerResult` objects back into provider-specific message formats for the chat history.

This design ensures that the `MCPClient` remains generic and unaware of the specific LLM API being used.

### 2\. Turn-Based Message Processing

To handle the asynchronous nature of tool execution and state changes, the client uses a strictly **turn-based message model**.

Each `AgentSession` maintains an asynchronous message queue. The processing logic distinguishes between two types of turns:

  * **User Turn:** Triggered by a text message from the user.
  * **Tool Turn:** Triggered by results returning from the MCP server (which may contain multiple tool results if the agent executed parallel calls).

### 3\. State Awareness

Unlike standard clients, this implementation is designed to interact with **State-Aware MCP Servers**. The client does not replicate the server's state machine. Instead, it reacts to state changes:

  * **Reactive Updates:** The client listens for notifications. When triggered, the adapter refreshes its internal capability cache and informs the agent.
  * **Error Tolerance:** If the agent attempts to call a tool that is no longer available (due to a state transition on the server), the client catches the error and feeds it back to the agent, allowing the LLM to self-correct.

## Setup and Installation

**Prerequisites:** Python **3.13.2** is required.

### 1\. Environment Setup

It is strictly recommended to use a virtual environment to manage dependencies.

**Windows:**

```powershell
# Create virtual environment
py -3.13 -m venv .venv

# Activate virtual environment
.venv\Scripts\activate
```

**Linux/macOS:**

```bash
# Create virtual environment
python3.13 -m venv .venv

# Activate virtual environment
source .venv/bin/activate
```

### 2\. Install Dependencies

Once the virtual environment is active, install the required packages using the provided `requirements.txt`.

```bash
pip install -r requirements.txt
```

## Configuration

The client configuration is managed via files located in:
`src/config/default/`

## Usage

### Command Line Interface (CLI)

The recommended way to use the client is via the CLI. This provides a terminal-based chat interface to interact with your MCP server.

**Windows:**

```powershell
py -m src.cli.main
```

**Linux/macOS:**

```bash
python -m src.cli.main
```

### API (Experimental)

The repository also includes a FastAPI implementation.
*Note: The API is currently experimental and has not been fully tested.*

**Windows:**

```powershell
py -m src.api.main
```

**Linux/macOS:**

```bash
python -m src.api.main
```

## Further Resources

  * **Server SDK:** To build the corresponding server, see the [fsm-mcp-python-sdk](https://github.com/TobiWan1995/fsm-mcp-python-sdk).
  * **Examples:** Complete end-to-end scenarios are available in [fsm-mcp-examples](https://github.com/TobiWan1995/fsm-mcp-examples).