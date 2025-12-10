# Agent Layer – MCP‑first, backend‑agnostic

A lean agent runtime that orchestrates **multiple agents per session** and uses **MCP (Model Context Protocol)** as the single IO surface (tools, prompts, resources). The system is intentionally **provider‑agnostic**: concrete model runtimes (e.g., Ollama, OpenAI, Claude) are integrated via adapters without changing the core.

---

## Purpose (meta)

* Provide a stable, model‑agnostic layer around MCP for conversational agents.
* Keep UI/API consumers simple: a session interface with streaming callbacks and a unified way to execute MCP requests.
* Isolate provider specifics behind a narrow adapter contract so swaps or additions do not ripple through the codebase.

---

## Design principles

* **Single protocol surface**: All tool/prompt/resource interactions go through MCP.
* **Separation of concerns**:

  * *AgentManager* handles session lifecycle, streaming callbacks, and the execution of MCP requests.
  * *Adapter* translates between provider features (function calling, schemas, content forms) and MCP, in both directions.
  * *MCP client* speaks MCP only and publishes capability changes.
* **Non‑intrusive artifacts**: large binaries (files, audio, images) are routed to the consumer/UI and do not enter the agent context by default.
* **Composable extension**: providers are added by implementing the same adapter contract; the manager and client remain unchanged.

---

## Role boundaries (conceptual)

* **AgentManager**

  * Owns sessions and coordinates message flow.
  * Emits streaming callbacks (thinking/content/tool‑call/completion).
  * Executes MCP requests and forwards results for mapping.
* **Adapter (per provider)**

  * Receives capability updates from the MCP client.
  * Exposes provider tool specifications to the agent when required.
  * Translates provider tool calls → MCP requests and MCP results → agent messages vs. UI artifacts.
* **MCP client**

  * Lists/calls tools, prompts, resources.
  * Caches capabilities and notifies adapters on change.

---

## Runtime flows (abstract)

* **Capabilities**: MCP client loads capabilities → adapter updates its internal maps (wired by the manager) → agent can include current tool specs in provider calls when applicable.
* **Tool calls**: agent returns content/thinking and provider‑native tool calls → adapter emits MCP requests → manager executes via MCP client → adapter maps results to agent messages and UI artifacts → manager issues callbacks.

## Turn orchestration & auto‑continue (conceptual)

* The manager uses a message queue per session to drive the conversation loop.
* A user message enqueues a turn; the agent produces content and/or tool calls.
* When tool calls execute, their results are appended as messages and **automatically enqueue a follow‑up turn**, allowing the agent to decide whether to respond, call another tool, or stop.
* This enables **semi‑autonomous** operation: human ↔ agent turns when no tools are needed; agent‑driven follow‑ups when tools are involved.
* UI callbacks (thinking/content/tool‑call/completion) reflect each step; large artifacts bypass the agent and are surfaced directly to the consumer.

---

## Configuration & extensibility (meta)

* Provider is selected per session; the same manager and client are reused.
* Adapters define how provider options/messages are represented internally; the base agent exposes generic hooks to carry these without prescribing formats.

---

## Error handling & observability (meta)

* Clear, actionable errors for unmapped tools/prompts/resources.
* Structured logging at adapter and manager boundaries; no sensitive payloads in logs.
* Fail‑closed defaults for unknown content types (treated as artifacts).

---

## Adapter pattern (at a glance)

```
Client (CLI / API)
        │
        ▼
  AgentManager  ── manages ──►  Session
        │                         │
        │                         ├─ Agent (provider impl)
        │                         │
        │                         ├─ Adapter (per provider)
        │                         │     ├─ maps provider tool_calls → MCP requests
        │                         │     └─ maps MCP results → agent messages / UI artifacts
        │                         │
        │                         └─ MCPClient (per session) ⇄ MCP Server
        │
        └─ UI callbacks (thinking/content/tool‑call/completion, file‑handler)
```

Flow summary:

* **Requests**: provider tool_calls → adapter → AgentManager → MCPClient → MCP server.
* **Results**: MCP server → MCPClient → AgentManager → adapter → (agent messages + UI artifacts).

Only the **MCPClient** communicates with the MCP server; the adapter never calls the server directly.

## AgentManager & session management (conceptual)

1. **Session creation**: a provider is selected; the manager constructs the provider agent and the matching adapter, then wires the adapter to capability updates from the MCP client.
2. **Turn handling**: for each user message the agent produces streaming **content/thinking** and provider‑native **tool calls**.
3. **Tool execution**: the manager forwards tool calls to the adapter, which emits MCP requests; the manager executes them through the MCP client.
4. **Result mapping**: the adapter splits MCP responses into agent messages vs. UI artifacts (bypass). The manager appends agent messages to the conversation and emits callbacks for artifacts/content/thinking/completion.
5. **Isolation**: provider swaps do not affect the manager or MCP client; only the adapter and agent change.

---

## Entry points (CLI & API)

* **CLI**: interactive loop that sends user input to the manager and renders streaming callbacks in-place. Supported callbacks: **thinking**, **content**, **tool-call**, **completion**. Artifacts (e.g., files/blobs) are emitted via a **file‑handler callback** for direct UI rendering, bypassing the agent context.
* **API**: service endpoints expose the same session abstraction and callback model. Streaming transports forward deltas for **thinking/content/tool‑call/completion**; artifacts are surfaced through the file‑handler path for clients to fetch or display.

Both entry points share the same core: *AgentManager + Adapter + MCPClient*. Provider selection is a session parameter.

---

## MCP client capabilities (current & planned)

**Current**

* **Tools / Prompts / Resources**: list, call, and read via MCP.
* **Server‑initiated Sampling**: the MCP server can request a sampling operation on the client; results are returned to the server.
* **Transactions (custom extension)**: the server can open/commit/abort a transaction; pending changes (e.g., system prompt or messages) are applied only on **commit** and discarded on **abort**. Requires server support.

**Planned**

* **Notifications**, **Logins**, **Elicitation** (design placeholders; not implemented).

---

## Notes

* Keep provider temperatures conservative for function/tool calling when argument stability matters.
* Schema validation for tool arguments is recommended on the client side in addition to MCP server validation.

---
