from __future__ import annotations

from typing import Any

# Mock prompt calls for the /promptmock CLI command.
# Extend this mapping with your own JSON-RPC payloads to simulate prompts/get calls.
MOCK_PROMPT_CALLS: dict[str, dict[str, Any]] = {
    "hello_mock": {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "prompts/get",
        "params": {
            "name": "fake_prompt_on_server",
            "arguments": {"foo": "bar"},
        },
    },
}
