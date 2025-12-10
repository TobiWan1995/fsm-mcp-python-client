from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any


class ConversationWriter:
    """Persists CLI conversation history grouped in timestamped directories."""

    def __init__(self, base_dir: str = "history") -> None:
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        self.root = Path(base_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.log_file = self.root / f"conversation_{timestamp}.txt"

    def _append(self, header: str, content: str) -> None:
        with self.log_file.open("a", encoding="utf-8") as fh:
            fh.write("=" * 80 + "\n")
            fh.write(f"{header}\n")
            fh.write("-" * 80 + "\n")
            fh.write(content.strip() + "\n\n")

    def record_user(self, message: str) -> None:
        self._append("USER MESSAGE", message)

    def record_agent(self, message: str) -> None:
        self._append("AGENT RESPONSE", message)

    def record_thinking(self, content: str) -> None:
        self._append("AGENT THINKING", content)

    def record_tool_call(self, method: str, params: dict) -> None:
        formatted = f"Method: {method}\nParameters:\n{json.dumps(params, indent=2)}"
        self._append("TOOL CALL", formatted)

    def record_tool_response(self, content: str) -> None:
        self._append("TOOL RESPONSE", content)
