from __future__ import annotations
from typing import Optional
from base64 import b64decode

from src.util.file.file_handler import BaseFileHandler


class OpenWebUIMarkdownFileHandler(BaseFileHandler):
    """
    OpenWebUI-specific implementation:
    - Currently supports only Markdown (text/markdown or text/x-markdown).
    - Converts the blob to a UTF-8 string.
    - The returned string can be sent directly to the UI via on_agent_response.
    """

    def _is_markdown(self, mime: str) -> bool:
        m = (mime or "").lower().strip()
        return m.startswith("text/markdown") or m == "text/x-markdown"

    async def stringify_if_supported(
        self,
        mime_type: str,
        blob_b64: str,
        name: Optional[str] = None,
        meta: Optional[dict] = None,
    ) -> Optional[str]:
        if not self._is_markdown(mime_type):
            return None

        try:
            md = b64decode(blob_b64).decode("utf-8", errors="replace")
        except Exception as e:
            md = f"[Error decoding markdown file: {e}]"

        title = f"**{name}**\n\n" if name else ""
        # Important: do not return this to the agent; this string is meant for the UI.
        return f"\n\n{title}{md}\n\n"
