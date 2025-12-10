from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional


class BaseFileHandler(ABC):
    """
    Base class for UI-side handling of embedded files.
    This class should *not* feed the agent context.
    It returns strings that can be routed directly to the UI (for example via on_agent_response).
    """

    @abstractmethod
    async def stringify_if_supported(
        self,
        mime_type: str,
        blob_b64: str,
        name: Optional[str] = None,
        meta: Optional[dict] = None,
    ) -> Optional[str]:
        """
        Attempt to convert the blob into a displayable string.
        Return the string or None when the type is not supported yet.
        """
        ...
