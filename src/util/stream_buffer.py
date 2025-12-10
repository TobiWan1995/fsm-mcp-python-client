# stream_buffer.py
from typing import Optional
from dataclasses import dataclass, field


@dataclass
class StreamBuffer:
    """
    A simple buffer that tracks sent content and returns deltas.
    Used by both CLI and API to calculate what's new.

    TODO: Investigate occasional truncation when used by the CLI streaming
    callbacks; this might stem from how AgentManager accumulates chunks or
    how we finalize/clear buffers between turns.
    """
    
    _buffers: dict[str, str] = field(default_factory=dict)
    
    def _get_buffer_key(self, user_id: str, chat_id: str, buffer_type: str) -> str:
        """Generate a unique key for each buffer"""
        return f"{user_id}:{chat_id}:{buffer_type}"
    
    def get_delta(
        self, 
        user_id: str, 
        chat_id: str, 
        content: str, 
        buffer_type: str = "response"
    ) -> tuple[Optional[str], bool]:
        """
        Get the delta (new content) by comparing with buffer.
        
        Args:
            user_id: User identifier
            chat_id: Chat identifier
            content: The full content (not just the new part)
            buffer_type: Type of buffer (response, thinking, tool)
            
        Returns:
            Tuple of (delta_content, is_first_output)
            - delta_content: The new content that hasn't been sent yet, or None if no change
            - is_first_output: True if this is the first output for this buffer
        """
        key = self._get_buffer_key(user_id, chat_id, buffer_type)
        
        # Get the current buffer content
        current_buffer = self._buffers.get(key, "")
        is_first = key not in self._buffers
        
        # If content hasn't changed, return None
        if current_buffer == content:
            return None, False
        
        # Find the new content (what's been added)
        if len(content) > len(current_buffer):
            # Content was added
            new_content = content[len(current_buffer):]
            # Update the buffer
            self._buffers[key] = content
            return new_content, is_first
        
        elif len(content) < len(current_buffer):
            # Content was replaced (not just appended)
            # This is a new message starting over
            self._buffers[key] = content
            return content, is_first
        
        return None, False
    
    def clear(self, user_id: str, chat_id: str, buffer_type: Optional[str] = None):
        """
        Clear buffer(s) for a specific user/chat combination.
        
        Args:
            user_id: User identifier
            chat_id: Chat identifier
            buffer_type: Specific buffer type to clear, or None to clear all
        """
        if buffer_type:
            key = self._get_buffer_key(user_id, chat_id, buffer_type)
            self._buffers.pop(key, None)
        else:
            # Clear all buffers for this user/chat
            prefix = f"{user_id}:{chat_id}:"
            keys_to_remove = [k for k in self._buffers.keys() if k.startswith(prefix)]
            for key in keys_to_remove:
                self._buffers.pop(key, None)
    
    def reset_all(self):
        """Reset all buffers"""
        self._buffers.clear()
