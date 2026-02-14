"""Message tool for sending messages to users."""

from contextvars import ContextVar
from typing import Any, Callable, Awaitable

from getall.agent.tools.base import Tool
from getall.bus.events import OutboundMessage

# Type alias for the handoff callback.
# Args: (channel, target_chat_id, content, source_channel, source_chat_id)
HandoffCallback = Callable[[str, str, str, str, str], None]


class MessageTool(Tool):
    """Tool to send messages to users on chat channels."""
    
    def __init__(
        self, 
        send_callback: Callable[[OutboundMessage], Awaitable[None]] | None = None,
        default_channel: str = "",
        default_chat_id: str = "",
        handoff_callback: HandoffCallback | None = None,
    ):
        self._send_callback = send_callback
        self._handoff_callback = handoff_callback
        self._default_channel_ctx: ContextVar[str] = ContextVar(
            "message_tool_default_channel",
            default=default_channel,
        )
        self._default_chat_id_ctx: ContextVar[str] = ContextVar(
            "message_tool_default_chat_id",
            default=default_chat_id,
        )
        self._sent_to_chat_ids_ctx: ContextVar[tuple[str, ...]] = ContextVar(
            "message_tool_sent_to_chat_ids",
            default=(),
        )
    
    def set_context(self, channel: str, chat_id: str) -> None:
        """Set the current message context."""
        self._default_channel_ctx.set(channel)
        self._default_chat_id_ctx.set(chat_id)
        self._sent_to_chat_ids_ctx.set(())  # Reset per message/task

    @property
    def sent_to_chat_ids(self) -> list[str]:
        """Return chat IDs messaged in the current task context."""
        return list(self._sent_to_chat_ids_ctx.get())
    
    def set_send_callback(self, callback: Callable[[OutboundMessage], Awaitable[None]]) -> None:
        """Set the callback for sending messages."""
        self._send_callback = callback
    
    @property
    def name(self) -> str:
        return "message"
    
    @property
    def description(self) -> str:
        return "Send a message to the user. Use this when you want to communicate something."
    
    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The message content to send"
                },
                "channel": {
                    "type": "string",
                    "description": "Optional: target channel (telegram, discord, etc.)"
                },
                "chat_id": {
                    "type": "string",
                    "description": "Optional: target chat/user ID"
                }
            },
            "required": ["content"]
        }
    
    async def execute(
        self, 
        content: str, 
        channel: str | None = None, 
        chat_id: str | None = None,
        **kwargs: Any
    ) -> str:
        source_channel = self._default_channel_ctx.get()
        source_chat_id = self._default_chat_id_ctx.get()
        channel = channel or source_channel
        chat_id = chat_id or source_chat_id
        
        if not channel or not chat_id:
            return "Error: No target channel/chat specified"
        
        if not self._send_callback:
            return "Error: Message sending not configured"
        
        msg = OutboundMessage(
            channel=channel,
            chat_id=chat_id,
            content=content
        )
        
        try:
            await self._send_callback(msg)
            sent_to = list(self._sent_to_chat_ids_ctx.get())
            sent_to.append(chat_id)
            self._sent_to_chat_ids_ctx.set(tuple(sent_to))

            # Cross-chat send: persist handoff so the target session
            # sees this message when the user replies in that chat.
            if self._handoff_callback and chat_id != source_chat_id:
                try:
                    self._handoff_callback(
                        channel, chat_id, content,
                        source_channel, source_chat_id,
                    )
                except Exception:
                    pass  # Best-effort; don't fail the send

            return f"Message sent to {channel}:{chat_id}"
        except Exception as e:
            return f"Error sending message: {str(e)}"
