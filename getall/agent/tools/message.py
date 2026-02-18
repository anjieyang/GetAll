"""Message tool for sending messages to users."""

from contextvars import ContextVar
from typing import Any, Callable, Awaitable

from loguru import logger

from getall.agent.tools.base import Tool
from getall.bus.events import OutboundMessage

# Type alias for the handoff callback.
# Args: (channel, target_chat_id, content, source_channel, source_chat_id)
HandoffCallback = Callable[[str, str, str, str, str], None]

# Callback: (channel, name_or_id) -> resolved_chat_id | None
RecipientResolver = Callable[[str, str], str | None]


class MessageTool(Tool):
    """Tool to send messages to users on chat channels."""
    
    def __init__(
        self, 
        send_callback: Callable[[OutboundMessage], Awaitable[None]] | None = None,
        default_channel: str = "",
        default_chat_id: str = "",
        handoff_callback: HandoffCallback | None = None,
        recipient_resolver: RecipientResolver | None = None,
    ):
        self._send_callback = send_callback
        self._handoff_callback = handoff_callback
        self._recipient_resolver = recipient_resolver
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
        # Tracks whether any same-chat message was marked final=true.
        self._final_delivered_ctx: ContextVar[bool] = ContextVar(
            "message_tool_final_delivered",
            default=False,
        )
    
    def set_context(self, channel: str, chat_id: str) -> None:
        """Set the current message context."""
        self._default_channel_ctx.set(channel)
        self._default_chat_id_ctx.set(chat_id)
        self._sent_to_chat_ids_ctx.set(())  # Reset per message/task
        self._final_delivered_ctx.set(False)

    @property
    def sent_to_chat_ids(self) -> list[str]:
        """Return chat IDs messaged in the current task context."""
        return list(self._sent_to_chat_ids_ctx.get())

    @property
    def final_delivered(self) -> bool:
        """Whether the agent marked a same-chat message as the final delivery."""
        return self._final_delivered_ctx.get()
    
    def set_send_callback(self, callback: Callable[[OutboundMessage], Awaitable[None]]) -> None:
        """Set the callback for sending messages."""
        self._send_callback = callback

    def set_recipient_resolver(self, resolver: RecipientResolver) -> None:
        """Set the callback for resolving user names to chat IDs."""
        self._recipient_resolver = resolver
    
    @property
    def name(self) -> str:
        return "message"
    
    @property
    def description(self) -> str:
        return (
            "Send a message to a user or chat. "
            "Use for progress updates (e.g. telling the user you're working on something) "
            "or for cross-chat delivery (e.g. DM from a group context). "
            "For cross-chat DMs, you can pass the user's display name as chat_id "
            "(e.g. chat_id=\"Andrew Yang\") — the system will resolve it to the "
            "correct platform ID. Do NOT fabricate platform IDs like ou_xxx. "
            "Set final=true ONLY when this message IS the complete answer and the loop's "
            "final response should be suppressed to avoid duplication."
        )
    
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
                    "description": (
                        "Optional: target chat/user ID or display name. "
                        "For DMs to group members, use their display name "
                        "(e.g. 'Andrew Yang') — the system resolves it automatically. "
                        "NEVER fabricate platform IDs."
                    ),
                },
                "final": {
                    "type": "boolean",
                    "description": (
                        "Set true when this message is the complete final answer "
                        "to suppress the loop's duplicate final response. "
                        "Default false (for progress updates, the loop's final "
                        "response will still be sent normally)."
                    ),
                    "default": False,
                },
            },
            "required": ["content"]
        }

    def _resolve_chat_id(self, channel: str, chat_id: str) -> str:
        """Resolve a chat_id that might be a display name to a platform ID.

        If the chat_id already looks like a platform ID (e.g. ou_xxx, oc_xxx),
        return as-is. Otherwise, try the recipient resolver.
        """
        # Already a platform ID — no resolution needed
        if chat_id.startswith(("ou_", "oc_", "on_")):
            return chat_id

        # Try resolver
        if self._recipient_resolver:
            resolved = self._recipient_resolver(channel, chat_id)
            if resolved:
                logger.info(
                    f"Resolved recipient '{chat_id}' -> {resolved} "
                    f"on {channel}"
                )
                return resolved

        logger.warning(
            f"Could not resolve recipient '{chat_id}' on {channel}, "
            f"using as-is"
        )
        return chat_id
    
    async def execute(
        self, 
        content: str, 
        channel: str | None = None, 
        chat_id: str | None = None,
        final: bool = False,
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

        # Resolve display names to platform IDs for cross-chat sends
        if chat_id != source_chat_id:
            chat_id = self._resolve_chat_id(channel, chat_id)
        
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

            # Mark final delivery for same-chat suppression
            if final and chat_id == source_chat_id:
                self._final_delivered_ctx.set(True)

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
