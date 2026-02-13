"""Event types for the message bus."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class InboundMessage:
    """Message received from a chat channel."""

    channel: str  # telegram, discord, slack, whatsapp, feishu, …
    sender_id: str  # Platform-level user identifier
    chat_id: str  # Chat/group identifier
    content: str  # Message text
    timestamp: datetime = field(default_factory=datetime.now)
    media: list[str] = field(default_factory=list)  # Media URLs
    metadata: dict[str, Any] = field(default_factory=dict)  # Channel-specific data

    # ── identity-routing enrichment (set by IdentityRouterHook) ──
    tenant_id: str = "default"
    principal_id: str = ""  # filled after IFT resolution
    agent_identity_id: str = ""
    thread_id: str = ""
    sender_name: str = ""  # display name for group-chat attribution
    chat_type: str = "private"  # "private" | "group"

    @property
    def session_key(self) -> str:
        """Composite session key (legacy-compatible when principal_id empty)."""
        if self.principal_id:
            parts = [
                self.tenant_id, self.principal_id, self.channel,
                self.chat_id or "-", self.sender_id or "-", self.thread_id or "-",
            ]
            return "/".join(parts)
        return f"{self.channel}:{self.chat_id}"


@dataclass
class OutboundMessage:
    """Message to send to a chat channel."""

    channel: str
    chat_id: str
    content: str
    reply_to: str | None = None
    media: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    # optional routing hint
    principal_id: str = ""
    thread_id: str = ""


