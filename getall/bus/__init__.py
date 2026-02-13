"""Message bus module for decoupled channel-agent communication."""

from getall.bus.events import InboundMessage, OutboundMessage
from getall.bus.queue import MessageBus

__all__ = ["MessageBus", "InboundMessage", "OutboundMessage"]
