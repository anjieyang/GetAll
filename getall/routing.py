"""Last-route tracking for automatic cron delivery.

When a user interacts via any external channel (Telegram, Feishu, Discord, ...),
we persist the route so that cron jobs can auto-deliver results without
hard-coding a target channel/chat_id.
"""

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from loguru import logger

from getall.utils.helpers import get_data_path


_ROUTE_FILE = "last_route.json"

# Channels considered "external" (a real user is on the other end).
# CLI and system channels are NOT recorded.
_EXTERNAL_CHANNELS = frozenset({
    "telegram", "feishu", "discord", "whatsapp", "slack",
    "dingtalk", "email", "qq", "mochat",
})


@dataclass
class LastRoute:
    """The most recent channel + chat_id the user interacted through."""
    channel: str
    chat_id: str
    updated_at: str


def _route_path() -> Path:
    return get_data_path() / _ROUTE_FILE


def save_last_route(channel: str, chat_id: str) -> None:
    """Persist the user's latest interaction route."""
    path = _route_path()
    route = LastRoute(
        channel=channel,
        chat_id=chat_id,
        updated_at=datetime.now().isoformat(),
    )
    try:
        path.write_text(json.dumps(asdict(route), ensure_ascii=False))
    except Exception as exc:
        logger.warning(f"Failed to save last route: {exc}")


def load_last_route() -> LastRoute | None:
    """Load the last known user interaction route, or ``None``."""
    path = _route_path()
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        return LastRoute(**data)
    except Exception as exc:
        logger.warning(f"Failed to load last route: {exc}")
        return None


def is_external_channel(channel: str) -> bool:
    """Return ``True`` if *channel* represents a real user-facing chat platform."""
    return channel.lower() in _EXTERNAL_CHANNELS
