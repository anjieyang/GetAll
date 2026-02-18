"""Per-principal route registry for notification delivery.

When a user interacts via any external channel (Telegram, Feishu, Discord, …),
we persist the route **keyed by principal_id** so that cron jobs, WS handlers,
and proactive notifications can deliver results to the correct user/chat.

Storage: ``~/.getall/data/routes.json``
Format:
    {
        "<principal_id>": {
            "channel": "telegram",
            "chat_id": "123456",
            "updated_at": "2026-02-13T15:30:00"
        },
        ...
    }
"""

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from loguru import logger

from getall.utils.helpers import get_data_path


_ROUTES_FILE = "routes.json"

# Channels considered "external" (a real user is on the other end).
# CLI and system channels are NOT recorded.
_EXTERNAL_CHANNELS = frozenset({
    "telegram", "feishu", "discord", "whatsapp", "slack",
    "dingtalk", "email", "qq", "mochat",
})


@dataclass
class LastRoute:
    """A principal's most recent channel + chat_id."""
    channel: str
    chat_id: str
    updated_at: str


# ── Internal storage ────────────────────────────────────────────────
#
# Structure per principal:
#   {
#       "<principal_id>": {
#           "private": {"channel": "...", "chat_id": "...", "updated_at": "..."},
#           "group":   {"channel": "...", "chat_id": "...", "updated_at": "..."}
#       }
#   }
#
# Monitoring (cron, WS) uses the **private** route only so personal
# trading data is never leaked to a group.
# Group route is stored for group-specific delivery if needed later.


def _routes_path() -> Path:
    return get_data_path() / _ROUTES_FILE


def _load_all() -> dict[str, dict[str, dict[str, str]]]:
    """Load the entire routes registry from disk."""
    path = _routes_path()
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        # Migrate legacy flat format → nested {private/group}
        migrated = False
        for pid, val in list(raw.items()):
            if isinstance(val, dict) and "channel" in val:
                # Old flat format: treat as private route
                raw[pid] = {"private": val}
                migrated = True
        if migrated:
            _save_all(raw)
        return raw
    except Exception as exc:
        logger.warning(f"Failed to load routes registry: {exc}")
        return {}


def _save_all(data: dict[str, dict[str, dict[str, str]]]) -> None:
    path = _routes_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    except Exception as exc:
        logger.warning(f"Failed to save routes registry: {exc}")


# ── Public API ──────────────────────────────────────────────────────

def save_last_route(
    channel: str,
    chat_id: str,
    principal_id: str,
    chat_type: str = "private",
) -> None:
    """Persist a principal's latest interaction route.

    Only records routes from external channels (telegram, feishu, etc.).
    Private and group routes are stored separately so monitoring
    notifications always go to private chat (never leaked to groups).
    """
    if not principal_id:
        return
    if channel.lower() not in _EXTERNAL_CHANNELS:
        return

    slot = "group" if chat_type == "group" else "private"
    data = _load_all()
    data.setdefault(principal_id, {})
    data[principal_id][slot] = {
        "channel": channel,
        "chat_id": chat_id,
        "updated_at": datetime.now().isoformat(),
    }
    _save_all(data)
    logger.debug(f"Route saved: principal={principal_id} [{slot}] → {channel}:{chat_id}")


def load_last_route(
    principal_id: str | None = None,
    prefer_private: bool = True,
) -> LastRoute | None:
    """Load the last known route for a specific principal.

    Args:
        principal_id: Target principal. ``None`` returns the most recent
            private route across all principals (backward-compatible).
        prefer_private: If ``True`` (default), return the private-chat
            route when available, falling back to group. Monitoring
            callers should always use ``prefer_private=True``.
    """
    data = _load_all()
    if not data:
        return None

    def _pick(entry: dict[str, dict[str, str]]) -> LastRoute | None:
        if prefer_private:
            r = entry.get("private") or entry.get("group")
        else:
            # Most recently updated across private and group
            candidates = [v for v in (entry.get("private"), entry.get("group")) if v]
            r = max(candidates, key=lambda v: v.get("updated_at", "")) if candidates else None
        if r:
            return LastRoute(
                channel=r["channel"],
                chat_id=r["chat_id"],
                updated_at=r.get("updated_at", ""),
            )
        return None

    if principal_id and principal_id in data:
        return _pick(data[principal_id])

    # Fallback: most recent private route across all principals
    best: LastRoute | None = None
    for entry in data.values():
        route = _pick(entry)
        if route and (best is None or route.updated_at > best.updated_at):
            best = route
    return best


def load_all_routes(prefer_private: bool = True) -> dict[str, LastRoute]:
    """Return routes for all principals: ``{principal_id: LastRoute}``.

    By default returns the private-chat route for each principal so
    monitoring notifications stay in private conversations.
    """
    data = _load_all()
    result: dict[str, LastRoute] = {}
    for pid, entry in data.items():
        try:
            if prefer_private:
                r = entry.get("private") or entry.get("group")
            else:
                candidates = [v for v in (entry.get("private"), entry.get("group")) if v]
                r = max(candidates, key=lambda v: v.get("updated_at", "")) if candidates else None
            if r:
                result[pid] = LastRoute(
                    channel=r["channel"],
                    chat_id=r["chat_id"],
                    updated_at=r.get("updated_at", ""),
                )
        except Exception:
            pass
    return result


def load_all_group_routes() -> list[tuple[str, str]]:
    """Return deduplicated ``(channel, chat_id)`` pairs for all known group chats.

    Iterates every principal's stored routes and collects unique group
    entries.  Useful for admin broadcast to all groups.
    """
    data = _load_all()
    seen: set[tuple[str, str]] = set()
    for entry in data.values():
        grp = entry.get("group")
        if grp:
            key = (grp["channel"], grp["chat_id"])
            seen.add(key)
    return sorted(seen)


def load_all_private_routes() -> list[tuple[str, str]]:
    """Return deduplicated ``(channel, chat_id)`` pairs for all known private chats.

    Iterates every principal's stored routes and collects unique private
    entries.  Useful for broadcasting system notifications to all users.
    """
    data = _load_all()
    seen: set[tuple[str, str]] = set()
    for entry in data.values():
        priv = entry.get("private")
        if priv and priv.get("channel") and priv.get("chat_id"):
            seen.add((priv["channel"], priv["chat_id"]))
    return sorted(seen)


def remove_stale_route(channel: str, chat_id: str) -> int:
    """Remove all route entries matching *channel* + *chat_id*.

    Called when a broadcast/send receives a permanent delivery failure
    (e.g. bot removed from chat, user deleted conversation).

    Returns the number of entries removed.
    """
    data = _load_all()
    removed = 0
    for pid in list(data):
        entry = data[pid]
        for slot in ("private", "group"):
            r = entry.get(slot)
            if r and r.get("channel") == channel and r.get("chat_id") == chat_id:
                del entry[slot]
                removed += 1
        # Drop the principal entirely if no routes remain
        if not entry:
            del data[pid]
    if removed:
        _save_all(data)
        logger.info(f"Removed {removed} stale route(s) for {channel}:{chat_id}")
    return removed


def is_external_channel(channel: str) -> bool:
    """Return ``True`` if *channel* represents a real user-facing chat platform."""
    return channel.lower() in _EXTERNAL_CHANNELS
