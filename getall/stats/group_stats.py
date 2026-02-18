"""Group chat message statistics backed by Redis sorted sets.

Records every group message (regardless of @-mention) and provides
time-windowed queries for the agent to answer questions like
"who talks the most" or "how active is this group".

Redis key layout (per group chat):
  ga:grp:{chat_id}:msgs   — Sorted Set  (score=unix_ts, member=sender|flag|ts_ms)
  ga:grp:{chat_id}:names  — Hash         (sender_id → display_name)
"""

from __future__ import annotations

import random
import time
from collections import Counter
from typing import Any

import redis.asyncio as aioredis
from loguru import logger

# ---------------------------------------------------------------------------
# Key helpers
# ---------------------------------------------------------------------------

_PREFIX = "ga:grp"
_RETENTION_DAYS = 90


def _msgs_key(chat_id: str) -> str:
    return f"{_PREFIX}:{chat_id}:msgs"


def _names_key(chat_id: str) -> str:
    return f"{_PREFIX}:{chat_id}:names"


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class GroupStatsService:
    """Lightweight group message counter backed by Redis."""

    def __init__(self, redis_url: str) -> None:
        self._redis: aioredis.Redis = aioredis.from_url(
            redis_url,
            decode_responses=True,
            socket_connect_timeout=3,
        )

    # ── write path ──────────────────────────────────────────────────

    async def record(
        self,
        chat_id: str,
        sender_id: str,
        sender_name: str,
        bot_mentioned: bool,
    ) -> None:
        """Record a single group message. Fire-and-forget safe."""
        try:
            ts = time.time()
            ts_ns = time.time_ns()
            flag = "1" if bot_mentioned else "0"
            member = f"{sender_id}|{flag}|{ts_ns}"

            pipe = self._redis.pipeline(transaction=False)
            pipe.zadd(_msgs_key(chat_id), {member: ts})
            if sender_name and not sender_name.startswith("ou_"):
                pipe.hset(_names_key(chat_id), sender_id, sender_name)
            await pipe.execute()

            # Probabilistic cleanup: ~1 % of writes trigger trimming
            if random.random() < 0.01:
                await self.cleanup(chat_id)
        except Exception as exc:
            logger.debug(f"group_stats record failed (non-critical): {exc}")

    # ── read path ───────────────────────────────────────────────────

    async def top_senders(
        self,
        chat_id: str,
        days: int = 7,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Return top senders ranked by message count in the time window.

        Each entry: {name, sender_id, total, at_bot, not_at_bot}
        """
        start_ts = time.time() - days * 86400
        members: list[str] = await self._redis.zrangebyscore(
            _msgs_key(chat_id), start_ts, "+inf",
        )

        total_counter: Counter[str] = Counter()
        bot_counter: Counter[str] = Counter()
        for m in members:
            parts = m.split("|", 2)
            sid = parts[0]
            total_counter[sid] += 1
            if len(parts) >= 2 and parts[1] == "1":
                bot_counter[sid] += 1

        names: dict[str, str] = await self._redis.hgetall(_names_key(chat_id))

        result: list[dict[str, Any]] = []
        for sid, total in total_counter.most_common(limit):
            result.append({
                "sender_id": sid,
                "name": names.get(sid, sid),
                "total": total,
                "at_bot": bot_counter.get(sid, 0),
                "not_at_bot": total - bot_counter.get(sid, 0),
            })
        return result

    async def summary(
        self,
        chat_id: str,
        days: int = 7,
    ) -> dict[str, Any]:
        """Aggregate stats for a group chat over *days*."""
        start_ts = time.time() - days * 86400
        members: list[str] = await self._redis.zrangebyscore(
            _msgs_key(chat_id), start_ts, "+inf",
        )

        senders: set[str] = set()
        at_bot = 0
        for m in members:
            parts = m.split("|", 2)
            senders.add(parts[0])
            if len(parts) >= 2 and parts[1] == "1":
                at_bot += 1

        total = len(members)
        return {
            "total_messages": total,
            "unique_senders": len(senders),
            "at_bot_messages": at_bot,
            "not_at_bot_messages": total - at_bot,
            "daily_avg": round(total / max(days, 1), 1),
            "days": days,
        }

    # ── maintenance ─────────────────────────────────────────────────

    async def cleanup(
        self,
        chat_id: str,
        retention_days: int = _RETENTION_DAYS,
    ) -> int:
        """Remove entries older than *retention_days*. Returns removed count."""
        cutoff = time.time() - retention_days * 86400
        try:
            removed: int = await self._redis.zremrangebyscore(
                _msgs_key(chat_id), "-inf", cutoff,
            )
            if removed:
                logger.debug(
                    f"group_stats cleanup: removed {removed} old entries "
                    f"from {chat_id}"
                )
            return removed
        except Exception as exc:
            logger.debug(f"group_stats cleanup failed: {exc}")
            return 0


# ---------------------------------------------------------------------------
# Module-level lazy singleton
# ---------------------------------------------------------------------------

_instance: GroupStatsService | None = None


def get_stats_service() -> GroupStatsService:
    """Return (or create) the module-level GroupStatsService singleton."""
    global _instance
    if _instance is None:
        from getall.settings import get_settings
        _instance = GroupStatsService(get_settings().redis_url)
    return _instance
