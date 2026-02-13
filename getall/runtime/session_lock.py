"""Per-session distributed lock backed by Redis.

Ensures that messages for the same session key are processed sequentially,
even across multiple service instances.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from redis.asyncio import Redis


class SessionLockTimeout(RuntimeError):
    """Raised when lock acquisition times out."""


class SessionLock:
    def __init__(self, redis: Redis, ttl_seconds: int = 30) -> None:
        self._redis = redis
        self._ttl = ttl_seconds
        self._local: dict[str, asyncio.Lock] = {}

    @asynccontextmanager
    async def acquire(self, session_key: str, timeout: float = 5.0) -> AsyncIterator[None]:
        lock_key = f"getall:lock:{session_key}"
        rlock = self._redis.lock(lock_key, timeout=self._ttl)
        acquired = await rlock.acquire(blocking=True, blocking_timeout=timeout)
        if acquired:
            try:
                yield
            finally:
                try:
                    await rlock.release()
                except Exception:
                    pass
            return

        # Fallback to in-process lock (single-instance scenario)
        local = self._local.setdefault(session_key, asyncio.Lock())
        try:
            await asyncio.wait_for(local.acquire(), timeout=timeout)
        except TimeoutError as exc:
            raise SessionLockTimeout(f"lock timeout: {session_key}") from exc
        try:
            yield
        finally:
            local.release()
