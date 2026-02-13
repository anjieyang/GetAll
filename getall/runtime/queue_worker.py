"""Redis-backed async task queue for background work (research, review, evolution)."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from redis.asyncio import Redis


@dataclass(frozen=True, slots=True)
class QueueTask:
    task_type: str
    payload: dict[str, str]


TaskHandler = Callable[[QueueTask], Awaitable[None]]


class QueueWorker:
    def __init__(self, redis: Redis, queue_name: str = "getall:tasks") -> None:
        self._redis = redis
        self._queue = queue_name
        self._running = False
        self._handlers: dict[str, TaskHandler] = {}

    def register(self, task_type: str, handler: TaskHandler) -> None:
        self._handlers[task_type] = handler

    async def enqueue(self, task: QueueTask) -> None:
        raw = json.dumps({"task_type": task.task_type, "payload": task.payload})
        await self._redis.lpush(self._queue, raw)

    async def run_forever(self, poll_seconds: float = 1.0) -> None:
        self._running = True
        while self._running:
            entry = await self._redis.brpop(self._queue, timeout=int(poll_seconds))
            if entry is None:
                continue
            _, raw = entry
            await self._dispatch(raw.decode("utf-8"))

    async def _dispatch(self, raw: str) -> None:
        data = json.loads(raw)
        task = QueueTask(task_type=data["task_type"], payload=data["payload"])
        handler = self._handlers.get(task.task_type)
        if handler is not None:
            await handler(task)

    async def stop(self) -> None:
        self._running = False
        await asyncio.sleep(0)
