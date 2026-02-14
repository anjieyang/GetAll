"""Group chat statistics tool — exposes Redis-backed message counts to the agent."""

from __future__ import annotations

import json
from contextvars import ContextVar
from typing import Any

from getall.agent.tools.base import Tool


class GroupStatsTool(Tool):
    """Query group chat message statistics (who talks the most, activity summary, etc.)."""

    def __init__(self) -> None:
        self._chat_id_ctx: ContextVar[str] = ContextVar(
            "group_stats_chat_id", default=""
        )
        self._chat_type_ctx: ContextVar[str] = ContextVar(
            "group_stats_chat_type", default=""
        )

    # ── context injection (called per-message by AgentLoop) ──

    def set_context(self, chat_id: str, chat_type: str) -> None:
        self._chat_id_ctx.set(chat_id)
        self._chat_type_ctx.set(chat_type)

    # ── Tool interface ──────────────────────────────────────────

    @property
    def name(self) -> str:
        return "group_stats"

    @property
    def description(self) -> str:
        return (
            "Query group chat message statistics. "
            "Use 'top_senders' to rank members by message count, "
            "or 'summary' for aggregate activity metrics. "
            "Only works in group chats."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["top_senders", "summary"],
                    "description": (
                        "top_senders: rank members by message count. "
                        "summary: aggregate stats (total messages, unique senders, daily avg)."
                    ),
                },
                "days": {
                    "type": "integer",
                    "description": "Time window in days (default 7).",
                    "minimum": 1,
                    "maximum": 90,
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results for top_senders (default 10).",
                    "minimum": 1,
                    "maximum": 50,
                },
            },
            "required": ["action"],
        }

    async def execute(
        self,
        action: str,
        days: int = 7,
        limit: int = 10,
        **kwargs: Any,
    ) -> str:
        chat_type = self._chat_type_ctx.get()
        if chat_type != "group":
            return json.dumps({"error": "group_stats is only available in group chats."})

        chat_id = self._chat_id_ctx.get()
        if not chat_id:
            return json.dumps({"error": "No group chat_id in current context."})

        from getall.stats.group_stats import get_stats_service

        svc = get_stats_service()

        try:
            if action == "top_senders":
                data = await svc.top_senders(chat_id, days=days, limit=limit)
                if not data:
                    return json.dumps({
                        "result": [],
                        "note": f"No message data recorded for the past {days} day(s) yet.",
                    })
                return json.dumps({"days": days, "result": data}, ensure_ascii=False)

            if action == "summary":
                data = await svc.summary(chat_id, days=days)
                return json.dumps({"result": data}, ensure_ascii=False)

            return json.dumps({"error": f"Unknown action: {action}"})
        except Exception as exc:
            return json.dumps({"error": f"Stats query failed: {exc}"})
