"""Feedback tool — create, list, and resolve bug reports / suggestions.

Available to all users for creating feedback. Admin-only for listing all
feedback and resolving items. When an item is resolved, the original
reporter is notified via private message.
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Callable, Awaitable

from loguru import logger

from getall.agent.tools.base import Tool
from getall.bus.events import OutboundMessage
from getall.storage.models import Feedback, FeedbackSource, FeedbackStatus


@dataclass(frozen=True, slots=True)
class _FeedbackContext:
    principal_id: str = ""
    role: str = "user"
    session_key: str = ""
    session_factory: Any = None
    send_callback: Callable[[OutboundMessage], Awaitable[None]] | None = None


class FeedbackTool(Tool):
    """Create, list, and manage user feedback / bug reports."""

    name = "feedback"
    description = (
        "Handle user feedback, suggestions, and bug reports. "
        "Actions: "
        "create — submit new feedback (available to everyone, agent can also create with source='agent'); "
        "list — show feedback (users see their own, admins see all); "
        "resolve — mark feedback as resolved and notify the reporter (admin only); "
        "update_status — change feedback status (admin only)."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["create", "list", "resolve", "update_status"],
                "description": "The feedback action to perform",
            },
            "category": {
                "type": "string",
                "enum": ["bug", "suggestion", "complaint"],
                "description": "(create) Feedback category",
            },
            "summary": {
                "type": "string",
                "description": "(create) Short summary of the feedback",
            },
            "detail": {
                "type": "string",
                "description": "(create) Detailed description / context",
            },
            "source": {
                "type": "string",
                "enum": ["user", "agent"],
                "description": "(create) Who initiated this: 'user' (explicitly reported) or 'agent' (auto-detected). Default: user",
            },
            "feedback_id": {
                "type": "string",
                "description": "(resolve/update_status) The feedback item ID",
            },
            "status": {
                "type": "string",
                "enum": ["pending", "in_progress", "resolved", "wont_fix"],
                "description": "(update_status) New status",
            },
            "resolution_note": {
                "type": "string",
                "description": "(resolve) Explanation of the resolution",
            },
            "filter_status": {
                "type": "string",
                "enum": ["pending", "in_progress", "resolved", "wont_fix", "all"],
                "description": "(list) Filter by status. Default: pending",
            },
        },
        "required": ["action"],
    }

    def __init__(self) -> None:
        self._context: ContextVar[_FeedbackContext] = ContextVar(
            "feedback_tool_context", default=_FeedbackContext(),
        )

    def set_context(
        self,
        principal_id: str,
        role: str,
        session_key: str,
        session_factory: Any,
        send_callback: Callable[[OutboundMessage], Awaitable[None]] | None = None,
    ) -> None:
        self._context.set(_FeedbackContext(
            principal_id=principal_id,
            role=role,
            session_key=session_key,
            session_factory=session_factory,
            send_callback=send_callback,
        ))

    def clear_context(self) -> None:
        self._context.set(_FeedbackContext())

    async def execute(self, action: str = "", **kwargs: Any) -> str:
        ctx = self._context.get()
        if not ctx.principal_id or not ctx.session_factory:
            return "Error: feedback tool context not set"

        if action == "create":
            return await self._create(ctx, kwargs)
        elif action == "list":
            return await self._list(ctx, kwargs.get("filter_status", "pending"))
        elif action == "resolve":
            return await self._resolve(ctx, kwargs.get("feedback_id", ""), kwargs.get("resolution_note", ""))
        elif action == "update_status":
            return await self._update_status(
                ctx, kwargs.get("feedback_id", ""), kwargs.get("status", ""),
            )
        else:
            return f"Error: unknown action '{action}'"

    async def _create(self, ctx: _FeedbackContext, kwargs: dict[str, Any]) -> str:
        category = kwargs.get("category", "")
        summary = kwargs.get("summary", "")
        if not category or not summary:
            return "Error: category and summary are required"

        source_str = kwargs.get("source", "user")
        try:
            source = FeedbackSource(source_str)
        except ValueError:
            source = FeedbackSource.USER

        from getall.storage.repository import FeedbackRepo

        fb = Feedback(
            principal_id=ctx.principal_id,
            source=source,
            category=category,
            summary=summary,
            detail=kwargs.get("detail", ""),
            session_key=ctx.session_key,
        )

        async with ctx.session_factory() as session:
            repo = FeedbackRepo(session)
            fb = await repo.create(fb)
            await session.commit()

        short_id = fb.id[:8]
        return f"反馈已记录 (#{short_id})。类型: {category}，摘要: {summary}"

    async def _list(self, ctx: _FeedbackContext, filter_status: str) -> str:
        from getall.storage.repository import FeedbackRepo, IdentityRepo

        status_filter: FeedbackStatus | None = None
        if filter_status and filter_status != "all":
            try:
                status_filter = FeedbackStatus(filter_status)
            except ValueError:
                pass

        async with ctx.session_factory() as session:
            repo = FeedbackRepo(session)
            id_repo = IdentityRepo(session)

            if ctx.role == "admin":
                items = await repo.list_by_status(status_filter, limit=50)
            else:
                items = await repo.list_by_principal(ctx.principal_id, limit=20)

            if not items:
                return "没有找到反馈记录。"

            # Build a principal_id → name lookup for display
            principal_ids = {fb.principal_id for fb in items}
            names: dict[str, str] = {}
            for pid in principal_ids:
                p = await id_repo.get_by_id(pid)
                if p:
                    names[pid] = p.pet_name or p.ift[:12]

        lines = [f"共 {len(items)} 条反馈:\n"]
        status_icons = {
            FeedbackStatus.PENDING: "🔴",
            FeedbackStatus.IN_PROGRESS: "🟡",
            FeedbackStatus.RESOLVED: "🟢",
            FeedbackStatus.WONT_FIX: "⚪",
        }
        for fb in items:
            icon = status_icons.get(fb.status, "❓")
            who = names.get(fb.principal_id, "unknown")
            line = (
                f"{icon} #{fb.id[:8]} [{fb.category}] {fb.summary}\n"
                f"   来源: {fb.source.value} | 用户: {who} | "
                f"{fb.created_at.strftime('%m-%d %H:%M')} | 状态: {fb.status.value}"
            )
            if fb.resolution_note:
                line += f"\n   解决说明: {fb.resolution_note}"
            lines.append(line)

        return "\n".join(lines)

    async def _resolve(self, ctx: _FeedbackContext, feedback_id: str, note: str) -> str:
        if ctx.role != "admin":
            return "Error: 只有管理员可以解决反馈"
        if not feedback_id:
            return "Error: feedback_id is required"

        from getall.storage.repository import FeedbackRepo, IdentityRepo

        async with ctx.session_factory() as session:
            repo = FeedbackRepo(session)
            fb = await repo.update_status(
                feedback_id, FeedbackStatus.RESOLVED,
                resolution_note=note, resolved_by=ctx.principal_id,
            )
            if fb is None:
                await session.rollback()
                return f"Error: feedback {feedback_id} not found"

            # Get reporter info for notification
            id_repo = IdentityRepo(session)
            reporter = await id_repo.get_by_id(fb.principal_id)
            await session.commit()

        result = f"反馈 #{feedback_id[:8]} 已标记为已解决。"

        # Notify the original reporter via private message
        if reporter and ctx.send_callback:
            await self._notify_reporter(ctx, reporter, fb, note)
            result += " 已通知原始用户。"

        return result

    async def _update_status(self, ctx: _FeedbackContext, feedback_id: str, status: str) -> str:
        if ctx.role != "admin":
            return "Error: 只有管理员可以更新反馈状态"
        if not feedback_id or not status:
            return "Error: feedback_id and status are required"

        try:
            new_status = FeedbackStatus(status)
        except ValueError:
            return f"Error: invalid status '{status}'"

        from getall.storage.repository import FeedbackRepo

        async with ctx.session_factory() as session:
            repo = FeedbackRepo(session)
            fb = await repo.update_status(feedback_id, new_status)
            await session.commit()

        if fb is None:
            return f"Error: feedback {feedback_id} not found"
        return f"反馈 #{feedback_id[:8]} 状态已更新为 {status}"

    async def _notify_reporter(
        self,
        ctx: _FeedbackContext,
        reporter: Any,
        fb: Feedback,
        note: str,
    ) -> None:
        """Send a private message to the feedback reporter about resolution."""
        if not ctx.send_callback:
            return

        try:
            from getall.routing import load_last_route

            route = load_last_route(principal_id=reporter.id, prefer_private=True)
            if not route:
                logger.info(f"No route for reporter {reporter.id[:8]}, skip notification")
                return

            name = reporter.pet_name or "你"
            msg = (
                f"🎉 {name}，你之前反馈的问题已经处理好了！\n\n"
                f"反馈: {fb.summary}\n"
            )
            if note:
                msg += f"解决说明: {note}\n"
            msg += "\n可以再试试看，有问题随时说。"

            await ctx.send_callback(OutboundMessage(
                channel=route.channel,
                chat_id=route.chat_id,
                content=msg,
            ))
            logger.info(f"Feedback resolution notification sent to {reporter.id[:8]}")
        except Exception as exc:
            logger.warning(f"Failed to notify reporter {reporter.id[:8]}: {exc}")
