"""Reminders tool backed by cron scheduling."""

from __future__ import annotations

import asyncio
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Callable

from loguru import logger

from getall.agent.tools.base import Tool
from getall.bus.events import OutboundMessage
from getall.cron.service import CronService
from getall.cron.types import CronSchedule

# Type for the heartbeat factory set by the agent loop.
# (delay_seconds, message) -> heartbeat_id
HeartbeatFactory = Callable[[int, str], str]
# (heartbeat_id) -> bool
HeartbeatCanceller = Callable[[str], bool]


@dataclass(frozen=True, slots=True)
class _ReminderContext:
    channel: str = ""
    chat_id: str = ""
    tenant_id: str = "default"
    principal_id: str = ""
    agent_identity_id: str = ""
    sender_id: str = ""
    thread_id: str = ""
    chat_type: str = "private"
    synthetic: bool = False
    source: str = ""


class ReminderTool(Tool):
    """Tool to create/list/remove reminders and scheduled tasks."""
    
    def __init__(self, cron_service: CronService):
        self._cron = cron_service
        self._context: ContextVar[_ReminderContext] = ContextVar(
            "reminder_tool_context",
            default=_ReminderContext(),
        )
        # Task-scoped heartbeat callbacks (set by AgentLoop per processing cycle)
        self._heartbeat_factory: HeartbeatFactory | None = None
        self._heartbeat_canceller: HeartbeatCanceller | None = None

    def set_heartbeat_callbacks(
        self,
        factory: HeartbeatFactory,
        canceller: HeartbeatCanceller,
    ) -> None:
        """Set task-scoped heartbeat callbacks (called by AgentLoop)."""
        self._heartbeat_factory = factory
        self._heartbeat_canceller = canceller

    def clear_heartbeat_callbacks(self) -> None:
        """Clear heartbeat callbacks after task processing ends."""
        self._heartbeat_factory = None
        self._heartbeat_canceller = None
    
    def set_context(
        self,
        channel: str,
        chat_id: str,
        *,
        tenant_id: str = "default",
        principal_id: str = "",
        agent_identity_id: str = "",
        sender_id: str = "",
        thread_id: str = "",
        chat_type: str = "private",
        synthetic: bool = False,
        source: str = "",
    ) -> None:
        """Set the current session + identity context for delivery/isolation."""
        self._context.set(
            _ReminderContext(
                channel=channel,
                chat_id=chat_id,
                tenant_id=tenant_id or "default",
                principal_id=principal_id or "",
                agent_identity_id=agent_identity_id or "",
                sender_id=sender_id or "",
                thread_id=thread_id or "",
                chat_type=chat_type or "private",
                synthetic=synthetic,
                source=source or "",
            )
        )
    
    @property
    def name(self) -> str:
        return "reminders"
    
    @property
    def description(self) -> str:
        return (
            "Manage reminders and timed tasks. Actions: add, list, remove. "
            "For add: use every_seconds (recurring), cron_expr (calendar schedule), "
            "or at (one-time ISO datetime). "
            "Use mode='direct' to send message text as-is (default), "
            "or mode='agent' to run an agent turn each time. "
            "For finite loops, set max_runs (e.g. every_seconds=1, max_runs=10). "
            "IMPORTANT: if the user gives a clear scheduling instruction, create it immediately "
            "and report the created job id; do NOT ask for extra confirmation like 'reply start'. "
            "The reminder runs in the current chat context and starts automatically after creation. "
            "\n\n"
            "**Task-scoped heartbeat**: When you are about to execute a long-running tool "
            "(e.g. backtest, large data fetch), set task_scoped=true with delay_seconds to "
            "schedule a progress message. It fires once after delay_seconds if the task is "
            "still running, then auto-cancels when the current task completes. "
            "Use this to keep the user informed during long operations — decide the message "
            "and timing yourself based on estimated duration."
        )
    
    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["add", "list", "remove", "delete"],
                    "description": "Action to perform. Defaults to add.",
                    "default": "add",
                },
                "message": {
                    "type": "string",
                    "description": "Reminder/task prompt (for add)",
                },
                "every_seconds": {
                    "type": "integer",
                    "description": "Interval in seconds (for recurring tasks)",
                    "minimum": 1,
                },
                "cron_expr": {
                    "type": "string",
                    "description": "Cron expression like '0 9 * * *' (for scheduled tasks)",
                },
                "at": {
                    "type": "string",
                    "description": "ISO datetime for one-time execution (e.g. '2026-02-12T10:30:00')",
                },
                "job_id": {
                    "type": "string",
                    "description": "Job ID (for remove)",
                },
                "max_runs": {
                    "type": "integer",
                    "description": "Optional max number of runs for recurring reminders",
                    "minimum": 1,
                },
                "final_message": {
                    "type": "string",
                    "description": "Optional completion message context",
                },
                "mode": {
                    "type": "string",
                    "enum": ["agent", "direct"],
                    "description": "agent (default): run a full LLM turn each trigger to generate content; "
                    "direct: send the message text verbatim without LLM processing "
                    "(only for simple static reminders like '该喝水了')",
                    "default": "agent",
                },
                "task_scoped": {
                    "type": "boolean",
                    "description": (
                        "If true, creates a lightweight heartbeat that auto-cancels "
                        "when the current task finishes. Use with delay_seconds. "
                        "For keeping users informed during long-running operations."
                    ),
                    "default": False,
                },
                "delay_seconds": {
                    "type": "integer",
                    "description": (
                        "Seconds to wait before sending the heartbeat message. "
                        "Only used with task_scoped=true."
                    ),
                    "minimum": 5,
                },
            },
            "required": [],
        }
    
    async def execute(
        self,
        action: str = "add",
        message: str = "",
        every_seconds: int | None = None,
        cron_expr: str | None = None,
        at: str | None = None,
        job_id: str | None = None,
        max_runs: int | None = None,
        final_message: str | None = None,
        mode: str = "direct",
        task_scoped: bool = False,
        delay_seconds: int | None = None,
        **kwargs: Any
    ) -> str:
        inferred_action = (action or "").strip().lower()
        if not inferred_action:
            if job_id and not any([every_seconds, cron_expr, at, message]):
                inferred_action = "remove"
            else:
                inferred_action = "add"

        # ── Task-scoped heartbeat (lightweight, auto-cancelled) ──
        if task_scoped and inferred_action == "add":
            return self._add_heartbeat(message, delay_seconds)
        if inferred_action in {"remove", "delete"} and job_id and job_id.startswith("hb_"):
            return self._remove_heartbeat(job_id)

        if inferred_action == "add":
            return self._add_job(message, every_seconds, cron_expr, at, max_runs, final_message, mode)
        elif inferred_action == "list":
            return self._list_jobs()
        elif inferred_action in {"remove", "delete"}:
            return self._remove_job(job_id)
        return f"Error: unknown action '{inferred_action}'"

    # ── Task-scoped heartbeat helpers ──

    def _add_heartbeat(self, message: str, delay_seconds: int | None) -> str:
        """Schedule a task-scoped heartbeat via the agent loop callback."""
        if not message:
            return "Error: message is required for heartbeat"
        if not self._heartbeat_factory:
            return "Error: task-scoped heartbeats not available in this context"
        delay = delay_seconds or 30
        try:
            hb_id = self._heartbeat_factory(delay, message)
            logger.info(f"Task heartbeat scheduled: {hb_id} in {delay}s")
            return (
                f"Heartbeat scheduled (id: {hb_id}): will send \"{message}\" "
                f"in {delay}s if the task is still running. "
                f"Auto-cancels when the current task completes."
            )
        except Exception as e:
            logger.error(f"Failed to schedule heartbeat: {e}")
            return f"Error scheduling heartbeat: {e}"

    def _remove_heartbeat(self, hb_id: str) -> str:
        """Cancel a task-scoped heartbeat."""
        if not self._heartbeat_canceller:
            return f"Error: heartbeat cancellation not available"
        if self._heartbeat_canceller(hb_id):
            return f"Heartbeat {hb_id} cancelled."
        return f"Heartbeat {hb_id} not found or already fired."
    
    def _add_job(
        self,
        message: str,
        every_seconds: int | None,
        cron_expr: str | None,
        at: str | None,
        max_runs: int | None,
        final_message: str | None,
        mode: str,
    ) -> str:
        if not message:
            return "Error: message is required for add"
        context = self._context.get()
        if context.synthetic and context.source == "cron":
            return (
                "Error: reminder creation is blocked inside a cron-triggered run "
                "to prevent recursive scheduling loops."
            )
        if not context.channel or not context.chat_id:
            return "Error: no session context (channel/chat_id)"
        if not context.principal_id:
            return "Error: no principal identity context for reminder creation"
        
        # Build schedule
        delete_after = False
        if every_seconds:
            schedule = CronSchedule(kind="every", every_ms=every_seconds * 1000)
        elif cron_expr:
            schedule = CronSchedule(kind="cron", expr=cron_expr)
        elif at:
            from datetime import datetime
            try:
                dt = datetime.fromisoformat(at)
            except ValueError:
                return f"Error: invalid 'at' datetime '{at}', expected ISO format"
            at_ms = int(dt.timestamp() * 1000)
            schedule = CronSchedule(kind="at", at_ms=at_ms)
            delete_after = True
        else:
            return "Error: either every_seconds, cron_expr, or at is required"
        
        job = self._cron.add_job(
            name=message[:30],
            schedule=schedule,
            message=message,
            deliver=True,
            payload_kind="direct_message" if mode != "agent" else "agent_turn",
            channel=context.channel,
            to=context.chat_id,
            delete_after_run=delete_after,
            max_runs=max_runs,
            final_message=final_message,
            tenant_id=context.tenant_id,
            principal_id=context.principal_id,
            agent_identity_id=context.agent_identity_id,
            sender_id=context.sender_id,
            thread_id=context.thread_id,
            chat_type=context.chat_type,
        )
        if job.max_runs:
            return (
                f"Created reminder '{job.name}' (job_id: {job.id}, max_runs: {job.max_runs}). "
                "It is active immediately; no additional user confirmation is needed."
            )
        return (
            f"Created reminder '{job.name}' (job_id: {job.id}). "
            "It is active immediately; no additional user confirmation is needed."
        )
    
    def _list_jobs(self) -> str:
        if not self._context.get().principal_id:
            return "Error: no principal identity context for reminder listing"
        jobs = [
            j for j in self._cron.list_jobs()
            if self._is_same_scope(j)
        ]
        if not jobs:
            return "No active reminders."
        lines = [f"- {j.name} (id: {j.id}, {j.schedule.kind})" for j in jobs]
        return "Active reminders:\n" + "\n".join(lines)
    
    def _remove_job(self, job_id: str | None) -> str:
        if not job_id:
            return "Error: job_id is required for remove"
        if not self._context.get().principal_id:
            return "Error: no principal identity context for reminder removal"
        target = next(
            (j for j in self._cron.list_jobs(include_disabled=True) if j.id == job_id),
            None,
        )
        if target is None:
            return f"Reminder {job_id} not found"
        if not self._is_same_scope(target):
            return f"Reminder {job_id} not found"
        if self._cron.remove_job(job_id):
            return f"Removed reminder {job_id}"
        return f"Reminder {job_id} not found"

    def _is_same_scope(self, job: Any) -> bool:
        """Strict reminder isolation: tenant + principal + conversation scope."""
        context = self._context.get()
        payload = job.payload
        return (
            (payload.tenant_id or "default") == (context.tenant_id or "default")
            and (payload.principal_id or "") == (context.principal_id or "")
            and (payload.channel or "") == (context.channel or "")
            and (payload.to or "") == (context.chat_id or "")
            and (payload.thread_id or "") == (context.thread_id or "")
            and (payload.chat_type or "private") == (context.chat_type or "private")
        )


# Backward-compatible alias for existing imports.
CronTool = ReminderTool
