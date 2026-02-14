"""Reminders tool backed by cron scheduling."""

from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from getall.agent.tools.base import Tool
from getall.cron.service import CronService
from getall.cron.types import CronSchedule


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
            "The reminder runs in the current chat context and starts automatically after creation."
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
                    "enum": ["direct", "agent"],
                    "description": "direct: send text as-is; agent: run an LLM turn each trigger",
                    "default": "direct",
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
        **kwargs: Any
    ) -> str:
        inferred_action = (action or "").strip().lower()
        if not inferred_action:
            if job_id and not any([every_seconds, cron_expr, at, message]):
                inferred_action = "remove"
            else:
                inferred_action = "add"

        if inferred_action == "add":
            return self._add_job(message, every_seconds, cron_expr, at, max_runs, final_message, mode)
        elif inferred_action == "list":
            return self._list_jobs()
        elif inferred_action in {"remove", "delete"}:
            return self._remove_job(job_id)
        return f"Error: unknown action '{inferred_action}'"
    
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
