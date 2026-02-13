"""ReminderService — pure data persistence layer.

Schedule parsing is NOT done here. The LLM decides the schedule type,
value, and next_fire_at by understanding the user's natural language,
then passes structured parameters through tool calling.

This service only persists and queries reminders.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from croniter import croniter
from sqlalchemy.ext.asyncio import AsyncSession

from getall.storage.models import (
    GovernanceActionMode,
    Reminder,
    ReminderScheduleType,
    ReminderStatus,
)
from getall.storage.repository import ReminderRepo


def _utcnow() -> datetime:
    return datetime.now(tz=UTC)


class ReminderService:
    """Persistence layer for reminders. No NL parsing — the LLM does that."""

    def __init__(self, session: AsyncSession) -> None:
        self._repo = ReminderRepo(session)

    async def create(
        self,
        tenant_id: str,
        principal_id: str,
        title: str,
        natural_language_rule: str,
        schedule_type: ReminderScheduleType,
        schedule_value: str,
        action_mode: GovernanceActionMode,
        next_fire_at: datetime | None = None,
    ) -> Reminder:
        """Create a reminder with LLM-determined schedule parameters.

        The LLM has already interpreted the user's natural language and
        decided the schedule_type, schedule_value, and next_fire_at.
        This method just persists.
        """
        reminder = Reminder(
            tenant_id=tenant_id or "default",
            principal_id=principal_id,
            title=title,
            natural_language_rule=natural_language_rule,
            schedule_type=schedule_type,
            schedule_value=schedule_value,
            action_mode=action_mode,
            status=ReminderStatus.ACTIVE,
            next_fire_at=next_fire_at,
        )
        return await self._repo.create(reminder)

    async def list_due(self, now: datetime | None = None) -> list[Reminder]:
        """List reminders that are due for firing."""
        t = now if now is not None else _utcnow()
        return await self._repo.list_due(t)

    async def mark_fired(
        self, reminder: Reminder, now: datetime | None = None
    ) -> Reminder:
        """Update fired state and compute next_fire_at for recurring types."""
        t = now if now is not None else _utcnow()
        reminder.last_fired_at = t

        if reminder.schedule_type == ReminderScheduleType.ONCE:
            reminder.status = ReminderStatus.FIRED
            reminder.next_fire_at = None
        elif reminder.schedule_type == ReminderScheduleType.CRON:
            try:
                c = croniter(reminder.schedule_value, t)
                next_ts = c.get_next()
                reminder.next_fire_at = datetime.fromtimestamp(next_ts, tz=UTC)
            except Exception:
                reminder.status = ReminderStatus.FIRED
                reminder.next_fire_at = None
        elif reminder.schedule_type == ReminderScheduleType.INTERVAL:
            # schedule_value is an ISO 8601 duration or seconds count
            # provided by the LLM (e.g. "3600" for 1 hour)
            try:
                seconds = int(reminder.schedule_value)
                reminder.next_fire_at = t + timedelta(seconds=seconds)
            except (ValueError, TypeError):
                reminder.status = ReminderStatus.FIRED
                reminder.next_fire_at = None
        # TRIGGER: next_fire_at stays None, external event sets it

        await self._repo.flush()
        return reminder
