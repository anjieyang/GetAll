"""Reminders API – create and list due reminders.

The LLM determines schedule_type, schedule_value, and next_fire_at
from the user's natural language. This API just persists.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from getall.reminders.service import ReminderService
from getall.storage.database import get_session
from getall.storage.models import GovernanceActionMode, ReminderScheduleType

router = APIRouter()

SessionDep = Annotated[AsyncSession, Depends(get_session)]


class CreateReminderRequest(BaseModel):
    tenant_id: str = "default"
    principal_id: str
    title: str
    natural_language_rule: str
    schedule_type: str  # once / interval / cron / trigger
    schedule_value: str  # seconds for interval, cron expr for cron, etc.
    action_mode: str = "act_then_report"
    next_fire_at: str | None = None  # ISO 8601 datetime


class ReminderOut(BaseModel):
    id: str
    title: str
    schedule_type: str
    schedule_value: str
    action_mode: str
    status: str
    next_fire_at: str | None


@router.post("/create", response_model=ReminderOut)
async def create_reminder(body: CreateReminderRequest, session: SessionDep) -> ReminderOut:
    svc = ReminderService(session)
    from datetime import datetime, UTC

    next_fire = None
    if body.next_fire_at:
        next_fire = datetime.fromisoformat(body.next_fire_at)

    r = await svc.create(
        tenant_id=body.tenant_id,
        principal_id=body.principal_id,
        title=body.title,
        natural_language_rule=body.natural_language_rule,
        schedule_type=ReminderScheduleType(body.schedule_type),
        schedule_value=body.schedule_value,
        action_mode=GovernanceActionMode(body.action_mode),
        next_fire_at=next_fire,
    )
    return ReminderOut(
        id=r.id,
        title=r.title,
        schedule_type=r.schedule_type.value,
        schedule_value=r.schedule_value,
        action_mode=r.action_mode.value,
        status=r.status.value,
        next_fire_at=r.next_fire_at.isoformat() if r.next_fire_at else None,
    )


@router.get("/due", response_model=list[ReminderOut])
async def list_due(session: SessionDep) -> list[ReminderOut]:
    svc = ReminderService(session)
    due = await svc.list_due()
    return [
        ReminderOut(
            id=r.id,
            title=r.title,
            schedule_type=r.schedule_type.value,
            schedule_value=r.schedule_value,
            action_mode=r.action_mode.value,
            status=r.status.value,
            next_fire_at=r.next_fire_at.isoformat() if r.next_fire_at else None,
        )
        for r in due
    ]
