"""Append-only audit service for all critical operations."""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from getall.storage.models import AuditEvent
from getall.storage.repository import AuditRepo


@dataclass(frozen=True, slots=True)
class AuditCtx:
    tenant_id: str
    principal_id: str
    session_key: str


class AuditService:
    def __init__(self, session: AsyncSession) -> None:
        self._repo = AuditRepo(session)

    async def emit(
        self,
        ctx: AuditCtx,
        event_type: str,
        event_name: str,
        payload: dict[str, object],
        severity: str = "info",
    ) -> AuditEvent:
        ev = AuditEvent(
            tenant_id=ctx.tenant_id,
            principal_id=ctx.principal_id,
            session_key=ctx.session_key,
            event_type=event_type,
            event_name=event_name,
            severity=severity,
            payload=payload,
        )
        return await self._repo.append(ev)
