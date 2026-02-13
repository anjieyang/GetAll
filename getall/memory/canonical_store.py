"""CanonicalStore – wrapper around MemoryRepo for writing and listing canonical NL memories."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy.ext.asyncio import AsyncSession

from getall.storage.models import CanonicalMemory, TradeEvent
from getall.storage.repository import MemoryRepo


@dataclass
class CanonicalMemoryInput:
    """Input payload for creating a canonical memory."""

    tenant_id: str
    principal_id: str
    memory_type: str
    summary: str
    narrative: str
    source_channel: str = ""
    source_message_id: str = ""
    evidence: dict | None = None
    occurred_at: datetime | None = None

    def __post_init__(self) -> None:
        if self.evidence is None:
            self.evidence = {}
        if self.occurred_at is None:
            self.occurred_at = datetime.now(tz=UTC)


class CanonicalStore:
    """Store for canonical NL memories backed by MemoryRepo."""

    def __init__(self, session: AsyncSession) -> None:
        self._repo = MemoryRepo(session)

    async def append(self, payload: CanonicalMemoryInput) -> CanonicalMemory:
        """Create and persist a canonical memory from the given payload."""
        mem = CanonicalMemory(
            tenant_id=payload.tenant_id,
            principal_id=payload.principal_id,
            memory_type=payload.memory_type,
            summary=payload.summary,
            narrative=payload.narrative,
            source_channel=payload.source_channel,
            source_message_id=payload.source_message_id,
            evidence=payload.evidence or {},
            occurred_at=payload.occurred_at or datetime.now(tz=UTC),
        )
        return await self._repo.append_canonical(mem)

    async def append_trade_event(
        self,
        tenant_id: str,
        principal_id: str,
        action: str,
        symbol: str,
        strategy_id: str = "",
        reason: str = "",
        execution_result: dict | None = None,
        pnl: str = "",
    ) -> TradeEvent:
        """Create and persist a trade event."""
        ev = TradeEvent(
            tenant_id=tenant_id,
            principal_id=principal_id,
            action=action,
            symbol=symbol,
            strategy_id=strategy_id,
            reason=reason,
            execution_result=execution_result or {},
            pnl=pnl,
        )
        return await self._repo.append_trade_event(ev)

    async def list_recent(
        self,
        tenant_id: str,
        principal_id: str,
        memory_type: str | None = None,
        limit: int = 30,
    ) -> list[CanonicalMemory]:
        """List recent canonical memories for the given tenant and principal."""
        return await self._repo.list_recent(
            tenant_id=tenant_id,
            principal_id=principal_id,
            memory_type=memory_type,
            limit=limit,
        )
