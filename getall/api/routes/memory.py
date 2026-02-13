"""Memory API – recall memories with evidence."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from getall.memory.recall_orchestrator import RecallOrchestrator
from getall.storage.database import get_session

router = APIRouter()

SessionDep = Annotated[AsyncSession, Depends(get_session)]


class RecallRequest(BaseModel):
    tenant_id: str = "default"
    principal_id: str
    query: str
    strategy_id: str | None = None
    symbol: str | None = None
    time_window_days: int = 90
    limit: int = 6


class EvidenceItem(BaseModel):
    memory_id: str
    summary: str
    narrative: str
    occurred_at: str
    score: float
    evidence_reason: str


@router.post("/recall", response_model=list[EvidenceItem])
async def recall_memories(body: RecallRequest, session: SessionDep) -> list[EvidenceItem]:
    orch = RecallOrchestrator(session)
    results = await orch.recall(
        tenant_id=body.tenant_id,
        principal_id=body.principal_id,
        query=body.query,
        strategy_id=body.strategy_id,
        symbol=body.symbol,
        time_window_days=body.time_window_days,
        limit=body.limit,
    )
    return [
        EvidenceItem(
            memory_id=r.memory_id,
            summary=r.summary,
            narrative=r.narrative,
            occurred_at=r.occurred_at.isoformat(),
            score=r.score,
            evidence_reason=r.evidence_reason,
        )
        for r in results
    ]
