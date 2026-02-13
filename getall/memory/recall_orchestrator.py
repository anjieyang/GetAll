"""RecallOrchestrator – main recall entry point."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from sqlalchemy.ext.asyncio import AsyncSession

from getall.memory.derived_index import DerivedIndex, RecallCandidate
from getall.storage.models import CanonicalMemory
from getall.storage.repository import MemoryRepo


@dataclass
class RecalledEvidence:
    """Evidence recalled from canonical memory with score and reason."""

    memory_id: str
    summary: str
    narrative: str
    occurred_at: datetime
    score: float
    evidence_reason: str


class RecallOrchestrator:
    """Orchestrates recall from canonical memories via the derived index."""

    def __init__(self, session: AsyncSession) -> None:
        self._repo = MemoryRepo(session)
        self._index = DerivedIndex(session)

    async def recall(
        self,
        tenant_id: str,
        principal_id: str,
        query: str,
        strategy_id: str | None = None,
        symbol: str | None = None,
        time_window_days: int = 90,
        limit: int = 6,
    ) -> list[RecalledEvidence]:
        """
        Recall relevant evidence for the given query.

        1. Build composed query from query + optional strategy_id + symbol
        2. Get candidates from DerivedIndex.query (limit * 2)
        3. Get recent canonical memories
        4. Match candidates to memories, filter by time window
        5. Sort by score, return top `limit`
        """
        parts: list[str] = [query]
        if strategy_id:
            parts.append(strategy_id)
        if symbol:
            parts.append(symbol)
        composed_query = " ".join(parts)

        candidates = await self._index.query(
            tenant_id=tenant_id,
            principal_id=principal_id,
            query_text=composed_query,
            limit=limit * 2,
        )

        memories = await self._repo.list_recent(
            tenant_id=tenant_id,
            principal_id=principal_id,
            memory_type=None,
            limit=limit * 4,
        )
        memory_map: dict[str, CanonicalMemory] = {m.id: m for m in memories}

        cutoff = datetime.now(tz=UTC) - timedelta(days=time_window_days)
        results: list[RecalledEvidence] = []

        for c in candidates:
            mem = memory_map.get(c.memory_id)
            if mem is None:
                continue
            if mem.occurred_at and mem.occurred_at < cutoff:
                continue
            results.append(
                RecalledEvidence(
                    memory_id=mem.id,
                    summary=mem.summary,
                    narrative=mem.narrative,
                    occurred_at=mem.occurred_at or datetime.now(tz=UTC),
                    score=c.score,
                    evidence_reason=c.reason,
                )
            )

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:limit]
