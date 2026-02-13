"""DerivedIndex – builds and queries the derived recall layer."""

from __future__ import annotations

import re
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from getall.storage.models import CanonicalMemory, DerivedRecallIndex
from getall.storage.repository import MemoryRepo


def _tokenize(text: str) -> set[str]:
    """Tokenize text: lowercase, split on non-alphanumeric, filter tokens >= 2 chars."""
    if not text:
        return set()
    tokens = re.split(r"[^a-zA-Z0-9]+", text.lower())
    return {t for t in tokens if len(t) >= 2}


@dataclass
class RecallCandidate:
    """A candidate memory from the derived index with score and reason."""

    memory_id: str
    score: float
    reason: str


class DerivedIndex:
    """Index for derived recall – keywords, entities, time buckets."""

    def __init__(self, session: AsyncSession) -> None:
        self._repo = MemoryRepo(session)
        self._session = session

    def _extract_entities(self, text: str) -> set[str]:
        """Extract entity-like tokens (uppercase-capitalized or all-caps words)."""
        if not text:
            return set()
        tokens = re.split(r"[^a-zA-Z0-9]+", text)
        return {t for t in tokens if len(t) >= 2 and t[0].isupper()}

    async def upsert_from_memory(self, memory: CanonicalMemory) -> DerivedRecallIndex:
        """Extract keywords, entities, time_bucket from memory and persist index row."""
        combined = f"{memory.summary} {memory.narrative}"
        keywords = _tokenize(combined)
        entities = self._extract_entities(combined)
        time_bucket = memory.occurred_at.strftime("%Y-%m") if memory.occurred_at else ""

        keywords_str = " ".join(sorted(keywords))
        entities_str = " ".join(sorted(entities))

        stmt = select(DerivedRecallIndex).where(DerivedRecallIndex.memory_id == memory.id)
        existing = await self._session.scalar(stmt)
        if existing is not None:
            existing.keywords = keywords_str
            existing.entities = entities_str
            existing.time_bucket = time_bucket
            await self._session.flush()
            return existing

        idx = DerivedRecallIndex(
            memory_id=memory.id,
            tenant_id=memory.tenant_id,
            principal_id=memory.principal_id,
            keywords=keywords_str,
            entities=entities_str,
            time_bucket=time_bucket,
        )
        return await self._repo.append_index(idx)

    async def query(
        self,
        tenant_id: str,
        principal_id: str,
        query_text: str,
        limit: int = 8,
    ) -> list[RecallCandidate]:
        """Fetch recent index rows, score by token overlap + recency bonus, return top candidates."""
        index_rows = await self._repo.list_recent_index(
            tenant_id=tenant_id,
            principal_id=principal_id,
            limit=limit * 4,
        )
        if not index_rows:
            return []

        query_tokens = _tokenize(query_text)
        query_entities = self._extract_entities(query_text)

        def _score(row: DerivedRecallIndex, rank: int) -> tuple[float, str]:
            overlap = 0.0
            reasons: list[str] = []

            row_keywords = set(row.keywords.split()) if row.keywords else set()
            row_entities = set(row.entities.split()) if row.entities else set()

            keyword_overlap = len(query_tokens & row_keywords)
            entity_overlap = len(query_entities & row_entities)

            overlap = keyword_overlap * 1.0 + entity_overlap * 2.0
            if keyword_overlap:
                reasons.append(f"keywords:{keyword_overlap}")
            if entity_overlap:
                reasons.append(f"entities:{entity_overlap}")

            recency_bonus = max(0, 1.0 - rank * 0.05)
            score = overlap + recency_bonus
            reason = "+".join(reasons) if reasons else "recency"
            return (score, reason)

        scored: list[tuple[RecallCandidate, float]] = []
        for i, row in enumerate(index_rows):
            score_val, reason = _score(row, i)
            if score_val > 0:
                scored.append((RecallCandidate(memory_id=row.memory_id, score=score_val, reason=reason), score_val))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [c for c, _ in scored[:limit]]
