"""ExperimentService — pure persistence for evolution charters and events.

Mutation proposal and evaluation are NOT done here.
The LLM generates hypotheses, proposes mutations, and evaluates candidates
based on the user/agent's natural language charter. This service only
persists the data.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from getall.storage.models import EvolutionCharter, EvolutionEvent
from getall.storage.repository import EvolutionRepo


@dataclass
class MutationCandidate:
    """LLM-generated mutation candidate. All fields are free-form text
    produced by the LLM — no hardcoded logic generates these."""

    policy_version: str
    skill_version: str
    hypothesis: str
    mutation: str


class ExperimentService:
    """Persistence layer for evolution experiments. No stub logic —
    the LLM drives mutation proposal and evaluation."""

    def __init__(self, session: AsyncSession) -> None:
        self._repo = EvolutionRepo(session)

    async def ensure_charter(
        self, principal_id: str, charter_type: str, text: str
    ) -> EvolutionCharter:
        """Get or create an active charter for the principal."""
        existing = await self._repo.get_active_charter(principal_id, charter_type)
        if existing is not None:
            return existing

        charter = EvolutionCharter(
            principal_id=principal_id,
            charter_type=charter_type,
            version=1,
            natural_language_text=text,
            active=True,
        )
        return await self._repo.create_charter(charter)

    async def append_event(
        self,
        principal_id: str,
        phase: str,
        candidate: MutationCandidate,
        evaluation_report: str,
        result: str,
    ) -> EvolutionEvent:
        """Persist an evolution event. All content is LLM-generated."""
        ev = EvolutionEvent(
            principal_id=principal_id,
            policy_version=candidate.policy_version,
            skill_version=candidate.skill_version,
            phase=phase,
            hypothesis=candidate.hypothesis,
            mutation=candidate.mutation,
            evaluation_report=evaluation_report,
            result=result,
        )
        return await self._repo.append_event(ev)
