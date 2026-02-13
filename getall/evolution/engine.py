"""EvolutionEngine — pure persistence orchestrator.

The LLM drives the entire evolution pipeline:
  Observe -> Reflect -> Mutate -> Evaluate -> Rollout -> CommitOrRollback

This engine only persists charters and events. It does NOT generate
hypotheses, propose mutations, or evaluate candidates — the LLM does
all of that through tool calling and natural language reasoning.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from getall.evolution.experiment_service import ExperimentService, MutationCandidate


@dataclass
class EvolutionResult:
    """Result of persisting an evolution event. Content is LLM-generated."""

    principal_id: str
    phase: str
    event_id: str


class EvolutionEngine:
    """Persistence orchestrator for per-identity evolution events.
    No stub logic — the LLM decides everything."""

    def __init__(self, session: AsyncSession) -> None:
        self._exp = ExperimentService(session)

    async def ensure_charter(
        self, principal_id: str, charter_type: str, text: str
    ) -> str:
        """Ensure a charter exists; return its id."""
        charter = await self._exp.ensure_charter(principal_id, charter_type, text)
        return charter.id

    async def record_event(
        self,
        principal_id: str,
        phase: str,
        policy_version: str,
        skill_version: str,
        hypothesis: str,
        mutation: str,
        evaluation_report: str,
        result: str,
    ) -> EvolutionResult:
        """Persist a single evolution event with LLM-generated content."""
        candidate = MutationCandidate(
            policy_version=policy_version,
            skill_version=skill_version,
            hypothesis=hypothesis,
            mutation=mutation,
        )
        ev = await self._exp.append_event(
            principal_id=principal_id,
            phase=phase,
            candidate=candidate,
            evaluation_report=evaluation_report,
            result=result,
        )
        return EvolutionResult(
            principal_id=principal_id,
            phase=phase,
            event_id=ev.id,
        )
