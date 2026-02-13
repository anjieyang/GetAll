"""Adaptive HITL governance — fully LLM-driven.

There are NO hardcoded thresholds, NO numeric limits, NO keyword matching.

The agent decides whether to ask the user, act then report, or just report,
based entirely on:
  - The system prompt (SOUL.md / AGENTS.md delegation governance section)
  - Its memory of the user's preferences, risk tolerance, and past feedback
  - Its own judgment of the current context and risk level

This module is intentionally empty of logic. It exists only as a namespace
for the dataclasses that other parts of the system reference.
"""

from __future__ import annotations

from dataclasses import dataclass

from getall.storage.models import GovernanceActionMode


@dataclass
class RiskContext:
    """Contextual information the LLM already has access to.
    This dataclass is kept for type compatibility only — the LLM
    does not receive it as a structured input. It reads context
    from conversation history and memory."""

    pass


@dataclass
class HitlDecision:
    """The LLM's decision on how to proceed.
    This is for internal routing/audit only — the LLM expresses
    its decision through its response and tool choice."""

    mode: GovernanceActionMode
    requires_confirmation: bool
    reason: str
