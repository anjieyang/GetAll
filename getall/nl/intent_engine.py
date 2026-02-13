"""NL-first intent detection — fully delegated to the LLM.

This module does NOT use hardcoded keywords or regex patterns.
All semantic understanding is handled by the LLM through tool calling
and system prompt context. The IntentEngine is kept as a transparent
pass-through for compatibility with the routing layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class Intent:
    """Represents a detected intent. In the AI-native architecture,
    the LLM determines intent implicitly by choosing which tool to call.
    This dataclass is used only for internal routing metadata."""

    name: str
    confidence: float
    evidence: list[str] = field(default_factory=list)
    slots: dict[str, str] = field(default_factory=dict)


class IntentEngine:
    """Transparent pass-through — no keyword heuristics.

    The LLM decides intent by reading the system prompt (SOUL.md + AGENTS.md)
    and selecting appropriate tools. This class exists only so that
    downstream code that references IntentEngine does not break.
    All messages are classified as 'chat.general' and handed to the LLM.
    """

    def detect(self, text: str) -> Intent:
        """Always returns chat.general — the LLM handles real understanding."""
        return Intent(
            name="chat.general",
            confidence=1.0,
            evidence=["llm_delegated"],
            slots={"raw_text": text},
        )
