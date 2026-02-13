"""Intent projector — transparent pass-through.

In the AI-native architecture, the LLM handles all semantic projection
through tool calling. This module is a no-op shim for compatibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from getall.nl.intent_engine import Intent


@dataclass(frozen=True, slots=True)
class ProjectedAction:
    """Internal action representation. The LLM determines the real action
    by choosing tools; this is only for routing metadata."""

    action_type: str
    payload: dict[str, str] = field(default_factory=dict)


class IntentProjector:
    """Transparent pass-through — all projection delegated to LLM."""

    def project(self, intent: Intent, message_text: str) -> ProjectedAction:
        """Returns a generic action — the LLM decides the real action via tools."""
        return ProjectedAction(
            action_type="llm_delegated",
            payload={"raw_text": message_text},
        )
