"""Pet persona tool — lets the LLM set or update the pet's identity.

Called during onboarding (new user) or anytime the persona should evolve
(user request, self-reflection, strategy shift). The LLM decides all
content — this tool only persists.
"""

from __future__ import annotations

from typing import Any

from getall.agent.tools.base import Tool


class PetPersonaTool(Tool):
    """Set or update the pet's name, personality, and trading style for this user."""

    name = "pet_persona"
    description = (
        "Set or update your (the pet's) name, personality, and trading style "
        "for this user. Call this during onboarding to establish who you are, "
        "or anytime the user asks you to change, or when you decide to evolve. "
        "Only the fields you provide will be updated; omit fields to keep them unchanged. "
        "Set onboarded=true once the user has finished initial setup."
    )
    parameters = {
        "type": "object",
        "properties": {
            "pet_name": {
                "type": "string",
                "description": "The pet's display name (e.g. Rocky, Mochi, Alpha)",
            },
            "persona_text": {
                "type": "string",
                "description": (
                    "Free-form personality description in natural language. "
                    "e.g. 'Aggressive and confident, speaks in short punchy sentences, "
                    "loves high-volatility meme coins, uses trader slang'"
                ),
            },
            "trading_style_text": {
                "type": "string",
                "description": (
                    "Free-form trading strategy/style in natural language. "
                    "e.g. 'Swing trader on 4H timeframe, prefers BTC and ETH, "
                    "max 3x leverage, always sets stop-loss at 2%'"
                ),
            },
            "onboarded": {
                "type": "boolean",
                "description": "Set to true once the user has completed initial onboarding",
            },
        },
        "required": [],
    }

    def __init__(self) -> None:
        self._principal_id: str = ""
        self._session_factory: Any = None

    def set_context(self, principal_id: str, session_factory: Any) -> None:
        """Called by the agent loop before each message to set the current user context."""
        self._principal_id = principal_id
        self._session_factory = session_factory

    async def execute(
        self,
        pet_name: str = "",
        persona_text: str = "",
        trading_style_text: str = "",
        onboarded: bool | None = None,
        **kw: Any,
    ) -> str:
        if not self._principal_id:
            return "Error: no principal_id in context — cannot update persona"

        if self._session_factory is None:
            return "Error: database session not available"

        from getall.storage.repository import IdentityRepo

        async with self._session_factory() as session:
            repo = IdentityRepo(session)
            p = await repo.update_persona(
                principal_id=self._principal_id,
                pet_name=pet_name or None,
                persona_text=persona_text or None,
                trading_style_text=trading_style_text or None,
                onboarded=onboarded,
            )
            await session.commit()

        if p is None:
            return "Error: principal not found"

        parts = []
        if pet_name:
            parts.append(f"name={pet_name}")
        if persona_text:
            parts.append(f"persona updated")
        if trading_style_text:
            parts.append(f"trading style updated")
        if onboarded is not None:
            parts.append(f"onboarded={onboarded}")
        return f"Persona updated: {', '.join(parts) or 'no changes'}"
