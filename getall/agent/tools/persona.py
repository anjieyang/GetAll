"""Pet persona tool — lets the LLM set or update the pet's identity.

Called during onboarding (new user) or anytime the persona should evolve
(user request, self-reflection, strategy shift). The LLM decides all
content — this tool only persists.

Supports two modes:
- **Private chat**: persona stored in the Principal DB row (per-user).
- **Group chat**: persona stored as a JSON file at the group's memory scope
  path, independent from any user's private persona.
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from loguru import logger

from getall.agent.tools.base import Tool


@dataclass(frozen=True, slots=True)
class _PersonaContext:
    principal_id: str = ""
    session_factory: Any = None
    group_persona_path: Path | None = None


class PetPersonaTool(Tool):
    """Set or update the pet's name, personality, and trading style."""

    name = "pet_persona"
    description = (
        "Set or update your (the pet's) name, personality, and trading style. "
        "In private chats this updates your personal identity for this user. "
        "In group chats this updates the group-level personality (independent "
        "from any user's private persona). "
        "Only the fields you provide will be updated; omit fields to keep them unchanged. "
        "Set onboarded=true once the user has finished initial setup (private chat only)."
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
                "description": "Set to true once the user has completed initial onboarding (private chat only)",
            },
        },
        "required": [],
    }

    def __init__(self) -> None:
        self._context: ContextVar[_PersonaContext] = ContextVar(
            "pet_persona_tool_context",
            default=_PersonaContext(),
        )

    # ── context setters (called by agent loop per message) ──

    def clear_context(self) -> None:
        """Clear all bound context to avoid cross-chat state leakage."""
        self._context.set(_PersonaContext())

    def set_context(self, principal_id: str, session_factory: Any) -> None:
        """Private-chat context: persona lives in Principal DB row."""
        self._context.set(
            _PersonaContext(
                principal_id=principal_id,
                session_factory=session_factory,
            )
        )

    def set_group_context(self, persona_path: Path) -> None:
        """Group-chat context: persona lives in a JSON file."""
        self._context.set(_PersonaContext(group_persona_path=persona_path))

    # ── execute ──

    async def execute(
        self,
        pet_name: str = "",
        persona_text: str = "",
        trading_style_text: str = "",
        onboarded: bool | None = None,
        **kw: Any,
    ) -> str:
        context = self._context.get()
        # Group mode → file-based storage
        if context.group_persona_path is not None:
            return self._execute_group(
                context.group_persona_path,
                pet_name,
                persona_text,
                trading_style_text,
            )

        # Private mode → DB storage
        if not context.principal_id:
            return "Error: no principal_id in context — cannot update persona"

        if context.session_factory is None:
            return "Error: database session not available"

        from getall.storage.repository import IdentityRepo

        async with context.session_factory() as session:
            repo = IdentityRepo(session)
            p = await repo.update_persona(
                principal_id=context.principal_id,
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
            parts.append("persona updated")
        if trading_style_text:
            parts.append("trading style updated")
        if onboarded is not None:
            parts.append(f"onboarded={onboarded}")
        return f"Persona updated: {', '.join(parts) or 'no changes'}"

    # ── group persona (file-based) ──

    def _execute_group(
        self,
        path: Path,
        pet_name: str,
        persona_text: str,
        trading_style_text: str,
    ) -> str:
        """Update group-level persona stored as a JSON file."""

        # Load existing
        existing: dict[str, str] = {}
        if path.is_file():
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                existing = {}

        # Merge updates (only provided fields)
        if pet_name:
            existing["pet_name"] = pet_name
        if persona_text:
            existing["persona_text"] = persona_text
        if trading_style_text:
            existing["trading_style_text"] = trading_style_text

        # Persist
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(existing, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        parts = []
        if pet_name:
            parts.append(f"name={pet_name}")
        if persona_text:
            parts.append("persona updated")
        if trading_style_text:
            parts.append("trading style updated")
        logger.info(f"Group persona updated at {path}: {parts}")
        return f"Group persona updated: {', '.join(parts) or 'no changes'}"

    @staticmethod
    def load_group_persona(path: Path) -> dict[str, str] | None:
        """Load group persona from JSON file. Returns None if not found/empty."""
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and any(data.values()):
                return data
        except (json.JSONDecodeError, OSError):
            pass
        return None
