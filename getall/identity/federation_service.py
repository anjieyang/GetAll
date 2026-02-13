"""Federation service for IFT issuance, binding, and resolution."""

from __future__ import annotations

import secrets
import uuid
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from getall.storage.models import Principal
from getall.storage.repository import IdentityRepo


# ── types ───────────────────────────────────────────────────────────────────


class IdentityError(RuntimeError):
    """Raised when identity operations fail (e.g. IFT not found)."""


@dataclass(frozen=True, slots=True)
class PrincipalRef:
    """Immutable reference to a principal (identity holder)."""

    principal_id: str
    ift: str
    agent_identity_id: str


def _principal_to_ref(p: Principal) -> PrincipalRef:
    return PrincipalRef(principal_id=p.id, ift=p.ift, agent_identity_id=p.agent_identity_id)


def _generate_ift() -> str:
    """Generate a unique IFT token: IFT- + 20 hex chars uppercase."""
    return "IFT-" + secrets.token_hex(10).upper()


# ── service ─────────────────────────────────────────────────────────────────


class FederationService:
    """Service for identity federation: IFT issuance, binding, resolution."""

    def __init__(self, session: AsyncSession) -> None:
        self._repo = IdentityRepo(session)

    async def issue_ift(self) -> PrincipalRef:
        """Generate a unique IFT, create principal, return PrincipalRef."""
        ift = _generate_ift()
        agent_identity_id = str(uuid.uuid4())
        p = await self._repo.create_principal(ift=ift, agent_identity_id=agent_identity_id)
        return _principal_to_ref(p)

    async def get_by_ift(self, ift: str) -> PrincipalRef | None:
        """Look up principal by IFT. Returns None if not found."""
        p = await self._repo.get_by_ift(ift)
        return _principal_to_ref(p) if p is not None else None

    async def resolve_from_platform(self, platform: str, platform_user_id: str) -> PrincipalRef | None:
        """Resolve principal from platform + platform_user_id. Returns None if not bound."""
        p = await self._repo.get_by_external(platform, platform_user_id)
        return _principal_to_ref(p) if p is not None else None

    async def bind_ift_to_platform(self, ift: str, platform: str, platform_user_id: str) -> PrincipalRef:
        """Look up principal by IFT, bind external account, log the bind. Raises IdentityError if IFT not found."""
        p = await self._repo.get_by_ift(ift)
        if p is None:
            raise IdentityError(f"IFT not found: {ift}")
        await self._repo.bind_external(platform, platform_user_id, p.id)
        await self._repo.log_bind(platform, platform_user_id, ift, result="bound", detail="")
        return _principal_to_ref(p)
