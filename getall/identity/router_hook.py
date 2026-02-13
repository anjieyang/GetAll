"""Router hook for identity resolution from IFT or platform."""

from __future__ import annotations

import re
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from getall.identity.federation_service import FederationService

IFT_PATTERN = re.compile(r"\bIFT[-_][A-Z0-9]{6,64}\b", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class RouteResolution:
    """Immutable result of route resolution."""

    tenant_id: str
    principal_id: str
    ift: str
    agent_identity_id: str
    channel: str
    chat_id: str | None
    user_id: str | None
    thread_id: str | None
    session_key: str


class IdentityRouterHook:
    """Hook for resolving identity from IFT in message text or platform context."""

    def __init__(self, session: AsyncSession) -> None:
        self._federation = FederationService(session)

    @staticmethod
    def extract_ift_from_text(text: str) -> str | None:
        """Extract IFT from natural language text using IFT_PATTERN regex."""
        if not text:
            return None
        m = IFT_PATTERN.search(text)
        return m.group(0).upper().replace("_", "-") if m else None

    async def resolve(
        self,
        tenant_id: str,
        channel: str,
        chat_id: str | None,
        user_id: str | None,
        thread_id: str | None,
        message_text: str,
        explicit_ift: str | None = None,
    ) -> RouteResolution:
        """Resolve identity: IFT from message → bind; else platform; else issue new."""
        platform = channel
        platform_user_id = user_id or "-"

        raw_ift = explicit_ift or self.extract_ift_from_text(message_text)
        ift: str | None = raw_ift.upper().replace("_", "-") if raw_ift else None

        if ift:
            ref = await self._federation.bind_ift_to_platform(ift, platform, platform_user_id)
        else:
            ref = await self._federation.resolve_from_platform(platform, platform_user_id)

        if ref is None:
            ref = await self._federation.issue_ift()
            ref = await self._federation.bind_ift_to_platform(ref.ift, platform, platform_user_id)

        session_key = "/".join([
            tenant_id,
            ref.principal_id,
            channel,
            chat_id or "-",
            user_id or "-",
            thread_id or "-",
        ])

        return RouteResolution(
            tenant_id=tenant_id,
            principal_id=ref.principal_id,
            ift=ref.ift,
            agent_identity_id=ref.agent_identity_id,
            channel=channel,
            chat_id=chat_id,
            user_id=user_id,
            thread_id=thread_id,
            session_key=session_key,
        )
