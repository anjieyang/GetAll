"""Thin data-access helpers on top of SQLAlchemy async sessions.

Each repository is instantiated with a scoped AsyncSession and provides
typed CRUD for one domain aggregate.  Business logic stays in the service layer.
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession

from getall.storage.models import (
    AuditEvent,
    CanonicalMemory,
    ConversationSession,
    DerivedRecallIndex,
    EvolutionCharter,
    EvolutionEvent,
    ExternalAccount,
    IdentityBindLog,
    Principal,
    Reminder,
    ReminderStatus,
    TradeEvent,
    UserCredential,
)


# ── identity ──────────────────────────────────────────────────────────────


class IdentityRepo:
    def __init__(self, s: AsyncSession) -> None:
        self._s = s

    async def create_principal(self, ift: str, agent_identity_id: str) -> Principal:
        p = Principal(ift=ift, agent_identity_id=agent_identity_id)
        self._s.add(p)
        await self._s.flush()
        return p

    async def get_by_ift(self, ift: str) -> Principal | None:
        return await self._s.scalar(select(Principal).where(Principal.ift == ift))

    async def get_by_external(self, platform: str, uid: str) -> Principal | None:
        return await self._s.scalar(
            select(Principal)
            .join(ExternalAccount, ExternalAccount.principal_id == Principal.id)
            .where(and_(ExternalAccount.platform == platform, ExternalAccount.platform_user_id == uid))
        )

    async def bind_external(self, platform: str, uid: str, principal_id: str) -> ExternalAccount:
        stmt = select(ExternalAccount).where(
            and_(ExternalAccount.platform == platform, ExternalAccount.platform_user_id == uid)
        )
        existing = await self._s.scalar(stmt)
        if existing is not None:
            existing.principal_id = principal_id
            await self._s.flush()
            return existing
        ea = ExternalAccount(platform=platform, platform_user_id=uid, principal_id=principal_id)
        self._s.add(ea)
        await self._s.flush()
        return ea

    async def log_bind(self, platform: str, uid: str, ift: str, result: str, detail: str = "") -> IdentityBindLog:
        log = IdentityBindLog(platform=platform, platform_user_id=uid, ift=ift, result=result, detail=detail)
        self._s.add(log)
        await self._s.flush()
        return log

    async def get_by_id(self, principal_id: str) -> Principal | None:
        return await self._s.get(Principal, principal_id)

    async def update_delegation_policy(self, principal_id: str, text: str) -> None:
        p = await self._s.get(Principal, principal_id)
        if p is not None:
            p.delegation_policy_text = text
            await self._s.flush()

    async def update_persona(
        self,
        principal_id: str,
        pet_name: str | None = None,
        persona_text: str | None = None,
        trading_style_text: str | None = None,
        onboarded: bool | None = None,
    ) -> Principal | None:
        p = await self._s.get(Principal, principal_id)
        if p is None:
            return None
        if pet_name is not None:
            p.pet_name = pet_name
        if persona_text is not None:
            p.persona_text = persona_text
        if trading_style_text is not None:
            p.trading_style_text = trading_style_text
        if onboarded is not None:
            p.onboarded = onboarded
        await self._s.flush()
        return p


# ── credentials ──────────────────────────────────────────────────────────


class CredentialRepo:
    """Encrypted exchange credential storage per principal."""

    def __init__(self, s: AsyncSession) -> None:
        self._s = s

    async def save(
        self,
        principal_id: str,
        provider: str,
        api_key: str,
        api_secret: str,
        passphrase: str,
    ) -> UserCredential:
        from getall.utils.crypto import encrypt

        stmt = select(UserCredential).where(
            and_(
                UserCredential.principal_id == principal_id,
                UserCredential.provider == provider,
            )
        )
        row = await self._s.scalar(stmt)
        if row is None:
            row = UserCredential(principal_id=principal_id, provider=provider)
            self._s.add(row)
        row.encrypted_api_key = encrypt(api_key)
        row.encrypted_api_secret = encrypt(api_secret)
        row.encrypted_passphrase = encrypt(passphrase)
        await self._s.flush()
        return row

    async def get(self, principal_id: str, provider: str = "bitget") -> dict[str, str] | None:
        """Return decrypted credentials dict or None if not found."""
        from getall.utils.crypto import decrypt

        stmt = select(UserCredential).where(
            and_(
                UserCredential.principal_id == principal_id,
                UserCredential.provider == provider,
            )
        )
        row = await self._s.scalar(stmt)
        if row is None:
            return None
        return {
            "api_key": decrypt(row.encrypted_api_key),
            "api_secret": decrypt(row.encrypted_api_secret),
            "passphrase": decrypt(row.encrypted_passphrase),
        }

    async def delete(self, principal_id: str, provider: str = "bitget") -> bool:
        stmt = select(UserCredential).where(
            and_(
                UserCredential.principal_id == principal_id,
                UserCredential.provider == provider,
            )
        )
        row = await self._s.scalar(stmt)
        if row is None:
            return False
        await self._s.delete(row)
        await self._s.flush()
        return True

    async def exists(self, principal_id: str, provider: str = "bitget") -> bool:
        stmt = select(UserCredential).where(
            and_(
                UserCredential.principal_id == principal_id,
                UserCredential.provider == provider,
            )
        )
        return (await self._s.scalar(stmt)) is not None


# ── session ───────────────────────────────────────────────────────────────


class SessionRepo:
    def __init__(self, s: AsyncSession) -> None:
        self._s = s

    async def upsert(
        self,
        tenant_id: str,
        principal_id: str,
        channel: str,
        chat_id: str,
        user_id: str,
        thread_id: str,
        session_key: str,
    ) -> ConversationSession:
        stmt = select(ConversationSession).where(
            and_(
                ConversationSession.tenant_id == tenant_id,
                ConversationSession.principal_id == principal_id,
                ConversationSession.channel == channel,
                ConversationSession.chat_id == chat_id,
                ConversationSession.user_id == user_id,
                ConversationSession.thread_id == thread_id,
            )
        )
        row = await self._s.scalar(stmt)
        if row is None:
            row = ConversationSession(
                tenant_id=tenant_id, principal_id=principal_id, channel=channel,
                chat_id=chat_id, user_id=user_id, thread_id=thread_id, session_key=session_key,
            )
            self._s.add(row)
        row.last_message_at = datetime.now(tz=UTC)
        await self._s.flush()
        return row


# ── memory ────────────────────────────────────────────────────────────────


class MemoryRepo:
    def __init__(self, s: AsyncSession) -> None:
        self._s = s

    async def append_canonical(self, mem: CanonicalMemory) -> CanonicalMemory:
        self._s.add(mem)
        await self._s.flush()
        return mem

    async def append_index(self, idx: DerivedRecallIndex) -> DerivedRecallIndex:
        self._s.add(idx)
        await self._s.flush()
        return idx

    async def list_recent(
        self, tenant_id: str, principal_id: str, memory_type: str | None = None, limit: int = 30,
    ) -> list[CanonicalMemory]:
        stmt = (
            select(CanonicalMemory)
            .where(and_(CanonicalMemory.tenant_id == tenant_id, CanonicalMemory.principal_id == principal_id))
            .order_by(CanonicalMemory.occurred_at.desc())
            .limit(limit)
        )
        if memory_type:
            stmt = stmt.where(CanonicalMemory.memory_type == memory_type)
        return list(await self._s.scalars(stmt))

    async def list_recent_index(self, tenant_id: str, principal_id: str, limit: int = 200) -> list[DerivedRecallIndex]:
        stmt = (
            select(DerivedRecallIndex)
            .where(and_(DerivedRecallIndex.tenant_id == tenant_id, DerivedRecallIndex.principal_id == principal_id))
            .order_by(DerivedRecallIndex.created_at.desc())
            .limit(limit)
        )
        return list(await self._s.scalars(stmt))

    async def append_trade_event(self, ev: TradeEvent) -> TradeEvent:
        self._s.add(ev)
        await self._s.flush()
        return ev


# ── reminders ─────────────────────────────────────────────────────────────


class ReminderRepo:
    def __init__(self, s: AsyncSession) -> None:
        self._s = s

    async def create(self, r: Reminder) -> Reminder:
        self._s.add(r)
        await self._s.flush()
        return r

    async def list_due(self, now: datetime, limit: int = 100) -> list[Reminder]:
        stmt = (
            select(Reminder)
            .where(and_(
                Reminder.status == ReminderStatus.ACTIVE,
                Reminder.next_fire_at.is_not(None),
                Reminder.next_fire_at <= now,
            ))
            .order_by(Reminder.next_fire_at.asc())
            .limit(limit)
        )
        return list(await self._s.scalars(stmt))

    async def flush(self) -> None:
        await self._s.flush()


# ── evolution ─────────────────────────────────────────────────────────────


class EvolutionRepo:
    def __init__(self, s: AsyncSession) -> None:
        self._s = s

    async def create_charter(self, c: EvolutionCharter) -> EvolutionCharter:
        self._s.add(c)
        await self._s.flush()
        return c

    async def get_active_charter(self, principal_id: str, charter_type: str) -> EvolutionCharter | None:
        stmt = (
            select(EvolutionCharter)
            .where(and_(
                EvolutionCharter.principal_id == principal_id,
                EvolutionCharter.charter_type == charter_type,
                EvolutionCharter.active.is_(True),
            ))
            .order_by(EvolutionCharter.version.desc())
            .limit(1)
        )
        return await self._s.scalar(stmt)

    async def append_event(self, ev: EvolutionEvent) -> EvolutionEvent:
        self._s.add(ev)
        await self._s.flush()
        return ev


# ── audit ─────────────────────────────────────────────────────────────────


class AuditRepo:
    def __init__(self, s: AsyncSession) -> None:
        self._s = s

    async def append(self, ev: AuditEvent) -> AuditEvent:
        self._s.add(ev)
        await self._s.flush()
        return ev

    async def list_recent(self, tenant_id: str, principal_id: str, limit: int = 100) -> list[AuditEvent]:
        stmt = (
            select(AuditEvent)
            .where(and_(AuditEvent.tenant_id == tenant_id, AuditEvent.principal_id == principal_id))
            .order_by(AuditEvent.created_at.desc())
            .limit(limit)
        )
        return list(await self._s.scalars(stmt))
