"""SQLAlchemy ORM models – all tables for GetAll."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import Enum as PyEnum

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Enum as SqlEnum,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.ext.asyncio import AsyncAttrs
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _uuid() -> str:
    return str(uuid.uuid4())


def _utcnow() -> datetime:
    return datetime.now(tz=UTC)


# ---------------------------------------------------------------------------
# enums
# ---------------------------------------------------------------------------


class ReminderScheduleType(str, PyEnum):
    ONCE = "once"
    INTERVAL = "interval"
    CRON = "cron"
    TRIGGER = "trigger"


class ReminderStatus(str, PyEnum):
    ACTIVE = "active"
    PAUSED = "paused"
    FIRED = "fired"
    CANCELLED = "cancelled"


class GovernanceActionMode(str, PyEnum):
    ASK = "ask"
    ACT_THEN_REPORT = "act_then_report"
    REPORT_ONLY = "report_only"


# ---------------------------------------------------------------------------
# base
# ---------------------------------------------------------------------------


class Base(AsyncAttrs, DeclarativeBase):
    pass


# ---------------------------------------------------------------------------
# identity
# ---------------------------------------------------------------------------


class Principal(Base):
    __tablename__ = "principals"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    ift: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    agent_identity_id: Mapped[str] = mapped_column(String(36), unique=True, index=True)
    status: Mapped[str] = mapped_column(String(32), default="active")
    role: Mapped[str] = mapped_column(String(32), default="user", index=True)  # "user" | "admin"
    delegation_policy_text: Mapped[str] = mapped_column(Text, default="")

    # ── per-user pet persona (all NL, no enums) ──
    pet_name: Mapped[str] = mapped_column(Text, default="")
    persona_text: Mapped[str] = mapped_column(Text, default="")
    trading_style_text: Mapped[str] = mapped_column(Text, default="")
    onboarded: Mapped[bool] = mapped_column(Boolean, default=False)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)

    external_accounts: Mapped[list[ExternalAccount]] = relationship(back_populates="principal")
    credentials: Mapped[list[UserCredential]] = relationship(back_populates="principal")


class ExternalAccount(Base):
    __tablename__ = "external_accounts"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    platform: Mapped[str] = mapped_column(String(32), index=True)
    platform_user_id: Mapped[str] = mapped_column(String(255), index=True)
    principal_id: Mapped[str] = mapped_column(ForeignKey("principals.id"), index=True)
    bound_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    principal: Mapped[Principal] = relationship(back_populates="external_accounts")

    __table_args__ = (
        Index("uq_ext_platform_user", "platform", "platform_user_id", unique=True),
    )


class IdentityBindLog(Base):
    __tablename__ = "identity_bind_logs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    platform: Mapped[str] = mapped_column(String(32), index=True)
    platform_user_id: Mapped[str] = mapped_column(String(255))
    ift: Mapped[str] = mapped_column(String(64))
    result: Mapped[str] = mapped_column(String(32))
    detail: Mapped[str] = mapped_column(Text, default="")
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, index=True)


# ---------------------------------------------------------------------------
# user credentials (encrypted exchange API keys)
# ---------------------------------------------------------------------------


class UserCredential(Base):
    __tablename__ = "user_credentials"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    principal_id: Mapped[str] = mapped_column(ForeignKey("principals.id"), index=True)
    provider: Mapped[str] = mapped_column(String(32), default="bitget")  # exchange name
    encrypted_api_key: Mapped[str] = mapped_column(Text, default="")
    encrypted_api_secret: Mapped[str] = mapped_column(Text, default="")
    encrypted_passphrase: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)

    principal: Mapped[Principal] = relationship(back_populates="credentials")

    __table_args__ = (
        Index("uq_cred_principal_provider", "principal_id", "provider", unique=True),
    )


# ---------------------------------------------------------------------------
# session
# ---------------------------------------------------------------------------


class ConversationSession(Base):
    __tablename__ = "conversation_sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    tenant_id: Mapped[str] = mapped_column(String(64), default="default")
    principal_id: Mapped[str] = mapped_column(ForeignKey("principals.id"), index=True)
    channel: Mapped[str] = mapped_column(String(32), index=True)
    chat_id: Mapped[str] = mapped_column(String(255), default="")
    user_id: Mapped[str] = mapped_column(String(255), default="")
    thread_id: Mapped[str] = mapped_column(String(255), default="")
    session_key: Mapped[str] = mapped_column(String(512), index=True)
    last_message_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    __table_args__ = (
        Index(
            "uq_session_partition",
            "tenant_id", "principal_id", "channel", "chat_id", "user_id", "thread_id",
            unique=True,
        ),
    )


# ---------------------------------------------------------------------------
# memory (canonical NL + derived index)
# ---------------------------------------------------------------------------


class CanonicalMemory(Base):
    __tablename__ = "canonical_memory"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    tenant_id: Mapped[str] = mapped_column(String(64), default="default")
    principal_id: Mapped[str] = mapped_column(ForeignKey("principals.id"), index=True)
    memory_type: Mapped[str] = mapped_column(String(32), index=True)  # identity / user / trade / conversation / reflection
    summary: Mapped[str] = mapped_column(String(512))
    narrative: Mapped[str] = mapped_column(Text)
    evidence: Mapped[dict] = mapped_column(JSON, default=dict)
    source_channel: Mapped[str] = mapped_column(String(32), default="")
    source_message_id: Mapped[str] = mapped_column(String(255), default="")
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class DerivedRecallIndex(Base):
    __tablename__ = "derived_recall_index"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    memory_id: Mapped[str] = mapped_column(ForeignKey("canonical_memory.id"), index=True)
    tenant_id: Mapped[str] = mapped_column(String(64), default="default")
    principal_id: Mapped[str] = mapped_column(ForeignKey("principals.id"), index=True)
    keywords: Mapped[str] = mapped_column(Text, default="")
    entities: Mapped[str] = mapped_column(Text, default="")
    embedding_model: Mapped[str] = mapped_column(String(128), default="lexical-v1")
    semantic_hint: Mapped[str] = mapped_column(Text, default="")
    time_bucket: Mapped[str] = mapped_column(String(32), index=True)  # e.g. "2026-02"
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


# ---------------------------------------------------------------------------
# trade events
# ---------------------------------------------------------------------------


class TradeEvent(Base):
    __tablename__ = "trade_events"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    tenant_id: Mapped[str] = mapped_column(String(64), default="default")
    principal_id: Mapped[str] = mapped_column(ForeignKey("principals.id"), index=True)
    strategy_id: Mapped[str] = mapped_column(String(64), default="", index=True)
    symbol: Mapped[str] = mapped_column(String(64), default="", index=True)
    side: Mapped[str] = mapped_column(String(16), default="")
    qty: Mapped[str] = mapped_column(String(64), default="")
    action: Mapped[str] = mapped_column(String(64), index=True)
    reason: Mapped[str] = mapped_column(Text, default="")
    execution_result: Mapped[dict] = mapped_column(JSON, default=dict)
    pnl: Mapped[str] = mapped_column(String(64), default="")
    risk_flags: Mapped[str] = mapped_column(Text, default="")
    happened_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


# ---------------------------------------------------------------------------
# reminders
# ---------------------------------------------------------------------------


class Reminder(Base):
    __tablename__ = "reminders"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    tenant_id: Mapped[str] = mapped_column(String(64), default="default")
    principal_id: Mapped[str] = mapped_column(ForeignKey("principals.id"), index=True)
    title: Mapped[str] = mapped_column(String(255))
    natural_language_rule: Mapped[str] = mapped_column(Text, default="")
    schedule_type: Mapped[ReminderScheduleType] = mapped_column(SqlEnum(ReminderScheduleType), index=True)
    schedule_value: Mapped[str] = mapped_column(String(255), default="")
    trigger_rule: Mapped[str] = mapped_column(Text, default="")
    action_mode: Mapped[GovernanceActionMode] = mapped_column(SqlEnum(GovernanceActionMode))
    status: Mapped[ReminderStatus] = mapped_column(
        SqlEnum(ReminderStatus), default=ReminderStatus.ACTIVE, index=True,
    )
    next_fire_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True, index=True)
    last_fired_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    payload: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)


# ---------------------------------------------------------------------------
# evolution (per-identity, strategy + skill only)
# ---------------------------------------------------------------------------


class EvolutionCharter(Base):
    __tablename__ = "evolution_charters"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    principal_id: Mapped[str] = mapped_column(ForeignKey("principals.id"), index=True)
    charter_type: Mapped[str] = mapped_column(String(32), index=True)  # user / agent / consensus
    version: Mapped[int] = mapped_column(Integer, default=1)
    natural_language_text: Mapped[str] = mapped_column(Text)
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    __table_args__ = (
        Index("uq_charter_ver", "principal_id", "charter_type", "version", unique=True),
    )


class EvolutionEvent(Base):
    __tablename__ = "evolution_events"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    principal_id: Mapped[str] = mapped_column(ForeignKey("principals.id"), index=True)
    policy_version: Mapped[str] = mapped_column(String(64), default="")
    skill_version: Mapped[str] = mapped_column(String(64), default="")
    phase: Mapped[str] = mapped_column(String(32), index=True)  # observe/reflect/mutate/evaluate/rollout/commit_or_rollback
    hypothesis: Mapped[str] = mapped_column(Text, default="")
    mutation: Mapped[str] = mapped_column(Text, default="")
    evaluation_report: Mapped[str] = mapped_column(Text, default="")
    rollout_scope: Mapped[str] = mapped_column(String(128), default="")
    result: Mapped[str] = mapped_column(String(32), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


# ---------------------------------------------------------------------------
# LLM usage tracking
# ---------------------------------------------------------------------------


class LLMUsageRecord(Base):
    """Per-call LLM token usage and cost tracking."""
    __tablename__ = "llm_usage_records"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    tenant_id: Mapped[str] = mapped_column(String(64), default="default", index=True)
    principal_id: Mapped[str] = mapped_column(String(36), default="", index=True)
    session_key: Mapped[str] = mapped_column(String(512), default="")
    model: Mapped[str] = mapped_column(String(128), index=True)
    provider: Mapped[str] = mapped_column(String(64), default="")  # e.g. "cloudsway", "openrouter"
    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0)
    total_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cost_usd: Mapped[str] = mapped_column(String(32), default="0")  # string to avoid float precision issues
    call_type: Mapped[str] = mapped_column(String(32), default="chat")  # chat / consolidation / subagent
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, index=True)

    __table_args__ = (
        Index("ix_usage_principal_created", "principal_id", "created_at"),
        Index("ix_usage_model_created", "model", "created_at"),
    )


# ---------------------------------------------------------------------------
# audit
# ---------------------------------------------------------------------------


class AuditEvent(Base):
    __tablename__ = "audit_events"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    tenant_id: Mapped[str] = mapped_column(String(64), default="default", index=True)
    principal_id: Mapped[str] = mapped_column(String(36), default="", index=True)
    session_key: Mapped[str] = mapped_column(String(512), default="")
    event_type: Mapped[str] = mapped_column(String(64), index=True)
    event_name: Mapped[str] = mapped_column(String(128), index=True)
    severity: Mapped[str] = mapped_column(String(16), default="info")
    payload: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, index=True)


# ---------------------------------------------------------------------------
# system config (key-value store for runtime settings)
# ---------------------------------------------------------------------------


class SystemConfig(Base):
    __tablename__ = "system_config"

    key: Mapped[str] = mapped_column(String(128), primary_key=True)
    value: Mapped[str] = mapped_column(Text, default="")
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)
    updated_by: Mapped[str] = mapped_column(String(36), default="")  # principal_id of who changed it


# ---------------------------------------------------------------------------
# feedback / bug reports
# ---------------------------------------------------------------------------


class FeedbackStatus(str, PyEnum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    WONT_FIX = "wont_fix"


class FeedbackSource(str, PyEnum):
    USER = "user"
    AGENT = "agent"


class Feedback(Base):
    __tablename__ = "feedback"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    principal_id: Mapped[str] = mapped_column(ForeignKey("principals.id"), index=True)
    source: Mapped[FeedbackSource] = mapped_column(SqlEnum(FeedbackSource))
    category: Mapped[str] = mapped_column(String(32), index=True)  # bug / suggestion / complaint
    summary: Mapped[str] = mapped_column(String(512))
    detail: Mapped[str] = mapped_column(Text, default="")
    session_key: Mapped[str] = mapped_column(String(255), default="")
    status: Mapped[FeedbackStatus] = mapped_column(
        SqlEnum(FeedbackStatus), default=FeedbackStatus.PENDING, index=True,
    )
    resolution_note: Mapped[str] = mapped_column(Text, default="")
    resolved_by: Mapped[str] = mapped_column(String(36), default="")  # admin principal_id
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)
