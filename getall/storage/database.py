"""Async database engine & session factory."""

from __future__ import annotations

from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from getall.settings import GetAllSettings, get_settings
from getall.storage.models import Base

# Module-level singleton (created on first call to get_engine / get_session_factory)
_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def get_engine(settings: GetAllSettings | None = None) -> AsyncEngine:
    global _engine
    if _engine is None:
        s = settings or get_settings()
        _engine = create_async_engine(
            s.database_url,
            echo=s.debug,
            pool_pre_ping=True,
            pool_recycle=1800,
        )
    return _engine


def get_session_factory(settings: GetAllSettings | None = None) -> async_sessionmaker[AsyncSession]:
    global _session_factory
    if _session_factory is None:
        engine = get_engine(settings)
        _session_factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    return _session_factory


async def get_session() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency – yields a scoped session per request."""
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def create_all_tables() -> None:
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def dispose_engine() -> None:
    global _engine, _session_factory
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _session_factory = None
