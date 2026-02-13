"""FastAPI application factory with lifespan for GetAll."""

from __future__ import annotations

from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from fastapi import FastAPI

from getall.settings import get_settings
from getall.storage.database import create_all_tables, dispose_engine


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup: ensure DB schema. Shutdown: dispose engine."""
    settings = get_settings()
    await create_all_tables()
    yield
    await dispose_engine()


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        lifespan=lifespan,
    )

    # ── mount routers ──
    from getall.api.routes import health, identity, memory, reminders
    from getall.api.routes import lark_webhook

    app.include_router(health.router)
    app.include_router(identity.router, prefix="/api/v1/identity", tags=["identity"])
    app.include_router(memory.router, prefix="/api/v1/memory", tags=["memory"])
    app.include_router(reminders.router, prefix="/api/v1/reminders", tags=["reminders"])
    app.include_router(lark_webhook.router, tags=["lark"])

    return app
