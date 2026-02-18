"""Centralised settings for GetAll, loaded from env / .env."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class GetAllSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="GETALL_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- general ---
    app_name: str = "GetAll"
    env: str = "dev"
    debug: bool = False

    # --- HTTP ---
    host: str = "0.0.0.0"
    port: int = 8080

    # --- database (PostgreSQL) ---
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/getall"

    # --- cache / queue / locks (Redis) ---
    redis_url: str = "redis://localhost:6379/0"

    # --- file-system paths ---
    state_dir: Path = Field(default_factory=lambda: Path.home() / ".getall")
    workspace_dir: Path = Field(default_factory=lambda: Path.home() / ".getall" / "workspace")

    # --- Bitget (direct REST; credentials are per-user via DB) ---
    bitget_base_url: str = "https://api.bitget.com"

    # --- credential encryption ---
    credential_key: str = ""  # Fernet key for encrypting user exchange credentials

    # --- worker tuning ---
    reminder_poll_seconds: int = 5
    queue_poll_seconds: int = 1
    max_concurrent_workers: int = 4


@lru_cache
def get_settings() -> GetAllSettings:
    s = GetAllSettings()
    s.state_dir.mkdir(parents=True, exist_ok=True)
    s.workspace_dir.mkdir(parents=True, exist_ok=True)
    return s
