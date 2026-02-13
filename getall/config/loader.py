"""Configuration loading utilities."""

import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from getall.config.schema import Config

# Load .env into os.environ (won't overwrite existing env vars)
load_dotenv(override=False)


def get_config_path() -> Path:
    """Get the default configuration file path."""
    return Path.home() / ".getall" / "config.json"


def get_data_dir() -> Path:
    """Get the GetAll data directory."""
    from getall.utils.helpers import get_data_path
    return get_data_path()


def load_config(config_path: Path | None = None) -> Config:
    """
    Load configuration from file, then apply env-var overrides.

    Priority (highest → lowest):
        1. GETALL_* environment variables / .env
        2. ~/.getall/config.json
        3. Built-in defaults
    """
    path = config_path or get_config_path()

    if path.exists():
        try:
            with open(path) as f:
                data = json.load(f)
            data = _migrate_config(data)
            config = Config.model_validate(convert_keys(data))
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Warning: Failed to load config from {path}: {e}")
            print("Using default configuration.")
            config = Config()
    else:
        config = Config()

    _apply_env_overrides(config)
    return config


# ---------------------------------------------------------------------------
# Flat env-var overrides — keeps .env readable (no ugly __ nesting)
# ---------------------------------------------------------------------------

_ENV_PROVIDER_MAP: dict[str, str] = {
    "GETALL_OPENAI_API_KEY": "openai",
    "GETALL_OPENROUTER_API_KEY": "openrouter",
    "GETALL_ANTHROPIC_API_KEY": "anthropic",
    "GETALL_DEEPSEEK_API_KEY": "deepseek",
    "GETALL_GROQ_API_KEY": "groq",
    "GETALL_GEMINI_API_KEY": "gemini",
    "GETALL_ZHIPU_API_KEY": "zhipu",
    "GETALL_DASHSCOPE_API_KEY": "dashscope",
    "GETALL_MOONSHOT_API_KEY": "moonshot",
    "GETALL_MINIMAX_API_KEY": "minimax",
    "GETALL_AIHUBMIX_API_KEY": "aihubmix",
    "GETALL_VLLM_API_KEY": "vllm",
}

_ENV_PROVIDER_BASE_MAP: dict[str, str] = {
    "GETALL_OPENAI_API_BASE": "openai",
    "GETALL_OPENROUTER_API_BASE": "openrouter",
    "GETALL_VLLM_API_BASE": "vllm",
}


def _apply_env_overrides(config: Config) -> None:
    """Apply flat GETALL_* env vars on top of the loaded config."""

    # --- LLM model ---
    if val := os.environ.get("GETALL_MODEL"):
        config.agents.defaults.model = val

    # --- Provider API keys ---
    for env_key, provider_name in _ENV_PROVIDER_MAP.items():
        if val := os.environ.get(env_key):
            provider = getattr(config.providers, provider_name, None)
            if provider is not None:
                provider.api_key = val

    # --- Provider API base URLs ---
    for env_key, provider_name in _ENV_PROVIDER_BASE_MAP.items():
        if val := os.environ.get(env_key):
            provider = getattr(config.providers, provider_name, None)
            if provider is not None:
                provider.api_base = val

    # --- Telegram ---
    if val := os.environ.get("GETALL_TELEGRAM_TOKEN"):
        config.channels.telegram.token = val
        config.channels.telegram.enabled = True
    if val := os.environ.get("GETALL_TELEGRAM_ALLOW_FROM"):
        config.channels.telegram.allow_from = [v.strip() for v in val.split(",") if v.strip()]
    if val := os.environ.get("GETALL_TELEGRAM_PROXY"):
        config.channels.telegram.proxy = val

    # --- Discord ---
    if val := os.environ.get("GETALL_DISCORD_TOKEN"):
        config.channels.discord.token = val
        config.channels.discord.enabled = True
    if val := os.environ.get("GETALL_DISCORD_ALLOW_FROM"):
        config.channels.discord.allow_from = [v.strip() for v in val.split(",") if v.strip()]

    # --- Feishu / Lark ---
    if val := os.environ.get("GETALL_FEISHU_APP_ID"):
        config.channels.feishu.app_id = val
        config.channels.feishu.enabled = True
    if val := os.environ.get("GETALL_FEISHU_APP_SECRET"):
        config.channels.feishu.app_secret = val
    if val := os.environ.get("GETALL_FEISHU_ENCRYPT_KEY"):
        config.channels.feishu.encrypt_key = val
    if val := os.environ.get("GETALL_FEISHU_VERIFICATION_TOKEN"):
        config.channels.feishu.verification_token = val
    if val := os.environ.get("GETALL_FEISHU_DOMAIN"):
        config.channels.feishu.domain = val
    if os.environ.get("GETALL_FEISHU_USE_WEBHOOK", "").lower() in ("1", "true", "yes"):
        config.channels.feishu.use_webhook = True

    # --- DingTalk ---
    if val := os.environ.get("GETALL_DINGTALK_CLIENT_ID"):
        config.channels.dingtalk.client_id = val
        config.channels.dingtalk.enabled = True
    if val := os.environ.get("GETALL_DINGTALK_CLIENT_SECRET"):
        config.channels.dingtalk.client_secret = val

    # --- Web search ---
    if val := os.environ.get("GETALL_BRAVE_API_KEY"):
        config.tools.web.search.api_key = val

    # --- Gateway ---
    if val := os.environ.get("GETALL_GATEWAY_PORT"):
        config.gateway.port = int(val)


def save_config(config: Config, config_path: Path | None = None) -> None:
    """
    Save configuration to file.
    
    Args:
        config: Configuration to save.
        config_path: Optional path to save to. Uses default if not provided.
    """
    path = config_path or get_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to camelCase format
    data = config.model_dump()
    data = convert_to_camel(data)
    
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _migrate_config(data: dict) -> dict:
    """Migrate old config formats to current."""
    # Move tools.exec.restrictToWorkspace → tools.restrictToWorkspace
    tools = data.get("tools", {})
    exec_cfg = tools.get("exec", {})
    if "restrictToWorkspace" in exec_cfg and "restrictToWorkspace" not in tools:
        tools["restrictToWorkspace"] = exec_cfg.pop("restrictToWorkspace")
    return data


def convert_keys(data: Any) -> Any:
    """Convert camelCase keys to snake_case for Pydantic."""
    if isinstance(data, dict):
        return {camel_to_snake(k): convert_keys(v) for k, v in data.items()}
    if isinstance(data, list):
        return [convert_keys(item) for item in data]
    return data


def convert_to_camel(data: Any) -> Any:
    """Convert snake_case keys to camelCase."""
    if isinstance(data, dict):
        return {snake_to_camel(k): convert_to_camel(v) for k, v in data.items()}
    if isinstance(data, list):
        return [convert_to_camel(item) for item in data]
    return data


def camel_to_snake(name: str) -> str:
    """Convert camelCase to snake_case."""
    result = []
    for i, char in enumerate(name):
        if char.isupper() and i > 0:
            result.append("_")
        result.append(char.lower())
    return "".join(result)


def snake_to_camel(name: str) -> str:
    """Convert snake_case to camelCase."""
    components = name.split("_")
    return components[0] + "".join(x.title() for x in components[1:])
