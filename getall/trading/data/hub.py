"""DataHub 统一数据入口.

初始化时加载 exchanges.yaml, 创建各适配器实例.
提供统一接口: hub.exchange, hub.coinglass, hub.followin
"""

import asyncio
import os
import time
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

from getall.config.schema import TradingConfig
from getall.trading.data.cache import DataCache
from getall.trading.data.coinglass import CoinglassAdapter
from getall.trading.data.exchange import ExchangeAdapter
from getall.trading.data.followin import FollowinAdapter


# 连接健康检查间隔 (秒)
_HEALTH_CHECK_INTERVAL_S = 300  # 5 minutes
# 连接验证超时 (秒)
_VALIDATION_TIMEOUT_S = 10


class DataHub:
    """
    Unified data hub for GetAll trading subsystem.

    Loads exchange credentials from exchanges.yaml and exposes
    adapters for exchange, Coinglass, and Followin APIs.

    Features:
    - Connection health checking
    - Automatic reconnection for stale connections
    - Connection validation on creation
    """

    def __init__(self, trading_config: TradingConfig, workspace_path: Path):
        """
        Initialize the DataHub.

        Args:
            trading_config: TradingConfig from getall config schema.
            workspace_path: Workspace root path (for resolving relative paths).
        """
        self._config = trading_config
        self._workspace = workspace_path

        # 缓存实例
        cache_dir = workspace_path / ".cache" / "trading"
        self._cache = DataCache(cache_dir=cache_dir)

        # 交易所配置 (从 yaml 加载)
        self._exchanges_cfg: dict[str, Any] = {}
        self._exchange_adapters: dict[str, ExchangeAdapter] = {}
        self._adapter_last_health_check: dict[str, float] = {}
        self._adapter_lock = asyncio.Lock()

        # 加载 exchanges.yaml
        self._load_exchanges_config()

        # 初始化默认交易所适配器
        self._init_default_exchange()

        # 初始化 Coinglass 适配器
        # Priority: config (env) > exchanges.yaml
        cg_api_key = (
            trading_config.coinglass_api_key
            or self._exchanges_cfg.get("coinglass", {}).get("api_key", "")
        )
        self._coinglass = CoinglassAdapter(
            base_url=trading_config.coinglass_base_url,
            api_key=cg_api_key,
        )
        if cg_api_key:
            logger.info("Coinglass adapter initialized with API key")
        else:
            logger.warning("Coinglass adapter initialized WITHOUT API key — derivatives data will fail")

        # 初始化 Followin 适配器
        fi_api_key = self._exchanges_cfg.get("followin", {}).get("api_key", "")
        self._followin = FollowinAdapter(
            base_url=trading_config.followin_base_url,
            api_key=fi_api_key,
            lang=trading_config.followin_lang,
        )

        logger.info("DataHub initialized successfully")

    # ──────────────────────── 配置加载 ────────────────────────

    def _load_exchanges_config(self) -> None:
        """从 exchanges.yaml 加载交易所凭据和 API 密钥."""
        config_path = self._workspace / self._config.exchanges_config_path
        if not config_path.exists():
            logger.warning(f"Exchanges config not found: {config_path}")
            logger.warning("Trading adapters will run with empty credentials")
            return
        try:
            with open(config_path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            self._exchanges_cfg = data
            logger.info(f"Loaded exchanges config from {config_path}")
        except Exception as e:
            logger.error(f"Failed to load exchanges config: {e}")

    def _init_default_exchange(self) -> None:
        """初始化默认交易所适配器."""
        name = self._config.default_exchange
        # exchanges.yaml 中交易所配置在 'exchanges' 键下
        exchanges_section = self._exchanges_cfg.get("exchanges", {})
        ex_cfg = exchanges_section.get(name, {})
        if not ex_cfg:
            logger.debug(f"No config for default exchange '{name}', skipping adapter init")
            return
        try:
            adapter = ExchangeAdapter(
                exchange_name=name,
                api_key=ex_cfg.get("api_key", "") or os.environ.get(f"{name.upper()}_API_KEY", ""),
                secret=ex_cfg.get("secret", "") or os.environ.get(f"{name.upper()}_SECRET", ""),
                password=ex_cfg.get("password", ""),
                sandbox=ex_cfg.get("sandbox", False),
            )
            self._exchange_adapters[name] = adapter
        except Exception as e:
            logger.error(f"Failed to init exchange adapter '{name}': {e}")

    # ──────────────────────── 公开属性 ────────────────────────

    @property
    def exchange(self) -> ExchangeAdapter | None:
        """Get the default exchange adapter."""
        return self._exchange_adapters.get(self._config.default_exchange)

    @property
    def coinglass(self) -> CoinglassAdapter:
        """Get the Coinglass data adapter."""
        return self._coinglass

    @property
    def followin(self) -> FollowinAdapter:
        """Get the Followin news adapter."""
        return self._followin

    @property
    def cache(self) -> DataCache:
        """Get the shared data cache."""
        return self._cache

    @property
    def configured_exchanges(self) -> list[str]:
        """返回 exchanges.yaml 中已配置的交易所名称列表."""
        exchanges = self._exchanges_cfg.get("exchanges", {})
        return [k for k in exchanges if isinstance(exchanges[k], dict) and exchanges[k].get("api_key")]

    # ──────────────────────── 动态加载 ────────────────────────

    async def get_exchange(self, name: str) -> ExchangeAdapter | None:
        """
        Get or create an exchange adapter by name with health checking.

        Args:
            name: Exchange name (binance / bitget).

        Returns:
            ExchangeAdapter instance or None if config missing.
        """
        name = name.lower()

        async with self._adapter_lock:
            # Check if adapter exists and is healthy
            if name in self._exchange_adapters:
                adapter = self._exchange_adapters[name]
                if await self._is_adapter_healthy(name, adapter):
                    return adapter
                else:
                    # Adapter unhealthy, close and recreate
                    logger.warning(f"Exchange adapter '{name}' unhealthy, recreating...")
                    try:
                        await adapter.close()
                    except Exception as e:
                        logger.debug(f"Error closing unhealthy adapter '{name}': {e}")
                    del self._exchange_adapters[name]
                    self._adapter_last_health_check.pop(name, None)

            # Try to create new adapter
            exchanges_section = self._exchanges_cfg.get("exchanges", {})
            ex_cfg = exchanges_section.get(name, {})
            if not ex_cfg:
                logger.warning(f"No config for exchange '{name}'")
                return None

            try:
                adapter = await self._create_and_validate_adapter(name, ex_cfg)
                if adapter:
                    self._exchange_adapters[name] = adapter
                    self._adapter_last_health_check[name] = time.time()
                return adapter
            except Exception as e:
                logger.error(f"Failed to create exchange adapter '{name}': {e}")
                return None

    def get_exchange_sync(self, name: str) -> ExchangeAdapter | None:
        """
        Synchronous version of get_exchange (without health check).

        Used for quick access when caller doesn't need health validation.
        Falls back to creating adapter synchronously if not cached.
        """
        name = name.lower()
        if name in self._exchange_adapters:
            return self._exchange_adapters[name]

        # Try to create new adapter synchronously (no validation)
        exchanges_section = self._exchanges_cfg.get("exchanges", {})
        ex_cfg = exchanges_section.get(name, {})
        if not ex_cfg:
            return None

        try:
            adapter = ExchangeAdapter(
                exchange_name=name,
                api_key=ex_cfg.get("api_key", ""),
                secret=ex_cfg.get("secret", ""),
                password=ex_cfg.get("password", ""),
                sandbox=ex_cfg.get("sandbox", False),
            )
            self._exchange_adapters[name] = adapter
            self._adapter_last_health_check[name] = time.time()
            return adapter
        except Exception as e:
            logger.error(f"Failed to create exchange adapter '{name}' (sync): {e}")
            return None

    async def _create_and_validate_adapter(
        self,
        name: str,
        ex_cfg: dict[str, Any]
    ) -> ExchangeAdapter | None:
        """
        Create a new adapter and validate the connection.

        Args:
            name: Exchange name
            ex_cfg: Exchange configuration dict

        Returns:
            Validated adapter or None if validation fails
        """
        adapter = ExchangeAdapter(
            exchange_name=name,
            api_key=ex_cfg.get("api_key", ""),
            secret=ex_cfg.get("secret", ""),
            password=ex_cfg.get("password", ""),
            sandbox=ex_cfg.get("sandbox", False),
        )

        # Validate connection by fetching balance
        try:
            result = await asyncio.wait_for(
                adapter.get_balance(),
                timeout=_VALIDATION_TIMEOUT_S
            )
            if isinstance(result, dict) and "error" in result:
                logger.error(f"Exchange adapter '{name}' validation failed: {result['error']}")
                await adapter.close()
                return None

            logger.info(f"Exchange adapter '{name}' created and validated successfully")
            return adapter
        except asyncio.TimeoutError:
            logger.error(f"Exchange adapter '{name}' validation timed out")
            await adapter.close()
            return None
        except Exception as e:
            logger.error(f"Exchange adapter '{name}' validation error: {e}")
            try:
                await adapter.close()
            except Exception:
                pass
            return None

    async def _is_adapter_healthy(self, name: str, adapter: ExchangeAdapter) -> bool:
        """
        Check if an adapter is healthy.

        Returns True if:
        - Health check interval has not passed (assume healthy)
        - Health check passes (can fetch server time or balance)
        """
        last_check = self._adapter_last_health_check.get(name, 0)
        if time.time() - last_check < _HEALTH_CHECK_INTERVAL_S:
            return True  # Skip check if recently validated

        # Perform health check
        try:
            # Try to load markets (lightweight operation)
            result = await asyncio.wait_for(
                adapter.get_markets(),
                timeout=_VALIDATION_TIMEOUT_S
            )
            if isinstance(result, dict) and "error" in result:
                logger.warning(f"Health check failed for '{name}': {result['error']}")
                return False

            self._adapter_last_health_check[name] = time.time()
            return True
        except asyncio.TimeoutError:
            logger.warning(f"Health check timed out for '{name}'")
            return False
        except Exception as e:
            logger.warning(f"Health check error for '{name}': {e}")
            return False

    # ──────────────────────── 生命周期 ────────────────────────

    async def close(self) -> None:
        """
        关闭所有适配器, 释放资源.

        Gracefully handles errors during close to ensure all adapters
        are attempted to be closed even if some fail.
        """
        errors: list[tuple[str, Exception]] = []

        async with self._adapter_lock:
            for name, adapter in list(self._exchange_adapters.items()):
                try:
                    await asyncio.wait_for(adapter.close(), timeout=5.0)
                    logger.debug(f"Closed exchange adapter '{name}'")
                except asyncio.TimeoutError:
                    errors.append((name, TimeoutError("Close timed out")))
                    logger.warning(f"Timeout closing exchange adapter '{name}'")
                except Exception as e:
                    errors.append((name, e))
                    logger.error(f"Error closing exchange adapter '{name}': {e}")

            self._exchange_adapters.clear()
            self._adapter_last_health_check.clear()

        if errors:
            error_summary = ", ".join(f"{name}: {e}" for name, e in errors)
            logger.warning(f"Some adapters failed to close: {error_summary}")

        logger.info("DataHub closed")
