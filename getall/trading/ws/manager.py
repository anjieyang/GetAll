# WS 连接管理器: 交易所账户 (ws:account) + bwenews 新闻 (ws:breaking-news)
# 职责:
#   1. 开机时为每个已配置的交易所建立 WebSocket 连接
#   2. 订阅 4 类账户事件 (仓位/订单/成交/余额)
#   3. 事件触发 → 调用 handlers 处理 → 静默写入 memory / 重大事件通知用户
#   4. 自动重连 (断线后指数退避重试)
#   5. 认证/权限错误 → 停止订阅, 通知 agent session (不直接推用户)
#   6. 服务注册与动态启停 (用户可通过 "关闭 ws:xxx" 管理)

import asyncio
import re
from pathlib import Path
from typing import Any, Callable, Awaitable

import yaml
from loguru import logger

from getall.bus.events import InboundMessage, OutboundMessage
from getall.config.schema import TradingConfig
from getall.trading.ws.handlers import WSEventHandlers


# ---------------------------------------------------------------------------
# 服务状态枚举
# ---------------------------------------------------------------------------
_STATUS_RUNNING = "running"
_STATUS_STOPPED = "stopped"
_STATUS_CONNECTING = "connecting"
_STATUS_ERROR = "error"

# ---------------------------------------------------------------------------
# 认证 / 权限错误识别模式 (匹配则不重连, 直接放弃)
# ---------------------------------------------------------------------------
_AUTH_ERROR_PATTERNS: list[re.Pattern] = [
    re.compile(r'"code"\s*:\s*-20(?:08|15)', re.IGNORECASE),   # Binance -2008 / -2015
    re.compile(r'"code"\s*:\s*30011', re.IGNORECASE),           # Bitget Invalid ACCESS_KEY
    re.compile(r'Invalid API.?Key', re.IGNORECASE),
    re.compile(r'Invalid ACCESS_KEY', re.IGNORECASE),
    re.compile(r'permissions? for action', re.IGNORECASE),
    re.compile(r'API key missing', re.IGNORECASE),
    re.compile(r'unauthorized', re.IGNORECASE),
]


def _is_auth_error(error: Exception) -> bool:
    """Check if an exception is an authentication / permission error."""
    msg = str(error)
    return any(p.search(msg) for p in _AUTH_ERROR_PATTERNS)


class WSManager:
    """
    Multi-exchange WebSocket connection manager.

    Manages two logical services:
    - ws:account   — exchange account streams (positions / orders / trades / balance)
    - ws:breaking-news — bwenews breaking-news stream

    All services are registered in ``_service_registry`` so users can
    toggle them with natural-language commands.
    """

    def __init__(
        self,
        config: TradingConfig,
        workspace_path: Path,
        send_callback: Callable[[OutboundMessage], Awaitable[None]],
        inbound_callback: Callable[[InboundMessage], Awaitable[None]] | None = None,
    ) -> None:
        """
        Initialize the WS manager.

        Args:
            config: TradingConfig from schema.
            workspace_path: Root workspace path (for memory files).
            send_callback: Async callable to push OutboundMessage to user.
            inbound_callback: Async callable to inject InboundMessage into
                agent session (system messages). When None, auth errors are
                only logged but not sent to the agent.
        """
        self.config = config
        self.workspace_path = workspace_path
        self.send_callback = send_callback
        self._inbound_callback = inbound_callback

        # 事件处理器
        self.handlers = WSEventHandlers(
            workspace_path,
            send_callback,
            inbound_callback=inbound_callback,
            debug_console_push=config.debug_console_push,
        )

        # ccxt.pro 交易所实例 {name: exchange}
        self._exchanges: dict[str, Any] = {}

        # 服务注册表 {service_name: {"status": str, "tasks": list[Task]}}
        self._service_registry: dict[str, dict[str, Any]] = {}

        # 已因认证错误放弃的 exchange/stream (避免重复通知)
        self._auth_failed: set[str] = set()

        # 全局运行标志
        self._running = False

    # ------------------------------------------------------------------
    # 交易所加载 (从 exchanges.yaml)
    # ------------------------------------------------------------------

    def _load_exchanges(self) -> dict[str, Any]:
        """
        Load exchange instances from exchanges.yaml using ccxt.pro.

        Returns:
            Dict of exchange name → ccxt.pro exchange instance.
        """
        exchanges_path = self.workspace_path / self.config.exchanges_config_path
        if not exchanges_path.exists():
            logger.warning(f"exchanges.yaml not found at {exchanges_path}")
            return {}

        try:
            raw = yaml.safe_load(exchanges_path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.error(f"Failed to parse exchanges.yaml: {e}")
            return {}

        exchange_cfgs: dict = raw.get("exchanges", {})
        instances: dict[str, Any] = {}

        for name, cfg in exchange_cfgs.items():
            try:
                # 动态导入 ccxt.pro (延迟导入, 减少启动依赖)
                import ccxt.pro as ccxtpro

                exchange_cls = getattr(ccxtpro, name, None)
                if exchange_cls is None:
                    logger.warning(f"ccxt.pro does not support exchange: {name}")
                    continue

                params: dict[str, Any] = {
                    "apiKey": cfg.get("api_key", ""),
                    "secret": cfg.get("secret", ""),
                    "enableRateLimit": True,
                }
                if cfg.get("password"):
                    params["password"] = cfg["password"]
                if cfg.get("sandbox"):
                    params["sandbox"] = True

                # 默认使用合约模式
                params["options"] = {"defaultType": "swap"}

                instances[name] = exchange_cls(params)
                logger.info(f"Loaded exchange: {name}")
            except Exception as e:
                logger.error(f"Failed to init exchange {name}: {e}")

        return instances

    # ------------------------------------------------------------------
    # 启动 / 停止
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Start all WebSocket services based on config."""
        if self._running:
            logger.warning("WSManager already running, skip start()")
            return

        self._running = True
        logger.info("WSManager starting...")

        # --- ws:account ---
        self._exchanges = self._load_exchanges()
        if self._exchanges:
            self._start_account_service()
        else:
            logger.warning("[ws:account] no exchanges loaded, service skipped")

        # --- ws:breaking-news ---
        self._start_news_service()

        logger.info(
            f"WSManager started — active services: "
            f"{[k for k, v in self._service_registry.items() if v['status'] == _STATUS_RUNNING]}"
        )

    async def stop(self) -> None:
        """Gracefully stop all WS services and close exchange connections."""
        logger.info("WSManager stopping...")
        self._running = False

        # 取消所有任务
        for svc_name, svc in self._service_registry.items():
            for task in svc.get("tasks", []):
                task.cancel()
            svc["status"] = _STATUS_STOPPED
            logger.info(f"[{svc_name}] stopped")

        # 关闭 ccxt.pro 交易所连接
        for name, exchange in self._exchanges.items():
            try:
                await exchange.close()
                logger.info(f"Closed exchange connection: {name}")
            except Exception as e:
                logger.warning(f"Error closing exchange {name}: {e}")

        self._exchanges.clear()
        logger.info("WSManager stopped")

    # ------------------------------------------------------------------
    # 服务注册与动态管理
    # ------------------------------------------------------------------

    def _start_account_service(self) -> None:
        """启动 ws:account 服务 — 为每个交易所创建 4 类 watcher."""
        tasks: list[asyncio.Task] = []

        for name, exchange in self._exchanges.items():
            # 初始快照: REST 拉一次全量挂单，作为 WS 增量更新的基线
            tasks.append(asyncio.create_task(self._bootstrap_open_orders(name, exchange)))

            # 检查交易所是否支持 WS 方法
            if hasattr(exchange, "watchPositions"):
                tasks.append(asyncio.create_task(self._watch_loop(name, exchange, "positions")))
            if hasattr(exchange, "watchOrders"):
                tasks.append(asyncio.create_task(self._watch_loop(name, exchange, "orders")))
            if hasattr(exchange, "watchMyTrades"):
                tasks.append(asyncio.create_task(self._watch_loop(name, exchange, "trades")))
            if hasattr(exchange, "watchBalance"):
                tasks.append(asyncio.create_task(self._watch_loop(name, exchange, "balance")))

            logger.info(f"[ws:account] watchers created for {name}")

        self._service_registry["ws:account"] = {
            "status": _STATUS_RUNNING,
            "tasks": tasks,
        }

    def _start_news_service(self) -> None:
        """启动 ws:breaking-news 服务."""
        from getall.trading.ws.news_ws import BweNewsWS

        news_ws = BweNewsWS(self.config, self.workspace_path, self.send_callback)
        task = asyncio.create_task(news_ws.start())

        self._service_registry["ws:breaking-news"] = {
            "status": _STATUS_RUNNING,
            "tasks": [task],
            "_instance": news_ws,
        }
        logger.info("[ws:breaking-news] bwenews WebSocket started")

    async def enable_service(self, name: str) -> str:
        """
        Enable (restart) a previously disabled service.

        Args:
            name: Service name, e.g. "ws:account" or "ws:breaking-news".

        Returns:
            Human-readable status message.
        """
        if name not in ("ws:account", "ws:breaking-news"):
            return f"Unknown service: {name}"

        svc = self._service_registry.get(name)
        if svc and svc["status"] == _STATUS_RUNNING:
            return f"{name} is already running"

        if name == "ws:account":
            if not self._exchanges:
                self._exchanges = self._load_exchanges()
            if self._exchanges:
                self._start_account_service()
                return f"{name} enabled"
            return f"{name} failed — no exchanges available"

        if name == "ws:breaking-news":
            self._start_news_service()
            return f"{name} enabled"

        return f"Cannot enable {name}"

    async def disable_service(self, name: str) -> str:
        """
        Disable a running service.

        Args:
            name: Service name.

        Returns:
            Human-readable status message.
        """
        svc = self._service_registry.get(name)
        if not svc:
            return f"{name} is not registered"
        if svc["status"] == _STATUS_STOPPED:
            return f"{name} is already stopped"

        for task in svc.get("tasks", []):
            task.cancel()

        # 如果是 news 服务, 调用其 stop()
        instance = svc.get("_instance")
        if instance and hasattr(instance, "stop"):
            await instance.stop()

        svc["status"] = _STATUS_STOPPED
        svc["tasks"] = []
        logger.info(f"[{name}] disabled by user")
        return f"{name} disabled"

    def get_service_status(self) -> dict[str, str]:
        """
        Get status of all registered services.

        Returns:
            Dict of service_name → status string.
        """
        result: dict[str, str] = {}

        # Show services that are registered but not yet started
        if "ws:account" not in self._service_registry:
            result["ws:account"] = _STATUS_STOPPED

        if "ws:breaking-news" not in self._service_registry:
            result["ws:breaking-news"] = _STATUS_STOPPED

        for name, svc in self._service_registry.items():
            result[name] = svc["status"]

        return result

    # ------------------------------------------------------------------
    # 内部: WS 监听循环 (带指数退避重连)
    # ------------------------------------------------------------------

    async def _watch_loop(self, exchange_name: str, exchange: Any, stream: str) -> None:
        """
        Generic watch loop with exponential-backoff reconnect.

        认证/权限类错误 (如 Invalid API-key, permissions) 会立刻停止该
        exchange/stream 的订阅, 并通过 inbound_callback 向 agent session
        发一条系统消息, 由 agent 自行决定是否通知用户.

        Args:
            exchange_name: Human-readable exchange name.
            exchange: ccxt.pro exchange instance.
            stream: One of "positions", "orders", "trades", "balance".
        """
        delay = self.config.ws_reconnect_delay
        max_delay = self.config.ws_reconnect_max_delay
        tag = f"{exchange_name}/{stream}"

        # Seed baseline state before starting WS loops (prevents noisy "new position" alerts on startup
        # and fills fields that WS streams may omit, e.g. Binance futures leverage).
        if stream == "positions":
            await self._bootstrap_positions(exchange_name, exchange)

        while self._running:
            try:
                if stream == "positions":
                    data = await exchange.watchPositions()
                    await self.handlers.on_positions_update(exchange_name, data)
                elif stream == "orders":
                    data = await exchange.watchOrders()
                    mtype = (getattr(exchange, "options", {}) or {}).get("defaultType", "swap")
                    await self.handlers.on_orders_update(exchange_name, data, market_type=str(mtype))
                elif stream == "trades":
                    data = await exchange.watchMyTrades()
                    await self.handlers.on_trades_update(exchange_name, data)
                elif stream == "balance":
                    data = await exchange.watchBalance()
                    await self.handlers.on_balance_update(exchange_name, data)

                # 成功后重置退避延迟
                delay = self.config.ws_reconnect_delay

            except asyncio.CancelledError:
                logger.info(f"[ws:account] {tag} watcher cancelled")
                break
            except Exception as e:
                # ── 认证 / 权限错误 → 停止订阅, 通知 agent ──
                if _is_auth_error(e):
                    logger.error(
                        f"[ws:account] {tag} auth/permission error: {e} — "
                        f"subscription cancelled (will not retry)"
                    )
                    await self._notify_agent_auth_error(exchange_name, stream, e)
                    break

                # ── 其他错误 → 指数退避重连 ──
                logger.warning(
                    f"[ws:account] {tag} error: {e}, "
                    f"reconnecting in {delay}s..."
                )
                await asyncio.sleep(delay)
                delay = min(delay * 2, max_delay)

    # ------------------------------------------------------------------
    # 初始快照: REST 拉一次持仓 (positions)
    # ------------------------------------------------------------------

    async def _bootstrap_positions(self, exchange_name: str, exchange: Any) -> None:
        """
        Fetch a one-time positions snapshot via REST to seed local cache.

        Some WS position streams (notably Binance Futures) may omit static fields
        like leverage. We seed once via REST and let WS updates keep PnL/size fresh.
        """
        try:
            # load markets first (some exchanges need it for symbol resolution)
            if hasattr(exchange, "loadMarkets"):
                await exchange.loadMarkets()
            elif hasattr(exchange, "load_markets"):
                await exchange.load_markets()

            if hasattr(exchange, "fetchPositions"):
                positions = await exchange.fetchPositions()
            elif hasattr(exchange, "fetch_positions"):
                positions = await exchange.fetch_positions()
            else:
                return

            if not isinstance(positions, list):
                return

            # For Binance, fetch leverage from positionRisk endpoint
            leverage_map: dict[str, float] = {}
            if exchange_name == "binance":
                try:
                    if hasattr(exchange, "fapiPrivateV2GetPositionRisk"):
                        risk_data = await exchange.fapiPrivateV2GetPositionRisk()
                        for p in (risk_data or []):
                            symbol = p.get("symbol", "")
                            leverage = p.get("leverage")
                            if symbol and leverage:
                                leverage_map[symbol] = float(leverage)
                        logger.debug(f"[ws:account] fetched leverage for {len(leverage_map)} symbols")
                except Exception as e:
                    logger.warning(f"[ws:account] failed to fetch leverage: {e}")

            # Merge leverage into positions
            if leverage_map:
                for pos in positions:
                    # ccxt symbol format: SOL/USDT:USDT, raw format: SOLUSDT
                    sym = pos.get("symbol", "")
                    raw_sym = sym.replace("/", "").replace(":USDT", "")
                    if raw_sym in leverage_map:
                        pos["leverage"] = leverage_map[raw_sym]

            await self.handlers.on_positions_update(exchange_name, positions, notify=False)
            logger.info(f"[ws:account] bootstrapped positions: {exchange_name} ({len(positions)} total)")
        except asyncio.CancelledError:
            return
        except Exception as e:
            # 失败不影响 WS 继续运行（watchPositions 仍能拿到数据）
            logger.warning(f"[ws:account] bootstrap positions failed for {exchange_name}: {e}")

    # ------------------------------------------------------------------
    # 初始快照: REST 拉全量挂单 (open orders)
    # ------------------------------------------------------------------

    async def _bootstrap_open_orders(self, exchange_name: str, exchange: Any) -> None:
        """
        Fetch a one-time open-orders snapshot via REST to seed local cache.

        After seeding, watchOrders() will keep the cache updated in real-time.
        """
        try:
            # load markets first (some exchanges need it for symbol resolution)
            if hasattr(exchange, "loadMarkets"):
                await exchange.loadMarkets()
            elif hasattr(exchange, "load_markets"):
                await exchange.load_markets()

            # Binance(ccxt): fetchOpenOrders without symbol triggers a warning exception unless disabled
            try:
                if hasattr(exchange, "options") and isinstance(exchange.options, dict):
                    exchange.options["warnOnFetchOpenOrdersWithoutSymbol"] = False
            except Exception:
                pass

            if hasattr(exchange, "fetchOpenOrders"):
                orders = await exchange.fetchOpenOrders()
            elif hasattr(exchange, "fetch_open_orders"):
                orders = await exchange.fetch_open_orders()
            else:
                return

            if not isinstance(orders, list):
                return

            mtype = (getattr(exchange, "options", {}) or {}).get("defaultType", "swap")
            await self.handlers.on_orders_update(
                exchange_name,
                orders,
                market_type=str(mtype),
                notify_agent=False,  # seed baseline only
            )
            logger.info(f"[ws:account] bootstrapped open orders: {exchange_name} ({len(orders)} open)")
        except asyncio.CancelledError:
            return
        except Exception as e:
            # 认证 / 权限错误: 不影响 WS 继续运行（watchOrders 可能仍能拿到数据）
            logger.warning(f"[ws:account] bootstrap open orders failed for {exchange_name}: {e}")
    # ------------------------------------------------------------------
    # 内部: 认证错误 → 通知 agent session
    # ------------------------------------------------------------------

    async def _notify_agent_auth_error(
        self, exchange_name: str, stream: str, error: Exception
    ) -> None:
        """
        Send a system message to the agent session about an auth error.

        每个 exchange 只通知一次 (首个 stream 触发时发送, 其余静默).
        """
        # 去重: 同一交易所只通知一次
        if exchange_name in self._auth_failed:
            return
        self._auth_failed.add(exchange_name)

        content = (
            f"[ws:account] {exchange_name} WebSocket 订阅因认证/权限错误已自动取消。\n"
            f"错误详情: {error}\n"
            f"可能原因: API key 无效、未开启合约权限、或 IP 不在白名单。\n"
            f"已停止 {exchange_name} 的所有 account stream 订阅 (positions/orders/trades/balance)。\n"
            f"用户如需恢复, 需在 {exchange_name} 后台检查 API key 权限设置。"
        )

        if self._inbound_callback:
            try:
                msg = InboundMessage(
                    channel="system",
                    sender_id="ws_manager",
                    chat_id="__active__",
                    content=content,
                )
                await self._inbound_callback(msg)
                logger.info(f"[ws:account] auth error notification sent to agent session for {exchange_name}")
            except Exception as cb_err:
                logger.warning(f"[ws:account] failed to notify agent: {cb_err}")
        else:
            logger.info(f"[ws:account] no inbound_callback, auth error only logged")
