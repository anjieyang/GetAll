# bwenews 方程式新闻 WebSocket: Breaking News 实时推送
# 服务名: ws:breaking-news
#
# 行为:
#   - 连接 bwenews WebSocket 长连接, 持续接收 Breaking News
#   - 🔴 无条件推送: Breaking News 直接发送给用户 (不过滤币种)
#   - 如涉及用户持仓币种 → 额外标注 ⚠️
#   - 写入 memory/trading/anomalies.jsonl (type: "breaking_news")
#   - 自动重连 (指数退避)
#
# 注意: bwenews 协议尚未最终确认, 当前为合理预设实现.
#       确认实际协议后, 调整 _handle_message 中的解析逻辑即可.

import asyncio
import json
import ssl
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Awaitable

from loguru import logger

from getall.bus.events import OutboundMessage
from getall.routing import load_all_routes
from getall.trading.watch_scope import get_active_watch_coins, extract_coin
from getall.config.schema import TradingConfig


# ---------------------------------------------------------------------------
# 默认配置
# ---------------------------------------------------------------------------

_DEFAULT_WS_URL = "wss://bwenews-api.bwe-ws.com/ws"
_DEFAULT_RECONNECT_DELAY = 5
_DEFAULT_RECONNECT_MAX_DELAY = 60


class BweNewsWS:
    """
    bwenews Breaking-News WebSocket client.

    Connects to the bwenews WS endpoint to receive real-time breaking news.
    All received news are unconditionally pushed to the user.

    Protocol assumption (placeholder — update when confirmed):
    - Server sends JSON frames:
      {
        "type": "breaking_news",
        "title": "...",
        "content": "...",
        "timestamp": 1707350400,
        "symbols": ["BTC", "ETH"],
        "url": "https://...",
        "importance": "high"
      }
    """

    def __init__(
        self,
        config: TradingConfig,
        workspace_path: Path,
        send_callback: Callable[[OutboundMessage], Awaitable[None]],
    ) -> None:
        """
        Initialize the bwenews WS client.

        Args:
            config: TradingConfig (reads bwenews ws_url from exchanges.yaml).
            workspace_path: Root workspace path (for memory files).
            send_callback: Async callable to push OutboundMessage to user.
        """
        self.config = config
        self.workspace_path = workspace_path
        self.send_callback = send_callback

        # WS 地址 (默认值, 启动时尝试从 exchanges.yaml 读取)
        self.ws_url = self._resolve_ws_url()

        # memory/trading/ 目录
        self._trading_dir = workspace_path / "memory" / "trading"
        self._trading_dir.mkdir(parents=True, exist_ok=True)

        # 运行标志
        self._running = False

        # 退避参数
        self._reconnect_delay = config.ws_reconnect_delay or _DEFAULT_RECONNECT_DELAY
        self._reconnect_max_delay = config.ws_reconnect_max_delay or _DEFAULT_RECONNECT_MAX_DELAY

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """
        Connect to bwenews WS and start listening.

        Runs indefinitely with auto-reconnect on failure.
        """
        self._running = True
        delay = self._reconnect_delay

        logger.info(f"[ws:breaking-news] connecting to {self.ws_url}")

        while self._running:
            try:
                # 延迟导入 websockets (可选依赖)
                import websockets

                # 构建宽松的 SSL 上下文 — 部分新闻 WS 服务器 SNI 配置不完善
                ssl_ctx = ssl.create_default_context()
                ssl_ctx.check_hostname = False
                ssl_ctx.verify_mode = ssl.CERT_NONE

                async with websockets.connect(
                    self.ws_url,
                    ssl=ssl_ctx,
                    ping_interval=None,   # 禁用协议级 ping, 使用应用级 ping
                    close_timeout=5,
                ) as ws:
                    logger.info("[ws:breaking-news] connected")
                    delay = self._reconnect_delay  # 重置退避

                    # 启动应用级 ping 心跳 (每 30s 发送 "ping" 文本, 与 bwenews 协议匹配)
                    async def _ping_loop() -> None:
                        try:
                            while self._running:
                                await asyncio.sleep(30)
                                await ws.send("ping")
                        except Exception:
                            pass

                    ping_task = asyncio.create_task(_ping_loop())

                    try:
                        async for raw_message in ws:
                            if not self._running:
                                break
                            # 忽略 pong 响应
                            if isinstance(raw_message, str) and raw_message.strip() == "pong":
                                continue
                            try:
                                data = json.loads(raw_message)
                                await self._handle_message(data)
                            except json.JSONDecodeError:
                                logger.warning(
                                    f"[ws:breaking-news] non-JSON message received, skipping"
                                )
                    finally:
                        ping_task.cancel()

            except asyncio.CancelledError:
                logger.info("[ws:breaking-news] task cancelled, shutting down")
                break
            except ImportError:
                logger.error(
                    "[ws:breaking-news] 'websockets' package not installed. "
                    "Install with: pip install websockets"
                )
                break
            except Exception as e:
                if not self._running:
                    break
                logger.warning(
                    f"[ws:breaking-news] connection error: {e}, "
                    f"reconnecting in {delay}s..."
                )
                await asyncio.sleep(delay)
                # 指数退避
                delay = min(delay * 2, self._reconnect_max_delay)

        logger.info("[ws:breaking-news] stopped")

    async def stop(self) -> None:
        """Gracefully stop the WS client."""
        self._running = False
        logger.info("[ws:breaking-news] stop requested")

    # ------------------------------------------------------------------
    # 消息处理
    # ------------------------------------------------------------------

    async def _handle_message(self, data: dict[str, Any]) -> None:
        """
        Process a single incoming WS message.

        Steps:
        1. Parse news fields (title, content, symbols, timestamp).
        2. 🔴 Unconditionally push to user.
        3. If news involves user's held coins → extra ⚠️ tag.
        4. Write to anomalies.jsonl with type "breaking_news".

        Args:
            data: Parsed JSON dict from the WS frame.
        """
        # --- 解析字段 (兼容多种可能的字段名) ---
        title = data.get("title") or data.get("headline") or data.get("subject") or ""
        content = data.get("content") or data.get("body") or data.get("text") or ""
        raw_ts = data.get("timestamp") or data.get("time") or data.get("ts")
        symbols = data.get("symbols") or data.get("related_symbols") or data.get("coins") or []
        url = data.get("url") or data.get("link") or ""
        importance = data.get("importance") or data.get("level") or "normal"

        # 如果 symbols 是字符串, 转为列表
        if isinstance(symbols, str):
            symbols = [s.strip() for s in symbols.split(",") if s.strip()]

        # 时间戳格式化
        ts_display = _format_timestamp(raw_ts)

        if not title and not content:
            logger.debug("[ws:breaking-news] empty message, skipping")
            return

        logger.info(f"[ws:breaking-news] received: {title[:80]}")

        # --- 检查是否涉及用户持仓币种 ---
        user_coins = self._get_user_coins()
        involved = set(s.upper() for s in symbols) & set(c.upper() for c in user_coins)
        coin_tag = ""
        if involved:
            coin_tag = f"\n⚠️ 涉及你的持仓: {', '.join(involved)}"

        # --- 构建推送消息 ---
        importance_icon = "🚨" if importance in ("high", "urgent", "critical") else "📰"
        parts = [f"{importance_icon} Breaking News"]
        if title:
            parts.append(f"\n**{title}**")
        if content:
            # 截取前 500 字符避免消息过长
            display_content = content[:500]
            if len(content) > 500:
                display_content += "…"
            parts.append(f"\n{display_content}")
        if symbols:
            parts.append(f"\n🏷️ {', '.join(symbols)}")
        if url:
            parts.append(f"\n🔗 {url}")
        if ts_display:
            parts.append(f"\n🕐 {ts_display}")
        if coin_tag:
            parts.append(coin_tag)

        message_text = "\n".join(parts)

        # --- 🔴 无条件推送 ---
        await self._push_to_user(message_text)

        # --- 写入 anomalies.jsonl ---
        await self._write_anomaly({
            "ts": datetime.now(timezone.utc).isoformat(),
            "type": "breaking_news",
            "source": "bwenews",
            "title": title,
            "content": content[:1000],
            "symbols": symbols,
            "url": url,
            "importance": importance,
            "notified": True,
            "involved_user_coins": list(involved),
        })

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------

    def _resolve_ws_url(self) -> str:
        """
        Resolve the bwenews WS URL from exchanges.yaml.

        Falls back to default URL if config not found.

        Returns:
            WebSocket URL string.
        """
        try:
            import yaml

            exchanges_path = self.workspace_path / self.config.exchanges_config_path
            if exchanges_path.exists():
                raw = yaml.safe_load(exchanges_path.read_text(encoding="utf-8"))
                bwenews_cfg = raw.get("bwenews", {})
                url = bwenews_cfg.get("ws_url")
                if url:
                    return url
        except Exception as e:
            logger.debug(f"[ws:breaking-news] failed to read ws_url from config: {e}")

        return _DEFAULT_WS_URL

    def _get_user_coins(self) -> list[str]:
        """
        Read user's current held coin symbols from positions.json.

        Returns:
            List of coin base symbols (e.g. ["BTC", "SOL"]).
        """
        coins: set[str] = set()

        # 1) positions.json (active holdings)
        path = self._trading_dir / "positions.json"
        try:
            if path.exists():
                data = json.loads(path.read_text(encoding="utf-8"))
                for exchange_data in data.get("exchanges", {}).values():
                    for pos in exchange_data.get("contracts", []):
                        base = extract_coin(pos.get("symbol", ""))
                        if base:
                            coins.add(base)
                    for spot in exchange_data.get("spot", []):
                        base = extract_coin(spot.get("symbol", ""))
                        if base:
                            coins.add(base)
        except (json.JSONDecodeError, OSError) as e:
            logger.debug(f"[ws:breaking-news] failed to read positions.json: {e}")
        
        # 2) orders.json (open orders — include entry orders not yet filled)
        orders_path = self._trading_dir / "orders.json"
        try:
            if orders_path.exists():
                odata = json.loads(orders_path.read_text(encoding="utf-8"))
                exchanges = odata.get("exchanges", {}) if isinstance(odata, dict) else {}
                if isinstance(exchanges, dict):
                    for ex_data in exchanges.values():
                        if not isinstance(ex_data, dict):
                            continue
                        for orders in ex_data.values():
                            if not isinstance(orders, list):
                                continue
                            for o in orders:
                                if isinstance(o, dict):
                                    base = extract_coin(str(o.get("symbol") or ""))
                                    if base:
                                        coins.add(base)
        except (json.JSONDecodeError, OSError) as e:
            logger.debug(f"[ws:breaking-news] failed to read orders.json: {e}")

        # 3) watch_scope.json (ephemeral watch coins)
        try:
            coins |= get_active_watch_coins(self.workspace_path)
        except Exception:
            pass

        return list(coins)

    async def _push_to_user(self, content: str) -> None:
        """
        Send breaking news notification to every registered principal route.

        Falls back to internal ws channel when no routes exist.

        Args:
            content: Formatted message string.
        """
        try:
            routes = load_all_routes()
            if routes:
                for _pid, route in routes.items():
                    msg = OutboundMessage(
                        channel=route.channel,
                        chat_id=route.chat_id,
                        content=content,
                    )
                    await self.send_callback(msg)
                return

            msg = OutboundMessage(
                channel="__ws_push__",
                chat_id="__active__",
                content=content,
            )
            await self.send_callback(msg)
        except Exception as e:
            logger.error(f"[ws:breaking-news] failed to push: {e}")

    async def _write_anomaly(self, record: dict[str, Any]) -> None:
        """
        Append a record to anomalies.jsonl.

        Args:
            record: Anomaly event dict.
        """
        path = self._trading_dir / "anomalies.jsonl"
        try:
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as e:
            logger.error(f"[ws:breaking-news] failed to write anomalies.jsonl: {e}")


# ---------------------------------------------------------------------------
# 模块级工具函数
# ---------------------------------------------------------------------------

def _format_timestamp(raw_ts: Any) -> str:
    """
    Format a raw timestamp value into a human-readable string.

    Handles: int (unix epoch), float, ISO string, or None.

    Returns:
        Formatted datetime string or empty string.
    """
    if raw_ts is None:
        return ""
    try:
        if isinstance(raw_ts, (int, float)):
            # 如果是毫秒级时间戳
            if raw_ts > 1e12:
                raw_ts = raw_ts / 1000
            dt = datetime.fromtimestamp(raw_ts, tz=timezone.utc)
            return dt.strftime("%Y-%m-%d %H:%M UTC")
        if isinstance(raw_ts, str):
            return raw_ts[:19]  # 截取到秒
    except (ValueError, OSError):
        pass
    return str(raw_ts)
