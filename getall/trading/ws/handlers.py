# 账户事件处理器: 仓位/订单/成交/余额变化处理
# 核心原则:
#   - 所有事件 → 先静默写入 memory 文件 (positions.json / trades.jsonl)
#   - 仅重大事件 → 主动通知用户 (新开仓/平仓/爆仓风险/订单被拒)
#   - 用户提问时 → Agent 从 memory 文件读取最新数据 (Tier 2 按需加载)
#
# 通知策略:
#   🔴 立即通知: 新仓位 / 仓位消失 / 爆仓距离<15% / 保证金率危险 / 订单被拒
#   🟡 静默更新: 仓位盈亏波动 / 余额正常变动 / 订单正常成交 / 加减仓

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Awaitable

from loguru import logger

from getall.bus.events import InboundMessage, OutboundMessage
from getall.routing import load_last_route
from getall.trading.watch_scope import extract_coin, upsert_watch_coins
from getall.utils.atomic_io import get_atomic_writer


# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

# 爆仓距离预警阈值 (百分比)
_LIQUIDATION_WARN_PCT = 15.0

# 保证金率安全阈值 (低于此值紧急通知)
_MARGIN_RATIO_UNSAFE = 0.07

# 订单被拒状态
_ORDER_REJECTED = "rejected"

_AGENT_EVENT_TTL_S = 300  # 5 min de-dupe window for repeated events


class WSEventHandlers:
    """
    WebSocket event handlers for account streams.

    Receives raw data from ccxt.pro watchers, compares with cached state,
    decides whether to notify the user (🔴) or silently update memory (🟡).
    """

    def __init__(
        self,
        workspace_path: Path,
        send_callback: Callable[[OutboundMessage], Awaitable[None]],
        inbound_callback: Callable[[InboundMessage], Awaitable[None]] | None = None,
        *,
        debug_console_push: bool = False,
    ) -> None:
        """
        Initialize event handlers.

        Args:
            workspace_path: Root workspace path (memory files live under memory/trading/).
            send_callback: Async callable to send OutboundMessage to user.
        """
        self.workspace_path = workspace_path
        self.send_callback = send_callback
        self.inbound_callback = inbound_callback
        self._debug_console_push = debug_console_push

        # memory/trading/ 目录
        self._trading_dir = workspace_path / "memory" / "trading"
        self._trading_dir.mkdir(parents=True, exist_ok=True)

        # 内存缓存: 上一次已知仓位快照 {exchange: {symbol: position_dict}}
        self._prev_positions: dict[str, dict[str, Any]] = {}

        # 内存缓存: 余额
        self._balance_cache: dict[str, dict[str, Any]] = {}

        # 内存缓存: 全量挂单快照 (仅保存 OPEN 状态)
        # {exchange: {market_type: {order_id: order_min_dict}}}
        self._open_orders: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}

        # Agent-event de-dupe (key -> last_ts)
        self._agent_event_last_ts: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Agent Event Bridge (WS -> AgentLoop system message)
    # ------------------------------------------------------------------

    async def _emit_agent_event(self, *, key: str, content: str, sender_id: str = "ws:account") -> None:
        """
        Send a system message into the agent loop so it can proactively analyze and message the user.

        Uses last-route routing (feishu/telegram/etc.) to deliver the agent's response.
        """
        if not self.inbound_callback:
            return

        # De-dupe repeated events (especially risk warnings)
        now_ts = datetime.now(timezone.utc).timestamp()
        last = self._agent_event_last_ts.get(key)
        if last is not None and now_ts - last < _AGENT_EVENT_TTL_S:
            return
        self._agent_event_last_ts[key] = now_ts

        route = load_last_route()
        if route:
            origin_channel = route.channel
            origin_chat_id = route.chat_id
        elif self._debug_console_push:
            # Debug mode: allow proactive analysis to show up in gateway console
            # even before any real external channel is configured.
            origin_channel = "__ws_push__"
            origin_chat_id = "__active__"
            logger.debug("[ws:account] no last-route, using __ws_push__ (debug_console_push)")
        else:
            logger.debug("[ws:account] no last-route, skip agent event")
            return

        msg = InboundMessage(
            channel="system",
            sender_id=sender_id,
            chat_id=f"{origin_channel}:{origin_chat_id}",
            content=content,
        )
        try:
            await self.inbound_callback(msg)
        except Exception as e:
            logger.debug(f"[ws:account] failed to emit agent event: {e}")

    async def _notify(self, content: str) -> None:
        """
        Send a notification to the user via send_callback.

        Prefer last-route delivery when available, otherwise fallback to internal ws channel.
        """
        try:
            route = load_last_route()
            if route:
                msg = OutboundMessage(
                    channel=route.channel,
                    chat_id=route.chat_id,
                    content=content,
                )
            else:
                msg = OutboundMessage(
                    channel="__ws_push__",
                    chat_id="__active__",
                    content=content,
                )
            await self.send_callback(msg)
        except Exception as e:
            logger.error(f"Failed to push notification: {e}")

    # ------------------------------------------------------------------
    # 仓位变化 (核心: 新开仓/平仓检测)
    # ------------------------------------------------------------------

    async def on_positions_update(self, exchange: str, positions: list[dict], *, notify: bool = True) -> None:
        """
        Handle position updates from watchPositions().

        Comparison logic:
        - New symbol in positions      → 🔴 NOTIFY (new position detected)
        - Symbol disappeared           → 🔴 NOTIFY (position closed, P&L summary)
        - Liquidation risk (< 15%)     → 🔴 URGENT NOTIFY
        - Normal PnL / size changes    → 🟡 SILENT (update positions.json)

        Args:
            exchange: Exchange name (e.g. "binance").
            positions: List of ccxt position dicts.
        """
        prev = self._prev_positions.get(exchange, {})

        # 过滤有效仓位 (合约数量 != 0)
        active: dict[str, dict] = {}
        for pos in positions:
            contracts = float(pos.get("contracts", 0) or 0)
            if contracts != 0:
                symbol = pos.get("symbol", "unknown")
                merged = dict(pos)
                if symbol in prev:
                    old = prev[symbol]
                    # ccxt watchPositions for Binance Futures often omits leverage; preserve last-known values.
                    for k in ("leverage", "marginMode", "liquidationPrice"):
                        if merged.get(k) in (None, 0, "", "N/A"):
                            if old.get(k) not in (None, 0, "", "N/A"):
                                merged[k] = old.get(k)
                active[symbol] = merged

        if not notify:
            # Seed baseline without any alerts/trade journaling.
            self._prev_positions[exchange] = active
            await self._save_positions()
            logger.debug(f"[ws:account] positions.json seeded for {exchange} ({len(active)} active)")
            return

        # ── Detect structural changes (new/close/size/leverage) ──
        new_symbols = set(active.keys()) - set(prev.keys())
        closed_symbols = set(prev.keys()) - set(active.keys())

        # size / leverage changes (ignore markPrice / unrealizedPnL noise)
        changed_symbols: list[dict[str, Any]] = []
        for sym in set(active.keys()) & set(prev.keys()):
            old_pos = prev.get(sym, {}) or {}
            new_pos = active.get(sym, {}) or {}
            try:
                old_contracts = float(old_pos.get("contracts", 0) or 0)
                new_contracts = float(new_pos.get("contracts", 0) or 0)
            except (TypeError, ValueError):
                continue
            if abs(new_contracts - old_contracts) > 1e-12:
                changed_symbols.append({
                    "symbol": sym,
                    "old_contracts": old_contracts,
                    "new_contracts": new_contracts,
                    "side": new_pos.get("side") or old_pos.get("side"),
                    "leverage": new_pos.get("leverage") or old_pos.get("leverage"),
                    "entryPrice": new_pos.get("entryPrice") or old_pos.get("entryPrice"),
                    "markPrice": new_pos.get("markPrice") or old_pos.get("markPrice"),
                    "liquidationPrice": new_pos.get("liquidationPrice") or old_pos.get("liquidationPrice"),
                })
                continue

            # leverage change (when contracts unchanged)
            old_lev = old_pos.get("leverage")
            new_lev = new_pos.get("leverage")
            try:
                if old_lev is not None and new_lev is not None and float(old_lev) != float(new_lev):
                    changed_symbols.append({
                        "symbol": sym,
                        "old_contracts": old_contracts,
                        "new_contracts": new_contracts,
                        "side": new_pos.get("side") or old_pos.get("side"),
                        "old_leverage": old_lev,
                        "new_leverage": new_lev,
                        "entryPrice": new_pos.get("entryPrice") or old_pos.get("entryPrice"),
                        "markPrice": new_pos.get("markPrice") or old_pos.get("markPrice"),
                        "liquidationPrice": new_pos.get("liquidationPrice") or old_pos.get("liquidationPrice"),
                    })
            except (TypeError, ValueError):
                pass

        # ── Emit proactive analysis events to agent (per-symbol) ──
        for sym in sorted(new_symbols):
            pos = active[sym]
            coin = extract_coin(sym)
            if coin:
                upsert_watch_coins(self.workspace_path, [coin], source="ws:new_position")
            payload = {
                "exchange": exchange,
                "symbol": sym,
                "coin": coin,
                "side": pos.get("side"),
                "contracts": pos.get("contracts"),
                "leverage": pos.get("leverage"),
                "entryPrice": pos.get("entryPrice"),
                "markPrice": pos.get("markPrice"),
                "liquidationPrice": pos.get("liquidationPrice"),
            }
            prompt = (
                "SYSTEM_EVENT: NEW_POSITION\n"
                "A new contract position just appeared (user may have opened it manually).\n"
                "Proactively message the user with:\n"
                "- What changed (symbol/side/size/leverage/entry).\n"
                "- Quick risk review (liq distance, leverage, whether TP/SL orders exist).\n"
                "- If TP/SL missing or unreasonable, suggest concrete trigger levels and offer a preview (预演, 模拟下单（Paper Trade）) setup.\n"
                "- Tell the user we've started monitoring this coin (orders/positions/trades realtime + anomaly/news/breaking-news).\n"
                "Keep it concise (<= 12 lines). Prefer reading positions.json/orders.json for context.\n\n"
                f"Payload:\n{json.dumps(payload, ensure_ascii=False)}"
            )
            await self._emit_agent_event(key=f"new_position:{exchange}:{sym}", content=prompt)

        for ch in changed_symbols:
            sym = str(ch.get("symbol") or "")
            if not sym:
                continue
            coin = extract_coin(sym)
            if coin:
                upsert_watch_coins(self.workspace_path, [coin], source="ws:position_change")
            payload = {
                "exchange": exchange,
                "coin": coin,
                **ch,
            }
            prompt = (
                "SYSTEM_EVENT: POSITION_CHANGED\n"
                "A contract position's size/leverage just changed (likely a fill/add/reduce/partial close).\n"
                "Proactively message the user with:\n"
                "- Describe the change (old->new, delta, add vs reduce).\n"
                "- Re-check risk and whether TP/SL coverage still makes sense.\n"
                "- If TP/SL missing, propose triggers; if present, judge reasonableness.\n"
                "- Mention monitoring is active for this coin.\n"
                "Keep it short and actionable.\n\n"
                f"Payload:\n{json.dumps(payload, ensure_ascii=False)}"
            )
            await self._emit_agent_event(
                key=f"position_changed:{exchange}:{sym}:{ch.get('old_contracts')}->{ch.get('new_contracts')}:{ch.get('old_leverage')}->{ch.get('new_leverage')}",
                content=prompt,
            )

        for sym in sorted(closed_symbols):
            old_pos = prev[sym]
            logger.info(f"[ws:account] Position closed: {exchange} {sym}")
            pnl = old_pos.get("unrealizedPnl", 0) or 0
            # 记录到 trades.jsonl
            await self._append_trade({
                "ts": _now_iso(),
                "symbol": sym,
                "exchange": exchange,
                "side": old_pos.get("side", "unknown"),
                "action": "close",
                "pnl": pnl,
                "entry_price": old_pos.get("entryPrice"),
                "source": "ws:account",
                "status": "closed",
            })
            coin = extract_coin(sym)
            if coin:
                upsert_watch_coins(self.workspace_path, [coin], source="ws:position_closed")
            payload = {
                "exchange": exchange,
                "symbol": sym,
                "coin": coin,
                "side": old_pos.get("side"),
                "contracts": old_pos.get("contracts"),
                "entryPrice": old_pos.get("entryPrice"),
                "lastMarkPrice": old_pos.get("markPrice"),
                "pnl": pnl,
            }
            prompt = (
                "SYSTEM_EVENT: POSITION_CLOSED\n"
                "A contract position just disappeared (closed or liquidated).\n"
                "Proactively message the user with a concise close summary (P&L, side, symbol) "
                "and 1-2 quick takeaways.\n\n"
                f"Payload:\n{json.dumps(payload, ensure_ascii=False)}"
            )
            await self._emit_agent_event(key=f"position_closed:{exchange}:{sym}:{old_pos.get('timestamp')}", content=prompt)

        # --- 检测爆仓风险 ---
        for sym, pos in active.items():
            liq_distance = self._calc_liquidation_distance(pos)
            if liq_distance is not None and liq_distance < _LIQUIDATION_WARN_PCT:
                logger.warning(
                    f"[ws:account] Liquidation risk! {exchange} {sym} "
                    f"distance={liq_distance:.1f}%"
                )
                await self._notify(
                    f"🚨 爆仓预警!\n\n"
                    f"{sym} | {exchange}\n"
                    f"当前价: ${_fmt_price(pos.get('markPrice'))}\n"
                    f"爆仓价: ${_fmt_price(pos.get('liquidationPrice'))}\n"
                    f"距爆仓: {liq_distance:.1f}%\n\n"
                    f"⚠️ 请立即检查仓位风险!"
                )

            # NOTE: marginRatio 是账户级别指标，不是仓位级别
            # 账户保证金率检查在 on_balance_update() 中处理

        # --- 静默更新: 覆盖 positions.json ---
        self._prev_positions[exchange] = active
        await self._save_positions()
        logger.debug(f"[ws:account] positions.json updated for {exchange} ({len(active)} active)")

    # ------------------------------------------------------------------
    # 订单变化
    # ------------------------------------------------------------------

    async def on_orders_update(
        self,
        exchange: str,
        orders: list[dict],
        market_type: str = "swap",
        *,
        notify_agent: bool = True,
    ) -> None:
        """
        Handle order updates from watchOrders().

        - rejected → 🔴 NOTIFY
        - filled   → 🟡 SILENT (record to trades.jsonl)
        - others   → 🟡 SILENT

        Args:
            exchange: Exchange name.
            orders: List of ccxt order dicts.
            market_type: ccxt market type for this stream (spot / swap / delivery / etc).
        """
        # Ensure cache buckets exist
        self._open_orders.setdefault(exchange, {}).setdefault(market_type, {})
        bucket = self._open_orders[exchange][market_type]
        changed = False

        for order in orders:
            status = (order.get("status") or "").lower()
            symbol = order.get("symbol", "unknown")
            side = order.get("side", "?")
            order_type = order.get("type", "?")
            price = order.get("price")
            amount = order.get("amount")
            order_id = order.get("id")
            info = order.get("info") if isinstance(order, dict) else None

            # --- Maintain OPEN orders cache ---
            # ccxt unified statuses: "open" / "closed" / "canceled" / "rejected"
            if order_id:
                if status == "open":
                    trigger_price = _extract_trigger_price(order)
                    raw_type = _extract_raw_order_type(order)
                    reduce_only = _coerce_bool(order.get("reduceOnly"))
                    if reduce_only is None and isinstance(info, dict):
                        reduce_only = _coerce_bool(info.get("reduceOnly"))
                    close_position = None
                    position_side = None
                    client_order_id = None
                    time_in_force = None
                    if isinstance(info, dict):
                        close_position = _coerce_bool(info.get("closePosition"))
                        position_side = info.get("positionSide") or info.get("ps")
                        client_order_id = info.get("clientOrderId")
                        time_in_force = info.get("timeInForce")
                    position_side = order.get("positionSide") or position_side
                    client_order_id = order.get("clientOrderId") or client_order_id
                    time_in_force = order.get("timeInForce") or time_in_force

                    minimal = {
                        "id": order_id,
                        "symbol": symbol,
                        "market_type": market_type,
                        "side": side,
                        "type": order_type,
                        "raw_type": raw_type,
                        "price": price,
                        "trigger_price": trigger_price,
                        "amount": amount,
                        "remaining": order.get("remaining"),
                        "status": status,
                        "time_in_force": time_in_force,
                        "reduce_only": reduce_only,
                        "close_position": close_position,
                        "position_side": position_side,
                        "client_order_id": client_order_id,
                        "timestamp": order.get("timestamp"),
                        "datetime": order.get("datetime"),
                        "updated_at": _now_iso(),
                    }
                    prev = bucket.get(str(order_id))
                    if prev != minimal:
                        bucket[str(order_id)] = minimal
                        changed = True

                    # ── Proactive analysis trigger (NEW open orders only) ──
                    if notify_agent and prev is None:
                        # Skip orders we placed ourselves (best-effort tag).
                        cid = str(client_order_id or "")
                        if cid.startswith("getall_"):
                            continue

                        coin = extract_coin(symbol)
                        if coin:
                            upsert_watch_coins(
                                self.workspace_path,
                                [coin],
                                source="ws:order_open",
                            )

                        raw = str(raw_type or "").upper()
                        reduce_flag = (reduce_only is True) or (close_position is True)
                        kind = "entry"
                        if reduce_flag:
                            kind = "exit"
                        tp_sl = None
                        if "TAKE_PROFIT" in raw:
                            tp_sl = "TP"
                        elif "STOP_LOSS" in raw or raw.startswith("STOP") or "TRAILING_STOP" in raw:
                            tp_sl = "SL"

                        payload = {
                            "exchange": exchange,
                            "market_type": market_type,
                            "kind": kind,
                            "tp_sl": tp_sl,
                            "coin": coin,
                            "order": minimal,
                        }

                        prompt = (
                            "SYSTEM_EVENT: NEW_OPEN_ORDER\n"
                            "You detected a new user-created OPEN order on the exchange.\n"
                            "Your task: proactively message the user with a concise analysis.\n\n"
                            "Requirements:\n"
                            "- Summarize what the order is (entry vs TP/SL, side, amount, trigger/price).\n"
                            "- If it's TP/SL: judge whether trigger price is reasonable vs entry/mark; "
                            "if missing TP or SL for this coin, suggest levels and offer a preview (预演, 模拟下单（Paper Trade）) plan.\n"
                            "- If it's an entry order: assess risk (position sizing/leverage if known), "
                            "and recommend SL/TP if missing.\n"
                            "- Tell the user we've started monitoring this coin: "
                            "ws:account (orders/trades/positions realtime) + "
                            "cron anomaly-scan (OI/funding/liquidations/whales) + "
                            "cron news-feed + breaking-news.\n"
                            "- Prefer to read `memory/trading/positions.json` and `memory/trading/orders.json` for context.\n"
                            "- Keep it short and actionable (<= 12 lines).\n\n"
                            f"Payload:\n{json.dumps(payload, ensure_ascii=False)}"
                        )
                        await self._emit_agent_event(
                            key=f"new_open_order:{exchange}:{market_type}:{order_id}",
                            content=prompt,
                        )
                elif status in ("closed", "canceled", "cancelled", "filled", "rejected", "expired"):
                    if str(order_id) in bucket:
                        bucket.pop(str(order_id), None)
                        changed = True

            if status == _ORDER_REJECTED:
                # 🔴 订单被拒 → 通知用户
                reason = order.get("info", {}).get("msg", "unknown reason")
                logger.warning(f"[ws:account] Order rejected: {exchange} {symbol} — {reason}")
                await self._notify(
                    f"❌ 订单被拒\n\n"
                    f"{symbol} {side} {order_type}\n"
                    f"数量: {amount} | 价格: ${_fmt_price(price)}\n"
                    f"原因: {reason}\n"
                    f"交易所: {exchange}"
                )
            elif status == "closed" or status == "filled":
                # 🟡 正常成交 → 静默记录
                logger.info(f"[ws:account] Order filled: {exchange} {symbol} {side} {amount}")
                await self._append_trade({
                    "ts": _now_iso(),
                    "symbol": symbol,
                    "exchange": exchange,
                    "side": side,
                    "type": order_type,
                    "price": price,
                    "amount": amount,
                    "order_id": order.get("id"),
                    "action": "fill",
                    "source": "ws:account",
                    "status": "filled",
                })
            else:
                # 🟡 其他状态 (open / canceled 等) — 静默
                logger.debug(f"[ws:account] Order {status}: {exchange} {symbol}")

        # --- Persist full open-orders snapshot (changed only) ---
        if changed:
            await self._save_open_orders()

    # ------------------------------------------------------------------
    # 成交记录
    # ------------------------------------------------------------------

    async def on_trades_update(self, exchange: str, trades: list[dict]) -> None:
        """
        Handle trade updates from watchMyTrades().

        All trades → 🟡 SILENT: append to trades.jsonl.

        Args:
            exchange: Exchange name.
            trades: List of ccxt trade dicts.
        """
        for trade in trades:
            logger.debug(
                f"[ws:account] Trade: {exchange} {trade.get('symbol')} "
                f"{trade.get('side')} {trade.get('amount')}@{trade.get('price')}"
            )
            await self._append_trade({
                "ts": trade.get("datetime") or _now_iso(),
                "symbol": trade.get("symbol", "unknown"),
                "exchange": exchange,
                "side": trade.get("side"),
                "price": trade.get("price"),
                "amount": trade.get("amount"),
                "cost": trade.get("cost"),
                "fee": trade.get("fee"),
                "trade_id": trade.get("id"),
                "order_id": trade.get("order"),
                "action": "trade",
                "source": "ws:account",
            })

    # ------------------------------------------------------------------
    # 余额变动
    # ------------------------------------------------------------------

    async def on_balance_update(self, exchange: str, balance: dict) -> None:
        """
        Handle balance updates from watchBalance().

        - Mostly 🟡 SILENT: update in-memory cache.
        - Margin ratio unsafe → 🔴 NOTIFY.

        Args:
            exchange: Exchange name.
            balance: ccxt balance dict (contains 'total', 'free', 'used' etc.).
        """
        self._balance_cache[exchange] = balance
        logger.debug(f"[ws:account] Balance updated for {exchange}")

        # 检查保证金率 (若交易所返回)
        # NOTE: Binance watchBalance 不返回 marginRatio，需要 REST API
        info = balance.get("info", {})
        margin_ratio = info.get("marginRatio") or info.get("totalMarginRatio")

        if margin_ratio is not None:
            try:
                ratio = float(margin_ratio)
                if ratio < _MARGIN_RATIO_UNSAFE:
                    total_balance = balance.get("total", {}).get("USDT", "?")
                    await self._notify(
                        f"🚨 账户保证金率危险!\n\n"
                        f"交易所: {exchange}\n"
                        f"保证金率: {ratio * 100:.1f}%\n"
                        f"总余额: {total_balance} USDT\n"
                        f"⚠️ 请立即检查风险敞口!"
                    )
            except (ValueError, TypeError):
                pass

    # ------------------------------------------------------------------
    # 辅助方法: orders.json I/O
    # ------------------------------------------------------------------

    def _load_open_orders(self) -> dict[str, Any]:
        """
        Read orders.json from disk.

        Returns:
            Parsed JSON dict, or empty structure if file missing / corrupt.
        """
        path = self._trading_dir / "orders.json"
        if not path.exists():
            return {"last_sync": None, "exchanges": {}}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to read orders.json: {e}")
            return {"last_sync": None, "exchanges": {}}

    async def _save_open_orders(self) -> None:
        """
        Write current OPEN orders snapshot to orders.json atomically.

        Structure:
          {
            "last_sync": "...",
            "exchanges": {
              "binance": {
                "swap": [ ... ],
                "spot": [ ... ]
              }
            }
          }
        """
        data = self._load_open_orders()
        data["last_sync"] = _now_iso()
        data.setdefault("exchanges", {})

        # Merge from in-memory cache
        for exchange_name, by_type in self._open_orders.items():
            ex_data = data["exchanges"].setdefault(exchange_name, {})
            for mtype, orders_map in by_type.items():
                # stable sort for deterministic diffs
                orders_list = list(orders_map.values())
                orders_list.sort(key=lambda o: (str(o.get("symbol", "")), str(o.get("id", ""))))
                ex_data[mtype] = orders_list

        path = self._trading_dir / "orders.json"
        writer = get_atomic_writer()
        success = await writer.write_json(path, data)
        if success:
            logger.debug(
                f"[ws:account] orders.json updated ({sum(len(v) for ex in self._open_orders.values() for v in ex.values())} open)"
            )
        else:
            logger.error("Failed to write orders.json atomically")

    # ------------------------------------------------------------------
    # 辅助方法: 文件 I/O
    # ------------------------------------------------------------------

    def _load_positions(self) -> dict[str, Any]:
        """
        Read positions.json from disk.

        Returns:
            Parsed JSON dict, or empty structure if file missing / corrupt.
        """
        path = self._trading_dir / "positions.json"
        if not path.exists():
            return {"last_sync": None, "exchanges": {}}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to read positions.json: {e}")
            return {"last_sync": None, "exchanges": {}}

    async def _save_positions(self) -> None:
        """
        Write current position state to positions.json atomically.

        Merges all exchanges from ``_prev_positions`` cache into the file.
        """
        data = self._load_positions()
        data["last_sync"] = _now_iso()

        for exchange_name, symbols in self._prev_positions.items():
            contracts = []
            for sym, pos in symbols.items():
                contracts.append({
                    "symbol": sym,
                    "side": pos.get("side"),
                    "amount": pos.get("contracts"),
                    "entry_price": pos.get("entryPrice"),
                    "mark_price": pos.get("markPrice"),
                    "leverage": pos.get("leverage"),
                    "liquidation_price": pos.get("liquidationPrice"),
                    "unrealized_pnl": pos.get("unrealizedPnl"),
                    "margin_ratio": pos.get("marginRatio"),
                    "percentage": pos.get("percentage"),
                })

            if exchange_name not in data.get("exchanges", {}):
                data.setdefault("exchanges", {})[exchange_name] = {}
            data["exchanges"][exchange_name]["contracts"] = contracts

        path = self._trading_dir / "positions.json"
        writer = get_atomic_writer()
        success = await writer.write_json(path, data)
        if not success:
            logger.error("Failed to write positions.json atomically")

    async def _append_trade(self, trade: dict[str, Any]) -> None:
        """
        Append a single trade record to trades.jsonl atomically.

        Args:
            trade: Trade dict to persist.
        """
        path = self._trading_dir / "trades.jsonl"
        writer = get_atomic_writer()
        success = await writer.append_json_line(path, trade)
        if not success:
            logger.error("Failed to append to trades.jsonl atomically")

    # ------------------------------------------------------------------
    # 辅助方法: 通知策略
    # ------------------------------------------------------------------

    def _should_notify(self, event_type: str, data: dict[str, Any]) -> bool:
        """
        Check notification policy.

        Args:
            event_type: One of "new_position", "close_position", "liquidation_risk",
                        "margin_unsafe", "order_rejected".
            data: Event-specific data dict.

        Returns:
            True if the event warrants a push notification.
        """
        # 以下事件类型始终通知
        always_notify = {
            "new_position",
            "close_position",
            "liquidation_risk",
            "margin_unsafe",
            "order_rejected",
        }
        return event_type in always_notify

    # ------------------------------------------------------------------
    # 辅助方法: 计算
    # ------------------------------------------------------------------

    @staticmethod
    def _calc_liquidation_distance(pos: dict) -> float | None:
        """
        Calculate percentage distance to liquidation price.

        Returns:
            Distance as positive percentage, or None if data insufficient.
        """
        mark = pos.get("markPrice")
        liq = pos.get("liquidationPrice")
        if mark is None or liq is None:
            return None
        try:
            mark_f = float(mark)
            liq_f = float(liq)
            if mark_f == 0:
                return None
            return abs(mark_f - liq_f) / mark_f * 100
        except (ValueError, TypeError):
            return None


# ---------------------------------------------------------------------------
# 模块级工具函数
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    """Return current UTC timestamp in ISO format."""
    return datetime.now(timezone.utc).isoformat()


def _fmt_price(price: Any) -> str:
    """Format a price value for display, handling None gracefully."""
    if price is None:
        return "N/A"
    try:
        p = float(price)
        # 大价格用逗号分隔
        if p >= 1:
            return f"{p:,.2f}"
        # 小价格保留更多小数
        return f"{p:.6f}"
    except (ValueError, TypeError):
        return str(price)


def _coerce_bool(value: Any) -> bool | None:
    """
    Best-effort bool coercion for exchange fields.

    Binance often returns booleans as "true"/"false" strings in raw `info`.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("true", "1", "yes"):
            return True
        if v in ("false", "0", "no"):
            return False
    return None


def _coerce_float(value: Any) -> float | None:
    """Best-effort float coercion for numeric exchange fields."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_trigger_price(order: dict[str, Any]) -> float | None:
    """
    Extract trigger/stop price from a ccxt order dict.

    For Binance Futures conditional orders, `price` can be null (STOP_MARKET / TAKE_PROFIT_MARKET),
    while `stopPrice` (aka trigger price) is present either on the unified order or in `info`.
    """
    if not isinstance(order, dict):
        return None

    for k in ("triggerPrice", "stopPrice"):
        v = order.get(k)
        out = _coerce_float(v)
        if out is not None:
            return out

    info = order.get("info")
    if isinstance(info, dict):
        for k in ("triggerPrice", "stopPrice"):
            out = _coerce_float(info.get(k))
            if out is not None:
                return out

    return None


def _extract_raw_order_type(order: dict[str, Any]) -> str | None:
    """Extract raw exchange order type (e.g., Binance: STOP_MARKET/TAKE_PROFIT_MARKET)."""
    if not isinstance(order, dict):
        return None
    info = order.get("info")
    if isinstance(info, dict):
        for k in ("type", "origType", "orderType"):
            v = info.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
    # fallback: some exchanges expose it on the unified object
    v2 = order.get("rawType") or order.get("raw_type")
    if isinstance(v2, str) and v2.strip():
        return v2.strip()
    return None
