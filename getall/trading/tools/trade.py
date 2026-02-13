"""交易执行工具 - 下单、止盈止损、取消订单、模拟下单（Paper Trade）预览"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

from getall.agent.tools.base import Tool
from getall.trading.action_log import TradingActionLog
from getall.trading.data.hub import DataHub


class TradeTool(Tool):
    """执行加密货币交易操作的工具，默认 dry_run 模式，需明确确认才执行真实交易。"""

    def __init__(self, hub: DataHub):
        # 注入 DataHub，统一管理交易所连接
        self.hub = hub
        # 交易操作日志（仅记录真实交易）
        trading_dir = Path.home() / ".getall" / "workspace" / "trading"
        trading_dir.mkdir(parents=True, exist_ok=True)
        self._action_log = TradingActionLog(trading_dir)

    @property
    def name(self) -> str:
        return "trade"

    @property
    def description(self) -> str:
        return (
            "Execute trades: place orders (market/limit), set stop-loss/take-profit, "
            "cancel orders. Supports Paper Trade (模拟下单) preview before execution.\n\n"
            "CRITICAL - Required parameters by action:\n"
            "• place_order / dry_run: symbol, side, type, amount (+ price for limit orders)\n"
            "• cancel_order: symbol, order_id\n"
            "• set_stop_loss / set_take_profit: symbol, stop_loss or take_profit\n\n"
            "IMPORTANT: Collect ALL required parameters in ONE call. Do NOT make multiple calls "
            "asking for one parameter at a time. If user says 'buy BTC 2000 USDT', infer:\n"
            "  side=buy, symbol=BTC/USDT, type=market, amount=calculated from 2000 USDT.\n\n"
            "CLOSING POSITIONS: When user wants to close/平仓, FIRST call portfolio(action='positions') "
            "to show all holdings, then execute the close order. For LONG positions, use side=sell to close. "
            "For SHORT positions, use side=buy to close."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "The trade action to perform",
                    "enum": [
                        "place_order",     # 下单（市价/限价）
                        "cancel_order",    # 取消订单
                        "set_stop_loss",   # 设置止损
                        "set_take_profit", # 设置止盈
                        "dry_run",         # 模拟预览
                    ],
                },
                "symbol": {
                    "type": "string",
                    "description": "Trading pair, e.g. 'BTC/USDT:USDT' for futures, 'BTC/USDT' for spot",
                },
                "side": {
                    "type": "string",
                    "description": "REQUIRED for place_order/dry_run: Order side (buy=long, sell=short for futures)",
                    "enum": ["buy", "sell"],
                },
                "type": {
                    "type": "string",
                    "description": "REQUIRED for place_order/dry_run: Order type (market=instant execution, limit=specific price)",
                    "enum": ["market", "limit"],
                },
                "amount": {
                    "type": "number",
                    "description": (
                        "REQUIRED for place_order/dry_run: Order amount in base currency (e.g., 0.5 BTC). "
                        "For USDT-based sizing, calculate: amount = USDT_value / current_price. "
                        "Optional for set_stop_loss/set_take_profit (will infer from open position)."
                    ),
                    "minimum": 0,
                },
                "price": {
                    "type": "number",
                    "description": "Limit price. Required for limit orders.",
                },
                "leverage": {
                    "type": "integer",
                    "description": "Leverage multiplier for futures trading",
                    "minimum": 1,
                    "maximum": 125,
                },
                "stop_loss": {
                    "type": "number",
                    "description": "Stop-loss trigger price",
                },
                "take_profit": {
                    "type": "number",
                    "description": "Take-profit trigger price",
                },
                "order_id": {
                    "type": "string",
                    "description": "Order ID for cancel_order action",
                },
                "exchange": {
                    "type": "string",
                    "description": "Exchange name. If omitted, uses default.",
                },
                "paper_trade": {
                    "type": "boolean",
                    "description": "Paper Trade (simulate order). Default: true. Set to false only after explicit confirmation.",
                },
                "dry_run": {
                    "type": "boolean",
                    "description": "(Deprecated alias) Same as paper_trade. Prefer paper_trade.",
                },
            },
            "required": ["action", "symbol"],
        }

    async def execute(self, **kwargs: Any) -> str:
        action: str = kwargs["action"]
        symbol: str = kwargs["symbol"]
        paper_trade = kwargs.get("paper_trade")
        dry_run: bool = kwargs.get("dry_run", True)
        if paper_trade is not None:
            dry_run = bool(paper_trade)
        exchange: str | None = kwargs.get("exchange")

        try:
            handlers = {
                "place_order": self._place_order,
                "cancel_order": self._cancel_order,
                "set_stop_loss": self._set_stop_loss,
                "set_take_profit": self._set_take_profit,
                "dry_run": self._dry_run,
            }
            handler = handlers.get(action)
            if handler is None:
                return f"Error: unknown action '{action}'"

            # dry_run action 始终走预览
            if action == "dry_run":
                dry_run = True

            return await handler(
                symbol=symbol,
                dry_run=dry_run,
                exchange=exchange,
                **{k: v for k, v in kwargs.items()
                   if k not in ("action", "symbol", "dry_run", "paper_trade", "exchange")},
            )
        except Exception as e:
            return f"Error in trade/{action}: {e}"

    # ──────────────────────────────────────────────
    # 下单
    # ──────────────────────────────────────────────

    async def _place_order(
        self,
        symbol: str,
        dry_run: bool,
        exchange: str | None,
        side: str | None = None,
        type: str | None = None,
        amount: float | None = None,
        price: float | None = None,
        leverage: int | None = None,
        stop_loss: float | None = None,
        take_profit: float | None = None,
        **_,
    ) -> str:
        # 参数校验
        if not side:
            return "Error: 'side' (buy/sell) is required for place_order"
        if not type:
            return "Error: 'type' (market/limit) is required for place_order"
        if not amount or amount <= 0:
            return "Error: 'amount' must be a positive number"
        if type == "limit" and not price:
            return "Error: 'price' is required for limit orders"

        ex = self._resolve_exchange(exchange)

        # 获取当前价格用于预览
        price_data = await ex.get_price(symbol)
        current_price = price_data.get("price", 0) if isinstance(price_data, dict) else 0

        # 计算预估值
        exec_price = price if type == "limit" else current_price
        notional = amount * exec_price if exec_price else 0
        margin_required = notional / leverage if leverage else notional

        # 生成订单预览
        preview = self._format_order_preview(
            symbol=symbol,
            side=side,
            order_type=type,
            amount=amount,
            price=exec_price,
            current_price=current_price,
            leverage=leverage,
            stop_loss=stop_loss,
            take_profit=take_profit,
            notional=notional,
            margin_required=margin_required,
        )

        # dry_run 模式只返回预览
        if dry_run:
            return (
                "🔍 模拟下单（Paper Trade）— 订单预览（未执行）\n"
                f"{'═' * 50}\n"
                f"{preview}\n"
                f"{'═' * 50}\n"
                "⚠ 如需真实执行：请再次确认后下真实单。"
            )

        # 真实下单
        # 设置杠杆（如果是合约且指定了杠杆）
        if leverage:
            try:
                await ex._call("set_leverage", leverage, symbol)
            except Exception as e:
                return f"Error setting leverage to {leverage}x: {e}"

        # 构造下单参数
        params: dict[str, Any] = {}
        if stop_loss:
            params["stopLoss"] = {"triggerPrice": stop_loss}
        if take_profit:
            params["takeProfit"] = {"triggerPrice": take_profit}

        order = await ex.place_order(
            symbol=symbol,
            order_type=type,
            side=side,
            amount=amount,
            price=price,
            params=params,
        )

        if isinstance(order, dict) and "error" in order:
            error_msg = str(order["error"])

            # Check if this is a MIN_NOTIONAL error
            if "NOTIONAL" in error_msg.upper() or "太小" in error_msg:
                # Fetch market limits to provide helpful info
                try:
                    limits = await ex.get_market_limits(symbol)
                    if isinstance(limits, dict) and "error" not in limits:
                        min_notional = limits.get("min_notional", 0)
                        actual_notional = notional
                        error_msg = (
                            f"{error_msg}\n\n"
                            f"📋 最小下单要求:\n"
                            f"  • 最小名义价值: {min_notional} USDT\n"
                            f"  • 当前订单价值: {actual_notional:.2f} USDT\n"
                            f"  • 需要增加: {max(0, min_notional - actual_notional):.2f} USDT\n\n"
                            f"💡 建议: 增加下单金额到至少 {min_notional} USDT"
                        )
                except Exception as e:
                    logger.debug(f"Failed to fetch market limits: {e}")

            # Log failed order
            await self._action_log.log_order(
                action="place_order",
                exchange=exchange or "default",
                symbol=symbol,
                side=side,
                order_type=type,
                amount=amount,
                price=price,
                leverage=leverage,
                status="failed",
                error=order["error"],
                stop_loss=stop_loss,
                take_profit=take_profit,
            )
            return f"❌ Order Failed: {error_msg}"

        # Log successful order
        await self._action_log.log_order(
            action="place_order",
            exchange=exchange or "default",
            symbol=symbol,
            side=side,
            order_type=type,
            amount=amount,
            price=order.get("price", exec_price),
            leverage=leverage,
            order_id=order.get("id"),
            status="success",
            stop_loss=stop_loss,
            take_profit=take_profit,
        )

        return (
            f"✅ Order Placed Successfully\n"
            f"{'─' * 50}\n"
            f"  Order ID: {order.get('id', 'N/A')}\n"
            f"  Symbol: {symbol}\n"
            f"  Side: {side.upper()} | Type: {type.upper()}\n"
            f"  Amount: {amount}\n"
            f"  Price: {order.get('price', exec_price)}\n"
            f"  Status: {order.get('status', 'N/A')}\n"
            f"  Time: {datetime.now(timezone.utc).isoformat()}"
        )

    # ──────────────────────────────────────────────
    # 取消订单
    # ──────────────────────────────────────────────

    async def _cancel_order(
        self,
        symbol: str,
        dry_run: bool,
        exchange: str | None,
        order_id: str | None = None,
        **_,
    ) -> str:
        if not order_id:
            return "Error: 'order_id' is required for cancel_order"

        ex = self._resolve_exchange(exchange)

        if dry_run:
            return (
                f"🔍 模拟下单（Paper Trade）— 撤单预览（未执行）\n"
                f"{'─' * 50}\n"
                f"  Would cancel order: {order_id}\n"
                f"  Symbol: {symbol}\n"
                f"⚠ 如需真实执行：请再次确认后下真实单。"
            )

        result = await ex.cancel_order(order_id, symbol)

        # Log successful cancellation
        await self._action_log.log_order(
            action="cancel_order",
            exchange=exchange or "default",
            symbol=symbol,
            side="",
            order_type="cancel",
            amount=0,
            order_id=order_id,
            status="success",
        )

        return (
            f"✅ Order Cancelled\n"
            f"{'─' * 50}\n"
            f"  Order ID: {order_id}\n"
            f"  Symbol: {symbol}\n"
            f"  Status: {result.get('status', 'cancelled')}"
        )

    # ──────────────────────────────────────────────
    # 止损设置
    # ──────────────────────────────────────────────

    async def _set_stop_loss(
        self,
        symbol: str,
        dry_run: bool,
        exchange: str | None,
        stop_loss: float | None = None,
        side: str | None = None,
        amount: float | None = None,
        **_,
    ) -> str:
        if not stop_loss:
            return "Error: 'stop_loss' price is required"

        ex = self._resolve_exchange(exchange)
        price_data = await ex.get_price(symbol)
        current_price = price_data.get("price", 0) if isinstance(price_data, dict) else 0

        inferred_amount, inferred_close_side = await self._infer_close_amount_and_side(ex, symbol)
        effective_amount = amount if (amount is not None and amount > 0) else inferred_amount

        # 推断止损方向：如果不指定 side，根据价格关系推断
        if not side:
            side = inferred_close_side or ("sell" if stop_loss < current_price else "buy")

        # 计算止损距离
        sl_distance = abs(current_price - stop_loss) / current_price * 100 if current_price else 0

        if dry_run:
            return (
                f"🔍 模拟下单（Paper Trade）— 止损预览（未执行）\n"
                f"{'─' * 50}\n"
                f"  Symbol: {symbol}\n"
                f"  Current Price: {current_price}\n"
                f"  Stop Loss: {stop_loss} ({side.upper()} trigger)\n"
                f"  Distance: {sl_distance:.2f}%\n"
                f"  Amount: {effective_amount or 'all'}\n"
                f"⚠ 如需真实执行：请再次确认后下真实单。"
            )

        # Real execution requires a non-zero amount on Binance (reduceOnly "all" still needs amount).
        if not effective_amount or effective_amount <= 0:
            return (
                "Error: could not determine order amount for stop loss.\n"
                "Pass 'amount' explicitly, or ensure you have an active futures position for this symbol."
            )

        # 检查持仓模式（hedge mode 不需要 reduceOnly）
        mode_info = await ex.get_position_mode()
        is_hedge_mode = mode_info.get("dual_side_position", False)

        # 使用止损市价单
        # Exchange compatibility:
        # - Binance futures commonly expects "stopPrice"
        # - Some adapters use "stopLossPrice" / "takeProfitPrice"
        params: dict[str, Any] = {"stopPrice": stop_loss, "stopLossPrice": stop_loss}

        # 只在单向模式下添加 reduceOnly（hedge mode 不需要）
        if not is_hedge_mode:
            params["reduceOnly"] = True

        order = await ex.place_order(
            symbol=symbol,
            order_type="stop_market",
            side=side,
            amount=round(float(effective_amount), 8),
            params=params,
        )

        # Log successful stop loss
        await self._action_log.log_order(
            action="set_stop_loss",
            exchange=exchange or "default",
            symbol=symbol,
            side=side,
            order_type="stop_market",
            amount=effective_amount,
            order_id=order.get("id"),
            status="success",
            stop_loss=stop_loss,
        )

        return (
            f"✅ Stop Loss Set\n"
            f"{'─' * 50}\n"
            f"  Order ID: {order.get('id', 'N/A')}\n"
            f"  Symbol: {symbol}\n"
            f"  Stop Price: {stop_loss}\n"
            f"  Side: {side.upper()}\n"
            f"  Distance: {sl_distance:.2f}%"
        )

    # ──────────────────────────────────────────────
    # 止盈设置
    # ──────────────────────────────────────────────

    async def _set_take_profit(
        self,
        symbol: str,
        dry_run: bool,
        exchange: str | None,
        take_profit: float | None = None,
        side: str | None = None,
        amount: float | None = None,
        **_,
    ) -> str:
        if not take_profit:
            return "Error: 'take_profit' price is required"

        ex = self._resolve_exchange(exchange)
        price_data = await ex.get_price(symbol)
        current_price = price_data.get("price", 0) if isinstance(price_data, dict) else 0

        inferred_amount, inferred_close_side = await self._infer_close_amount_and_side(ex, symbol)
        effective_amount = amount if (amount is not None and amount > 0) else inferred_amount

        # 推断止盈方向
        if not side:
            side = inferred_close_side or ("sell" if take_profit > current_price else "buy")

        tp_distance = abs(take_profit - current_price) / current_price * 100 if current_price else 0

        if dry_run:
            return (
                f"🔍 模拟下单（Paper Trade）— 止盈预览（未执行）\n"
                f"{'─' * 50}\n"
                f"  Symbol: {symbol}\n"
                f"  Current Price: {current_price}\n"
                f"  Take Profit: {take_profit} ({side.upper()} trigger)\n"
                f"  Distance: {tp_distance:.2f}%\n"
                f"  Amount: {effective_amount or 'all'}\n"
                f"⚠ 如需真实执行：请再次确认后下真实单。"
            )

        if not effective_amount or effective_amount <= 0:
            return (
                "Error: could not determine order amount for take profit.\n"
                "Pass 'amount' explicitly, or ensure you have an active futures position for this symbol."
            )

        # 检查持仓模式（hedge mode 不需要 reduceOnly）
        mode_info = await ex.get_position_mode()
        is_hedge_mode = mode_info.get("dual_side_position", False)

        params: dict[str, Any] = {"stopPrice": take_profit, "takeProfitPrice": take_profit}

        # 只在单向模式下添加 reduceOnly（hedge mode 不需要）
        if not is_hedge_mode:
            params["reduceOnly"] = True

        order = await ex.place_order(
            symbol=symbol,
            order_type="take_profit_market",
            side=side,
            amount=round(float(effective_amount), 8),
            params=params,
        )

        # Log successful take profit
        await self._action_log.log_order(
            action="set_take_profit",
            exchange=exchange or "default",
            symbol=symbol,
            side=side,
            order_type="take_profit_market",
            amount=effective_amount,
            order_id=order.get("id"),
            status="success",
            take_profit=take_profit,
        )

        return (
            f"✅ Take Profit Set\n"
            f"{'─' * 50}\n"
            f"  Order ID: {order.get('id', 'N/A')}\n"
            f"  Symbol: {symbol}\n"
            f"  TP Price: {take_profit}\n"
            f"  Side: {side.upper()}\n"
            f"  Distance: {tp_distance:.2f}%"
        )

    @staticmethod
    async def _infer_close_amount_and_side(ex: Any, symbol: str) -> tuple[float | None, str | None]:
        """Infer full position size and close side (buy/sell) for reduceOnly TP/SL orders."""
        get_positions = getattr(ex, "get_positions", None)
        if not get_positions:
            return None, None
        try:
            positions = await get_positions(symbol=symbol)
        except Exception:
            return None, None
        if isinstance(positions, dict) and "error" in positions:
            return None, None
        if not isinstance(positions, list) or not positions:
            return None, None

        candidates = {str(symbol).strip()}
        if ":" in symbol:
            candidates.add(symbol.split(":", 1)[0])
        # Some exchanges may omit settlement suffix in position symbols
        if "/" in symbol and ":" not in symbol:
            base, quote = symbol.split("/", 1)
            quote = quote.split(":")[0]
            candidates.add(f"{base}/{quote}:{quote}")

        total_contracts = 0.0
        pos_side: str | None = None
        for p in positions:
            if not isinstance(p, dict):
                continue
            psym = str(p.get("symbol") or "").strip()
            if psym and psym not in candidates:
                continue
            try:
                total_contracts += float(p.get("contracts") or 0)
            except (TypeError, ValueError):
                continue
            # Capture a representative side if available
            if pos_side is None:
                pos_side = str(p.get("side") or "").lower() or None

        if total_contracts <= 0:
            return None, None

        close_side = None
        if pos_side in ("long", "buy"):
            close_side = "sell"
        elif pos_side in ("short", "sell"):
            close_side = "buy"

        return total_contracts, close_side

    # ──────────────────────────────────────────────
    # 模拟预览 (dry_run action)
    # ──────────────────────────────────────────────

    async def _dry_run(
        self,
        symbol: str,
        exchange: str | None,
        side: str | None = None,
        type: str | None = None,
        amount: float | None = None,
        price: float | None = None,
        leverage: int | None = None,
        stop_loss: float | None = None,
        take_profit: float | None = None,
        **_,
    ) -> str:
        """生成完整的交易模拟预览，包含风险评估"""
        if not all([side, type, amount]):
            return "Error: 模拟下单需要 side, type, amount"

        ex = self._resolve_exchange(exchange)
        price_data = await ex.get_price(symbol)
        current_price = price_data.get("price", 0) if isinstance(price_data, dict) else 0
        exec_price = price if type == "limit" else current_price
        notional = amount * exec_price if exec_price and amount else 0
        leverage_val = leverage or 1
        margin_required = notional / leverage_val

        # 风险评估
        risk_lines = self._assess_risk(
            side=side,
            exec_price=exec_price,
            current_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            leverage=leverage_val,
            notional=notional,
        )

        preview = self._format_order_preview(
            symbol=symbol,
            side=side,
            order_type=type,
            amount=amount,
            price=exec_price,
            current_price=current_price,
            leverage=leverage,
            stop_loss=stop_loss,
            take_profit=take_profit,
            notional=notional,
            margin_required=margin_required,
        )

        return (
            f"🔍 模拟下单（Paper Trade，不下真实单）\n"
            f"{'═' * 50}\n"
            f"{preview}\n"
            f"\n📋 Risk Assessment:\n"
            f"{risk_lines}\n"
            f"{'═' * 50}\n"
            f"仅为模拟下单（交易预演），不会产生真实下单。"
        )

    # ──────────────────────────────────────────────
    # 工具方法
    # ──────────────────────────────────────────────

    def _resolve_exchange(self, exchange: str | None):
        """解析交易所实例"""
        if exchange:
            return self.hub.get_exchange_sync(exchange)
        return self.hub.exchange

    @staticmethod
    def _format_order_preview(
        symbol: str,
        side: str,
        order_type: str,
        amount: float,
        price: float,
        current_price: float,
        leverage: int | None,
        stop_loss: float | None,
        take_profit: float | None,
        notional: float,
        margin_required: float,
    ) -> str:
        """格式化订单预览信息"""
        lines = [
            f"  Symbol: {symbol}",
            f"  Side: {side.upper()} | Type: {order_type.upper()}",
            f"  Amount: {amount}",
            f"  Price: {price} (current: {current_price})",
        ]
        if leverage:
            lines.append(f"  Leverage: {leverage}x")
        lines.append(f"  Notional Value: {notional:.4f}")
        lines.append(f"  Margin Required: {margin_required:.4f}")
        if stop_loss:
            sl_pct = abs(price - stop_loss) / price * 100 if price else 0
            lines.append(f"  Stop Loss: {stop_loss} (-{sl_pct:.2f}%)")
        if take_profit:
            tp_pct = abs(take_profit - price) / price * 100 if price else 0
            lines.append(f"  Take Profit: {take_profit} (+{tp_pct:.2f}%)")
        return "\n".join(lines)

    @staticmethod
    def _assess_risk(
        side: str,
        exec_price: float,
        current_price: float,
        stop_loss: float | None,
        take_profit: float | None,
        leverage: int,
        notional: float,
    ) -> str:
        """评估交易风险并生成报告"""
        lines: list[str] = []

        # 杠杆风险
        if leverage >= 20:
            lines.append(f"  ⚠ HIGH LEVERAGE ({leverage}x) — liquidation risk is elevated")
        elif leverage >= 10:
            lines.append(f"  ⚡ Moderate leverage ({leverage}x)")
        else:
            lines.append(f"  ✅ Conservative leverage ({leverage}x)")

        # 止损距离
        if stop_loss and exec_price:
            sl_pct = abs(exec_price - stop_loss) / exec_price * 100
            max_loss = notional * sl_pct / 100
            lines.append(f"  Stop loss distance: {sl_pct:.2f}%")
            lines.append(f"  Max loss at SL: {max_loss:.4f}")
            # 考虑杠杆后的实际亏损比例
            real_loss_pct = sl_pct * leverage
            if real_loss_pct > 50:
                lines.append(f"  ⚠ Leveraged loss at SL: {real_loss_pct:.1f}% of margin — VERY HIGH RISK")
            elif real_loss_pct > 20:
                lines.append(f"  ⚡ Leveraged loss at SL: {real_loss_pct:.1f}% of margin")
        else:
            lines.append("  ⚠ No stop loss set — unlimited downside risk")

        # 盈亏比
        if stop_loss and take_profit and exec_price:
            risk = abs(exec_price - stop_loss)
            reward = abs(take_profit - exec_price)
            rr = reward / risk if risk > 0 else 0
            if rr >= 2:
                lines.append(f"  ✅ Risk/Reward ratio: 1:{rr:.2f} (favorable)")
            elif rr >= 1:
                lines.append(f"  ⚡ Risk/Reward ratio: 1:{rr:.2f} (acceptable)")
            else:
                lines.append(f"  ⚠ Risk/Reward ratio: 1:{rr:.2f} (unfavorable)")

        # 限价单价格偏离
        if exec_price and current_price and exec_price != current_price:
            deviation = (exec_price - current_price) / current_price * 100
            lines.append(f"  Price deviation from market: {deviation:+.2f}%")

        return "\n".join(lines)
