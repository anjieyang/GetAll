"""持仓管理工具 - 合约仓位、现货持仓、账户余额、盈亏汇总、全量同步"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from getall.agent.tools.base import Tool
from getall.trading.data.exchange import ExchangeAdapter
from getall.trading.data.hub import DataHub


class PortfolioTool(Tool):
    """查询和同步加密货币持仓信息的工具。"""

    def __init__(self, hub: DataHub, workspace_path: Path | None = None):
        # 注入 DataHub，统一管理交易所连接
        self.hub = hub
        # 持仓快照路径: workspace/memory/trading/positions.json
        if workspace_path:
            self._memory_dir = workspace_path / "memory" / "trading"
        else:
            self._memory_dir = Path.home() / ".getall" / "workspace" / "memory" / "trading"
        self._memory_dir.mkdir(parents=True, exist_ok=True)
        self._positions_file = self._memory_dir / "positions.json"

    @property
    def name(self) -> str:
        return "portfolio"

    @property
    def description(self) -> str:
        return (
            "Query portfolio: contract positions, spot holdings, account balance, "
            "P&L summary. Supports sync_all for full position sync."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "The portfolio query to perform",
                    "enum": [
                        "positions",      # 合约持仓
                        "spot",           # 现货持仓
                        "balance",        # 账户余额
                        "open_orders",    # 全量挂单 (from WS cache)
                        "pnl_summary",    # 盈亏汇总
                        "sync_all",       # 全量同步
                    ],
                },
                "exchange": {
                    "type": "string",
                    "description": "Exchange name, e.g. 'binance'. If omitted, uses default exchange.",
                },
                "symbol": {
                    "type": "string",
                    "description": "Filter by symbol, e.g. 'BTC/USDT'. Optional.",
                },
            },
            "required": ["action"],
        }

    async def execute(self, **kwargs: Any) -> str:
        action: str = kwargs["action"]
        exchange: str | None = kwargs.get("exchange")
        symbol: str | None = kwargs.get("symbol")

        try:
            handlers = {
                "positions": self._positions,
                "spot": self._spot,
                "balance": self._balance,
                "open_orders": self._open_orders,
                "pnl_summary": self._pnl_summary,
                "sync_all": self._sync_all,
            }
            handler = handlers.get(action)
            if handler is None:
                return f"Error: unknown action '{action}'"
            return await handler(exchange=exchange, symbol=symbol)
        except Exception as e:
            return f"Error in portfolio/{action}: {e}"

    # ──────────────────────────────────────────────
    # 合约持仓
    # ──────────────────────────────────────────────

    async def _positions(self, exchange: str | None, symbol: str | None) -> str:
        """查询合约持仓"""
        ex = self._resolve_exchange(exchange)
        # 通过 ExchangeAdapter 获取持仓 (内部已过滤空仓)
        positions = await ex.get_positions(symbol=symbol)
        if isinstance(positions, dict) and "error" in positions:
            return f"Error: {positions['error']}"

        if not positions:
            return "📋 No active contract positions."

        lines = [f"📋 Contract Positions ({len(positions)} active)"]
        lines.append("─" * 50)

        total_unrealized = 0.0
        for pos in positions:
            sym = pos.get("symbol", "N/A")
            side = pos.get("side", "N/A")
            size = pos.get("contracts", 0)
            entry = pos.get("entry_price", "N/A")
            mark = pos.get("mark_price", "N/A")
            leverage = pos.get("leverage", "N/A")
            unrealized = float(pos.get("unrealized_pnl", 0))
            total_unrealized += unrealized
            liq_price = pos.get("liquidation_price", "N/A")

            lines.append(
                f"\n  {sym} | {side.upper()} | Size: {size}\n"
                f"    Entry: {entry} | Mark: {mark}\n"
                f"    Leverage: {leverage}x\n"
                f"    Unrealized PnL: {unrealized:+.4f}\n"
                f"    Liquidation: {liq_price}"
            )

        lines.append(f"\n{'─' * 50}")
        lines.append(f"  Total Unrealized PnL: {total_unrealized:+.4f}")
        return "\n".join(lines)

    # ──────────────────────────────────────────────
    # 全量挂单 (from WS cache)
    # ──────────────────────────────────────────────

    async def _open_orders(self, exchange: str | None, symbol: str | None) -> str:
        """
        Read open orders snapshot from memory/trading/orders.json (WS-updated).
        """
        path = self._memory_dir / "orders.json"
        if not path.exists():
            return (
                "📋 No open-orders cache found.\n"
                "Tip: enable ws:account so watchOrders() can populate memory/trading/orders.json."
            )
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            return f"Error reading orders.json: {e}"

        last_sync = data.get("last_sync") or data.get("lastUpdate") or data.get("updated_at") or "N/A"
        exchanges = data.get("exchanges", {}) if isinstance(data, dict) else {}

        # Load positions for enrichment (leverage/entry/side), best-effort.
        positions_by_ex_sym: dict[tuple[str, str], dict[str, Any]] = {}
        try:
            if self._positions_file.exists():
                pdata = json.loads(self._positions_file.read_text(encoding="utf-8"))
                pex = pdata.get("exchanges", {}) if isinstance(pdata, dict) else {}
                if isinstance(pex, dict):
                    for ex_name, ex_data in pex.items():
                        if not isinstance(ex_data, dict):
                            continue
                        contracts = ex_data.get("contracts") or []
                        if not isinstance(contracts, list):
                            continue
                        for p in contracts:
                            if not isinstance(p, dict):
                                continue
                            sym = p.get("symbol")
                            if sym:
                                positions_by_ex_sym[(str(ex_name), str(sym))] = p
        except Exception:
            positions_by_ex_sym = {}

        # Filter by exchange (if provided)
        if exchange:
            exchanges = {exchange: exchanges.get(exchange, {})} if isinstance(exchanges, dict) else {}

        # Collect orders across market types
        collected: list[dict[str, Any]] = []
        if isinstance(exchanges, dict):
            for ex_name, ex_data in exchanges.items():
                if not isinstance(ex_data, dict):
                    continue
                for mtype, orders in ex_data.items():
                    if not isinstance(orders, list):
                        continue
                    for o in orders:
                        if isinstance(o, dict):
                            # Ensure market_type is present
                            o2 = dict(o)
                            o2.setdefault("market_type", mtype)
                            o2.setdefault("exchange", ex_name)
                            collected.append(o2)

        # Symbol filter (handle common futures/spot symbol variants)
        if symbol:
            raw = symbol.strip()
            sym_spot = ExchangeAdapter._normalize_symbol_for_market(raw, "spot")
            sym_swap = ExchangeAdapter._normalize_symbol_for_market(raw, "swap")
            sym_delivery = ExchangeAdapter._normalize_symbol_for_market(raw, "delivery")
            candidates = {raw, sym_spot, sym_swap, sym_delivery}
            collected = [o for o in collected if str(o.get("symbol", "")).strip() in candidates]

        if not collected:
            filter_note = []
            if exchange:
                filter_note.append(f"exchange={exchange}")
            if symbol:
                filter_note.append(f"symbol={symbol}")
            note = f" ({', '.join(filter_note)})" if filter_note else ""
            return f"📋 No open orders{note}."

        # Sort deterministic: exchange, market_type, symbol, id
        collected.sort(key=lambda o: (str(o.get("exchange", "")), str(o.get("market_type", "")), str(o.get("symbol", "")), str(o.get("id", ""))))

        lines = [f"📋 Open Orders ({len(collected)})"]
        lines.append("─" * 50)
        lines.append(f"  Cache last_sync: {last_sync}")

        def _classify_tp_sl(order: dict[str, Any], pos: dict[str, Any] | None) -> str | None:
            raw_type = str(order.get("raw_type") or "").upper()
            if "TAKE_PROFIT" in raw_type:
                return "TP"
            if "STOP_LOSS" in raw_type or "TRAILING_STOP" in raw_type or raw_type.startswith("STOP"):
                return "SL"

            trigger = order.get("trigger_price")
            if trigger is None or not pos:
                return None
            try:
                trig = float(trigger)
                entry = float(pos.get("entry_price") or 0)
            except (TypeError, ValueError):
                return None
            side = str(pos.get("side") or "").lower()
            if not entry or side not in ("long", "short"):
                return None
            if side == "long":
                return "TP" if trig > entry else "SL" if trig < entry else None
            # short
            return "TP" if trig < entry else "SL" if trig > entry else None

        # Show up to 30 lines (each order is one line)
        for o in collected[:30]:
            pos = positions_by_ex_sym.get((str(o.get("exchange", "")), str(o.get("symbol", ""))))
            lev = pos.get("leverage") if isinstance(pos, dict) else None
            tp_sl = _classify_tp_sl(o, pos if isinstance(pos, dict) else None)
            raw_type = o.get("raw_type")
            kind = raw_type or str(o.get("type", "")).upper()
            if tp_sl:
                kind = f"{tp_sl}/{kind}"

            trigger = o.get("trigger_price")
            reduce_only = o.get("reduce_only")
            extras: list[str] = []
            if trigger is not None:
                extras.append(f"trigger={trigger}")
            if reduce_only is True:
                extras.append("reduceOnly")
            if o.get("position_side"):
                extras.append(f"posSide={o.get('position_side')}")
            if lev not in (None, "", 0, "0"):
                extras.append(f"lev={lev}x")

            lines.append(
                "  "
                + " | ".join(
                    [
                        f"{o.get('exchange')}/{o.get('market_type')}",
                        f"{o.get('symbol')}",
                        f"id={o.get('id')}",
                        f"{str(o.get('side', '')).upper()} {kind}",
                        f"price={o.get('price')}",
                        f"amount={o.get('amount')}",
                        f"remaining={o.get('remaining')}",
                        f"status={o.get('status')}",
                    ]
                    + (extras if extras else [])
                )
            )
        if len(collected) > 30:
            lines.append(f"  ... ({len(collected) - 30} more omitted)")

        return "\n".join(lines)

    # ──────────────────────────────────────────────
    # 现货持仓
    # ──────────────────────────────────────────────

    async def _spot(self, exchange: str | None, symbol: str | None) -> str:
        """查询现货持仓"""
        ex = self._resolve_exchange(exchange)
        balance = await ex.get_spot_balances()

        # 提取非零余额
        holdings: list[dict] = []
        total_info = balance.get("total", {})

        for asset, amount in total_info.items():
            if float(amount) <= 0:
                continue
            # 如果指定了 symbol，只返回匹配的
            if symbol and asset.upper() != symbol.split("/")[0].upper():
                continue
            free = float(balance.get("free", {}).get(asset, 0))
            used = float(balance.get("used", {}).get(asset, 0))
            holdings.append({
                "asset": asset,
                "total": float(amount),
                "free": free,
                "used": used,
            })

        if not holdings:
            filter_note = f" (filter: {symbol})" if symbol else ""
            return f"📋 No spot holdings{filter_note}."

        # 按 total 降序排列
        holdings.sort(key=lambda x: x["total"], reverse=True)

        lines = [f"📋 Spot Holdings ({len(holdings)} assets)"]
        lines.append("─" * 50)
        for h in holdings:
            lines.append(
                f"  {h['asset']}: {h['total']:.8g} "
                f"(free: {h['free']:.8g}, in orders: {h['used']:.8g})"
            )

        return "\n".join(lines)

    # ──────────────────────────────────────────────
    # 账户余额
    # ──────────────────────────────────────────────

    async def _balance(self, exchange: str | None, **_) -> str:
        """查询账户余额概览"""
        ex = self._resolve_exchange(exchange)
        stablecoins = {"USDT", "USDC", "BUSD", "DAI", "TUSD"}

        # Default: include ALL wallet types (Binance often splits assets across them)
        if isinstance(ex, ExchangeAdapter):
            balances_by_type = await ex.get_all_balances()
        else:
            # Fallback for mocks/other adapters
            bal = await ex.get_balance()
            balances_by_type = {"default": bal} if isinstance(bal, dict) else {}

        # Compute stable totals per wallet + aggregate
        stable_breakdown: dict[str, dict[str, float]] = {}
        stable_total_all = 0.0
        stable_free_all = 0.0
        stable_used_all = 0.0

        # Aggregate non-stable totals across all wallets
        asset_totals: dict[str, float] = {}
        # Aggregate all asset totals across all wallets
        all_assets_total: dict[str, float] = {}

        for wtype, bal in balances_by_type.items():
            if not isinstance(bal, dict) or "error" in bal:
                continue

            total = bal.get("total", {}) or {}
            free = bal.get("free", {}) or {}
            used = bal.get("used", {}) or {}

            st_total = sum(float(total.get(s, 0) or 0) for s in stablecoins)
            st_free = sum(float(free.get(s, 0) or 0) for s in stablecoins)
            st_used = sum(float(used.get(s, 0) or 0) for s in stablecoins)
            stable_breakdown[wtype] = {"total": st_total, "free": st_free, "used": st_used}
            stable_total_all += st_total
            stable_free_all += st_free
            stable_used_all += st_used

            for asset, amount in total.items():
                try:
                    amt_all = float(amount or 0)
                except (TypeError, ValueError):
                    continue
                if amt_all > 0:
                    all_assets_total[asset] = all_assets_total.get(asset, 0.0) + amt_all

                if asset in stablecoins:
                    continue
                try:
                    amt = float(amount or 0)
                except (TypeError, ValueError):
                    continue
                if amt > 0:
                    asset_totals[asset] = asset_totals.get(asset, 0.0) + amt

        lines = [f"💰 Account Balance (all wallets)"]
        lines.append("─" * 50)
        lines.append(
            f"  Stablecoin Total: ${stable_total_all:,.2f} "
            f"(free: ${stable_free_all:,.2f}, used: ${stable_used_all:,.2f})"
        )

        if stable_breakdown:
            lines.append("  Stablecoin breakdown:")
            for wtype in ["spot", "funding", "swap", "delivery", "margin", "default"]:
                if wtype in stable_breakdown:
                    v = stable_breakdown[wtype]
                    lines.append(
                        f"    {wtype}: ${v['total']:,.2f} "
                        f"(free: ${v['free']:,.2f}, used: ${v['used']:,.2f})"
                    )

        if all_assets_total:
            lines.append("")
            lines.append("  All assets (total across wallets):")
            for asset, amount in sorted(all_assets_total.items(), key=lambda x: -x[1])[:50]:
                lines.append(f"    {asset}: {amount:.8g}")

        # Show non-stable holdings so total assets aren't understated
        if asset_totals:
            lines.append("")
            lines.append("  Non-stable assets (total across wallets):")
            for asset, amount in sorted(asset_totals.items(), key=lambda x: -x[1])[:30]:
                lines.append(f"    {asset}: {amount:.8g}")

        # Best-effort: estimate USDT value for a few spot assets (kept small to avoid rate limits)
        if isinstance(ex, ExchangeAdapter) and asset_totals:
            est_value = 0.0
            valued: list[str] = []
            try:
                ex._set_market_type("spot")  # prefer spot tickers for valuation
                for asset, amount in sorted(asset_totals.items(), key=lambda x: -x[1])[:5]:
                    sym = f"{asset}/USDT"
                    price_data = await ex.get_price(sym)
                    if isinstance(price_data, dict) and "error" not in price_data:
                        p = price_data.get("price")
                        if p:
                            est_value += float(amount) * float(p)
                            valued.append(asset)
            except Exception:
                pass
            finally:
                try:
                    ex._set_market_type("swap")
                except Exception:
                    pass

            if est_value > 0 and valued:
                lines.append("")
                lines.append(f"  Estimated spot value (USDT) for {', '.join(valued)}: ~${est_value:,.2f}")
                lines.append(f"  Estimated total (stable + valued spot): ~${stable_total_all + est_value:,.2f}")

        return "\n".join(lines)

    # ──────────────────────────────────────────────
    # 盈亏汇总
    # ──────────────────────────────────────────────

    async def _pnl_summary(self, exchange: str | None, symbol: str | None) -> str:
        """汇总当前所有仓位的盈亏情况"""
        ex = self._resolve_exchange(exchange)
        positions = await ex.get_positions(symbol=symbol)
        if isinstance(positions, dict) and "error" in positions:
            return f"Error: {positions['error']}"

        if not positions:
            return "📊 No active positions — no PnL to report."

        total_unrealized = 0.0
        total_notional = 0.0
        winners = 0
        losers = 0

        lines = [f"📊 PnL Summary ({len(positions)} positions)"]
        lines.append("─" * 50)

        for pos in positions:
            sym = pos.get("symbol", "N/A")
            side = pos.get("side", "N/A")
            unrealized = float(pos.get("unrealized_pnl", 0))
            notional = float(pos.get("notional", 0))
            entry = float(pos.get("entry_price", 0))
            mark = float(pos.get("mark_price", 0))
            # 计算百分比
            pct = ((mark - entry) / entry * 100) if entry else 0
            if side == "short":
                pct = -pct

            total_unrealized += unrealized
            total_notional += notional
            if unrealized >= 0:
                winners += 1
            else:
                losers += 1

            status = "✅" if unrealized >= 0 else "❌"
            lines.append(f"  {status} {sym} ({side}): {unrealized:+.4f} ({pct:+.2f}%)")

        # 汇总
        roi = (total_unrealized / total_notional * 100) if total_notional else 0
        lines.append(f"\n{'─' * 50}")
        lines.append(f"  Total Unrealized PnL: {total_unrealized:+.4f}")
        lines.append(f"  Total Notional: {total_notional:.4f}")
        lines.append(f"  Overall ROI: {roi:+.2f}%")
        lines.append(f"  Winners: {winners} | Losers: {losers}")

        return "\n".join(lines)

    # ──────────────────────────────────────────────
    # 全量同步
    # ──────────────────────────────────────────────

    async def _sync_all(self, **_) -> str:
        """同步所有已配置交易所的全部持仓，写入 positions.json"""
        snapshot = {
            "last_sync": datetime.now(timezone.utc).isoformat(),
            "synced_at": datetime.now(timezone.utc).isoformat(),
            "exchanges": {},
        }

        exchange_names = self._get_configured_exchanges()
        if not exchange_names:
            # 只同步默认交易所
            exchange_names = ["default"]

        errors: list[str] = []

        for name in exchange_names:
            try:
                ex = self.hub.exchange if name == "default" else self.hub.get_exchange_sync(name)
                if not ex:
                    errors.append(f"{name}: adapter not available")
                    continue
                positions = await ex.get_positions()
                # positions 已经是过滤后的活跃仓位 (ExchangeAdapter 内部处理)
                active = positions if isinstance(positions, list) else []

                # Normalize to positions.json schema expected by ws/news modules:
                # - contracts: list of contract positions
                # - spot: list of spot holdings (base symbols)
                contracts = []
                for p in active:
                    if not isinstance(p, dict):
                        continue
                    contracts.append({
                        "symbol": p.get("symbol"),
                        "side": p.get("side"),
                        "amount": p.get("contracts"),
                        "notional": p.get("notional"),
                        "entry_price": p.get("entry_price"),
                        "mark_price": p.get("mark_price"),
                        "unrealized_pnl": p.get("unrealized_pnl"),
                        "leverage": p.get("leverage"),
                        "margin_mode": p.get("margin_mode"),
                        "liquidation_price": p.get("liquidation_price"),
                    })

                # Spot holdings snapshot (non-zero)
                spot_list: list[dict[str, Any]] = []
                try:
                    if isinstance(ex, ExchangeAdapter):
                        spot_bal = await ex.get_balance_by_type("spot")
                    else:
                        spot_bal = await ex.get_spot_balances()
                    if isinstance(spot_bal, dict) and "error" not in spot_bal:
                        tot = spot_bal.get("total", {}) or {}
                        fre = spot_bal.get("free", {}) or {}
                        usd = spot_bal.get("used", {}) or {}
                        for asset, amt in tot.items():
                            try:
                                a = float(amt or 0)
                            except (TypeError, ValueError):
                                continue
                            if a > 0:
                                spot_list.append({
                                    "symbol": asset,
                                    "total": a,
                                    "free": float(fre.get(asset, 0) or 0),
                                    "used": float(usd.get(asset, 0) or 0),
                                })
                except Exception:
                    spot_list = []

                # Balances across wallets (compact, non-zero totals only)
                balances_all = {}
                if isinstance(ex, ExchangeAdapter):
                    balances_all = await ex.get_all_balances()
                else:
                    bal = await ex.get_balance()
                    balances_all = {"default": bal} if isinstance(bal, dict) else {}

                snapshot["exchanges"][name] = {
                    "contracts": contracts,
                    "spot": spot_list,
                    "balances": balances_all,
                    "position_count": len(contracts),
                }
            except Exception as e:
                errors.append(f"{name}: {e}")
                snapshot["exchanges"][name] = {"error": str(e)}

        # 写入文件
        try:
            self._memory_dir.mkdir(parents=True, exist_ok=True)
            with open(self._positions_file, "w", encoding="utf-8") as f:
                json.dump(snapshot, f, indent=2, default=str)
        except Exception as e:
            return f"Error writing positions.json: {e}"

        # 汇总结果
        total_positions = sum(
            ex_data.get("position_count", 0)
            for ex_data in snapshot["exchanges"].values()
            if isinstance(ex_data, dict) and "error" not in ex_data
        )

        lines = [f"🔄 Portfolio Sync Complete"]
        lines.append("─" * 50)
        lines.append(f"  Synced at: {snapshot['synced_at']}")
        lines.append(f"  Exchanges: {len(exchange_names)}")
        lines.append(f"  Total active positions: {total_positions}")
        lines.append(f"  Saved to: positions.json")

        if errors:
            lines.append(f"\n  ⚠ Errors ({len(errors)}):")
            for err in errors:
                lines.append(f"    - {err}")

        return "\n".join(lines)

    # ──────────────────────────────────────────────
    # 工具方法
    # ──────────────────────────────────────────────

    def _resolve_exchange(self, exchange: str | None):
        """解析交易所实例"""
        if exchange:
            return self.hub.get_exchange_sync(exchange)
        return self.hub.exchange

    def _get_configured_exchanges(self) -> list[str]:
        """获取已配置的交易所列表（从 DataHub 动态读取）"""
        if hasattr(self.hub, "configured_exchanges"):
            return self.hub.configured_exchanges
        return []
