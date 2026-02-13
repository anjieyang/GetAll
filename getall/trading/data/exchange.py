"""ccxt 封装: Binance + Bitget, 现货 + 合约.

支持: price, klines, ticker, positions, balance, place_order
优先使用 ccxt.pro (async), 如不可用则回退到 ccxt + asyncio.to_thread
"""

import asyncio
from typing import Any

from loguru import logger

# ccxt 导入: 优先 pro (原生 async), 回退到同步版本
try:
    import ccxt.pro as ccxtpro

    _HAS_PRO = True
except ImportError:
    _HAS_PRO = False

import ccxt


# 支持的交易所白名单
SUPPORTED_EXCHANGES = {"binance", "bitget"}


class ExchangeAdapter:
    """
    Unified exchange adapter wrapping ccxt.

    Supports Binance and Bitget, both spot and futures.
    """

    def __init__(
        self,
        exchange_name: str,
        api_key: str = "",
        secret: str = "",
        password: str = "",
        sandbox: bool = False,
    ):
        """
        Initialize exchange connection.

        Args:
            exchange_name: Exchange identifier (binance / bitget).
            api_key: API key for authentication.
            secret: API secret for authentication.
            password: API passphrase (required by bitget).
            sandbox: Whether to use testnet / sandbox mode.
        """
        name = exchange_name.lower()
        if name not in SUPPORTED_EXCHANGES:
            raise ValueError(f"Unsupported exchange: {name}. Supported: {SUPPORTED_EXCHANGES}")

        self.name = name
        self._sandbox = sandbox
        self._position_mode_cache: dict[str, Any] | None = None  # Cache for position mode

        # 构造 ccxt 配置
        config: dict[str, Any] = {
            "apiKey": api_key,
            "secret": secret,
            "enableRateLimit": True,
            "options": {"defaultType": "swap"},  # 默认合约市场
        }
        if password:
            config["password"] = password

        # 创建实例: 优先 pro, 回退 sync
        if _HAS_PRO:
            exchange_cls = getattr(ccxtpro, name, None)
            if exchange_cls:
                self._ex: Any = exchange_cls(config)
                self._is_pro = True
            else:
                self._ex = getattr(ccxt, name)(config)
                self._is_pro = False
        else:
            self._ex = getattr(ccxt, name)(config)
            self._is_pro = False

        if sandbox:
            self._ex.set_sandbox_mode(True)

        logger.info(f"Exchange adapter initialized: {name} (pro={self._is_pro}, sandbox={sandbox})")

    # ──────────────────────── 内部辅助 ────────────────────────

    async def _call(self, method: str, *args: Any, **kwargs: Any) -> Any:
        """
        统一调用: pro 直接 await, sync 走 to_thread.

        Args:
            method: ccxt method name.

        Returns:
            Method result.
        """
        fn = getattr(self._ex, method)
        try:
            if self._is_pro or asyncio.iscoroutinefunction(fn):
                return await fn(*args, **kwargs)
            return await asyncio.to_thread(fn, *args, **kwargs)
        except (ccxt.NetworkError, ccxt.ExchangeNotAvailable) as e:
            logger.error(f"[{self.name}] Network error in {method}: {e}")
            return {"error": f"Network error: {e}"}
        except ccxt.AuthenticationError as e:
            logger.error(f"[{self.name}] Auth error: {e}")
            return {"error": f"Authentication failed: {e}"}
        except ccxt.InsufficientFunds as e:
            logger.warning(f"[{self.name}] Insufficient funds: {e}")
            return {"error": f"Insufficient funds: {e}"}
        except ccxt.InvalidOrder as e:
            logger.warning(f"[{self.name}] Invalid order: {e}")
            return {"error": f"Invalid order: {e}"}
        except ccxt.BaseError as e:
            logger.error(f"[{self.name}] ccxt error in {method}: {e}")
            return {"error": f"Exchange error: {e}"}

    def _set_market_type(self, market_type: str = "swap") -> None:
        """切换市场类型 (spot / swap / future)."""
        self._ex.options["defaultType"] = market_type

    def _default_market_type(self) -> str:
        """Return current ccxt defaultType (best-effort)."""
        try:
            return str(self._ex.options.get("defaultType") or "").lower()
        except Exception:
            return ""

    @staticmethod
    def _normalize_pair_symbol(symbol: str, *, default_quote: str = "USDT") -> str:
        """Best-effort normalization for exchange market symbols.

        Accepts:
        - base coin: "BTC" -> "BTC/USDT"
        - dashed pair: "BTC-USDT" -> "BTC/USDT"
        - concatenated: "BTCUSDT" -> "BTC/USDT" (best-effort)
        - ccxt pair/futures: "BTC/USDT" / "BTC/USDT:USDT" -> unchanged
        """
        s = (symbol or "").strip()
        if not s:
            return s

        if "-" in s and "/" not in s:
            s = s.replace("-", "/")

        if "/" in s or ":" in s:
            return s

        upper = s.upper()
        dq = default_quote.upper()
        if upper == dq:
            return s

        if upper.endswith(dq) and len(upper) > len(dq):
            base = upper[: -len(dq)]
            return f"{base}/{dq}"

        return f"{upper}/{dq}"

    async def _fetch_ticker_with_fallback(self, symbol: str) -> tuple[Any, str]:
        """Fetch ticker with symbol normalization + swap suffix fallback."""
        sym = self._normalize_pair_symbol(symbol)
        ticker = await self._call("fetch_ticker", sym)
        if not (isinstance(ticker, dict) and "error" in ticker):
            return ticker, sym

        mtype = self._default_market_type()
        if mtype in ("swap", "future", "futures") and ":" not in sym and "/" in sym:
            sym2 = self._normalize_symbol_for_market(sym, "swap")
            if sym2 != sym:
                ticker2 = await self._call("fetch_ticker", sym2)
                if not (isinstance(ticker2, dict) and "error" in ticker2):
                    return ticker2, sym2
        return ticker, sym

    async def _fetch_ohlcv_with_fallback(
        self, symbol: str, timeframe: str, *, limit: int, since: int | None = None
    ) -> tuple[Any, str]:
        """Fetch OHLCV with symbol normalization + swap suffix fallback."""
        sym = self._normalize_pair_symbol(symbol)
        data = await self._call("fetch_ohlcv", sym, timeframe, since=since, limit=limit)
        if not (isinstance(data, dict) and "error" in data):
            return data, sym

        mtype = self._default_market_type()
        if mtype in ("swap", "future", "futures") and ":" not in sym and "/" in sym:
            sym2 = self._normalize_symbol_for_market(sym, "swap")
            if sym2 != sym:
                data2 = await self._call("fetch_ohlcv", sym2, timeframe, since=since, limit=limit)
                if not (isinstance(data2, dict) and "error" in data2):
                    return data2, sym2
        return data, sym

    @staticmethod
    def _normalize_symbol_for_market(symbol: str, market_type: str) -> str:
        """
        Normalize a unified ccxt symbol for the given market type.

        Binance futures commonly require settlement suffix, e.g.
        - swap (USDT-M):  ETH/USDT:USDT
        - delivery (COIN-M): ETH/USD:ETH
        """
        s = (symbol or "").strip()
        if not s:
            return s

        # spot symbols should NOT include settlement suffix
        if market_type == "spot":
            return s.split(":")[0] if ":" in s else s

        # futures symbols typically include settlement suffix; if missing, add a best-effort default
        if market_type in ("swap", "future"):
            if ":" not in s and "/" in s:
                base, quote = s.split("/", 1)
                quote = quote.split(":")[0]
                return f"{base}/{quote}:{quote}"
            return s

        if market_type == "delivery":
            if ":" not in s and "/" in s:
                base, quote = s.split("/", 1)
                quote = quote.split(":")[0]
                return f"{base}/{quote}:{base}"
            return s

        return s

    @staticmethod
    def _extract_nonzero_balance(bal: dict[str, Any]) -> dict[str, Any]:
        """Extract a compact balance dict with non-zero totals only."""
        result: dict[str, Any] = {"total": {}, "free": {}, "used": {}}
        for currency, amount in (bal.get("total") or {}).items():
            try:
                if amount and float(amount) > 0:
                    result["total"][currency] = float(amount)
                    result["free"][currency] = float((bal.get("free") or {}).get(currency, 0))
                    result["used"][currency] = float((bal.get("used") or {}).get(currency, 0))
            except (TypeError, ValueError):
                continue
        return result

    # ──────────────────────── 行情接口 ────────────────────────

    async def get_price(self, symbol: str) -> dict[str, Any]:
        """
        Get current price for a symbol.

        Args:
            symbol: Trading pair, e.g. "BTC/USDT".

        Returns:
            Dict with symbol, price, timestamp or error.
        """
        ticker, used_symbol = await self._fetch_ticker_with_fallback(symbol)
        if isinstance(ticker, dict) and "error" in ticker:
            return ticker
        return {
            "symbol": used_symbol,
            "price": ticker.get("last"),
            "bid": ticker.get("bid"),
            "ask": ticker.get("ask"),
            "timestamp": ticker.get("timestamp"),
        }

    async def get_klines(
        self, symbol: str, timeframe: str = "1h", limit: int = 100, since: int | None = None
    ) -> list[dict[str, Any]] | dict[str, Any]:
        """
        Get OHLCV candlestick data.

        Args:
            symbol: Trading pair.
            timeframe: Candle interval (1m, 5m, 15m, 1h, 4h, 1d, etc.).
            limit: Number of candles to fetch.
            since: Fetch candles since this timestamp (ms since epoch).

        Returns:
            List of OHLCV dicts or error dict.
        """
        data, _used_symbol = await self._fetch_ohlcv_with_fallback(
            symbol, timeframe, limit=limit, since=since
        )
        if isinstance(data, dict) and "error" in data:
            return data
        return [
            {
                "timestamp": c[0],
                "open": c[1],
                "high": c[2],
                "low": c[3],
                "close": c[4],
                "volume": c[5],
            }
            for c in (data or [])
        ]

    async def get_ticker(self, symbol: str) -> dict[str, Any]:
        """
        Get 24h ticker statistics.

        Args:
            symbol: Trading pair.

        Returns:
            Dict with 24h stats or error.
        """
        ticker, used_symbol = await self._fetch_ticker_with_fallback(symbol)
        if isinstance(ticker, dict) and "error" in ticker:
            return ticker
        return {
            "symbol": used_symbol,
            "last": ticker.get("last"),
            "high": ticker.get("high"),
            "low": ticker.get("low"),
            "volume": ticker.get("baseVolume"),
            "quote_volume": ticker.get("quoteVolume"),
            "change_pct": ticker.get("percentage"),
            "timestamp": ticker.get("timestamp"),
        }

    async def get_market_limits(self, symbol: str) -> dict[str, Any]:
        """
        Get market trading limits for a symbol (MIN_NOTIONAL, min amount, etc.).

        Args:
            symbol: Trading pair, e.g. "BTC/USDT".

        Returns:
            Dict with market limits or error.
            Example:
            {
                "symbol": "BTC/USDT",
                "min_notional": 5.0,
                "min_amount": 0.00001,
                "max_amount": 9000.0,
                "amount_precision": 5,
                "price_precision": 2
            }
        """
        sym = self._normalize_pair_symbol(symbol)

        # Load markets if not already loaded
        try:
            if not hasattr(self._ex, "markets") or not self._ex.markets:
                await self._call("load_markets")
        except Exception as e:
            logger.warning(f"[{self.name}] Failed to load markets: {e}")
            return {"error": f"Failed to load market info: {e}"}

        # Try to find the market
        market = self._ex.markets.get(sym)
        if not market:
            # Try with swap suffix for futures
            mtype = self._default_market_type()
            if mtype in ("swap", "future", "futures") and ":" not in sym:
                sym2 = self._normalize_symbol_for_market(sym, "swap")
                market = self._ex.markets.get(sym2)
                if market:
                    sym = sym2

        if not market:
            return {"error": f"Market not found: {symbol}"}

        # Extract limits
        limits = market.get("limits", {})
        cost_limits = limits.get("cost", {})
        amount_limits = limits.get("amount", {})
        price_limits = limits.get("price", {})

        precision = market.get("precision", {})

        return {
            "symbol": sym,
            "min_notional": float(cost_limits.get("min") or 0),
            "max_notional": float(cost_limits.get("max") or 0) if cost_limits.get("max") else None,
            "min_amount": float(amount_limits.get("min") or 0),
            "max_amount": float(amount_limits.get("max") or 0) if amount_limits.get("max") else None,
            "min_price": float(price_limits.get("min") or 0) if price_limits.get("min") else None,
            "max_price": float(price_limits.get("max") or 0) if price_limits.get("max") else None,
            "amount_precision": precision.get("amount"),
            "price_precision": precision.get("price"),
        }

    # ──────────────────────── 账户接口 ────────────────────────

    async def get_balance(self) -> dict[str, Any]:
        """
        Get full account balance (current market type).

        Returns:
            Balance dict or error.
        """
        bal = await self._call("fetch_balance")
        if isinstance(bal, dict) and "error" in bal:
            return bal
        return self._extract_nonzero_balance(bal)

    async def get_balance_by_type(self, balance_type: str) -> dict[str, Any]:
        """
        Get balance for a specific Binance wallet type.

        Common types: spot, funding, swap, delivery, margin
        """
        bal = await self._call("fetch_balance", {"type": balance_type})
        if isinstance(bal, dict) and "error" in bal:
            return bal
        return self._extract_nonzero_balance(bal)

    async def get_all_balances(
        self, balance_types: list[str] | None = None
    ) -> dict[str, dict[str, Any]]:
        """
        Get balances across multiple wallet types.

        Returns:
            Dict mapping wallet_type -> compact balance dict (non-zero totals only),
            or wallet_type -> {"error": "..."} if failed.
        """
        types = balance_types or ["spot", "funding", "swap", "delivery", "margin"]
        out: dict[str, dict[str, Any]] = {}
        for t in types:
            out[t] = await self.get_balance_by_type(t)
        return out

    async def get_positions(self, symbol: str | None = None) -> list[dict[str, Any]] | dict[str, Any]:
        """
        Get open contract positions.

        Args:
            symbol: Optional symbol filter, e.g. "BTC/USDT:USDT".

        Returns:
            List of position dicts or error.
        """
        self._set_market_type("swap")
        params: dict[str, Any] = {}
        sym = self._normalize_symbol_for_market(symbol, "swap") if symbol else None
        symbols = [sym] if sym else None

        # For Binance, use positionRisk endpoint which includes leverage
        if self.name == "binance":
            try:
                # Use fapiPrivateV2GetPositionRisk for accurate leverage info
                risk_data = await self._call("fapiPrivateV2GetPositionRisk", params)
                if isinstance(risk_data, dict) and "error" in risk_data:
                    return risk_data

                positions = []
                for p in (risk_data or []):
                    pos_amt = float(p.get("positionAmt") or 0)
                    if pos_amt == 0:
                        continue

                    # Filter by symbol if specified
                    raw_symbol = p.get("symbol", "")
                    if symbol:
                        # Remove :USDT suffix for comparison
                        target_symbol = symbol.replace("/", "").replace(":USDT", "")
                        if raw_symbol != target_symbol:
                            continue

                    # Convert raw symbol to ccxt format (e.g., SOLUSDT -> SOL/USDT:USDT)
                    # Try safe_symbol first, fallback to manual conversion
                    ccxt_symbol = self._ex.safe_symbol(raw_symbol)
                    if ccxt_symbol == raw_symbol and raw_symbol.endswith("USDT"):
                        # Manual conversion for USDT-margined contracts
                        base = raw_symbol[:-4]  # Remove USDT suffix
                        ccxt_symbol = f"{base}/USDT:USDT"

                    positions.append({
                        "symbol": ccxt_symbol,
                        "side": "long" if pos_amt > 0 else "short",
                        "contracts": abs(pos_amt),
                        "notional": float(p.get("notional") or 0),
                        "entry_price": float(p.get("entryPrice") or 0),
                        "mark_price": float(p.get("markPrice") or 0),
                        "unrealized_pnl": float(p.get("unRealizedProfit") or 0),
                        "leverage": float(p.get("leverage") or 0),
                        "margin_mode": "cross" if p.get("marginType") == "cross" else "isolated",
                        "liquidation_price": float(p.get("liquidationPrice") or 0),
                    })
                return positions
            except Exception as e:
                logger.error(f"[{self.name}] Failed to fetch position risk: {e}")
                # Fall through to default method

        # Default method for other exchanges or as fallback
        data = await self._call("fetch_positions", symbols, params)
        if isinstance(data, dict) and "error" in data:
            return data
        positions = []
        for p in (data or []):
            contracts = float(p.get("contracts") or 0)
            if contracts == 0:
                continue

            # Try to calculate leverage if not provided
            leverage = float(p.get("leverage") or 0)
            if leverage == 0:
                notional = float(p.get("notional") or 0)
                initial_margin = float(p.get("initialMargin") or 0)
                if notional > 0 and initial_margin > 0:
                    leverage = notional / initial_margin

            positions.append({
                "symbol": p.get("symbol"),
                "side": p.get("side"),
                "contracts": contracts,
                "notional": float(p.get("notional") or 0),
                "entry_price": float(p.get("entryPrice") or 0),
                "mark_price": float(p.get("markPrice") or 0),
                "unrealized_pnl": float(p.get("unrealizedPnl") or 0),
                "leverage": leverage,
                "margin_mode": p.get("marginMode"),
                "liquidation_price": p.get("liquidationPrice"),
            })
        return positions

    async def get_spot_balances(self) -> dict[str, Any]:
        """
        Get spot account balances.

        Returns:
            Balance dict (spot market) or error.
        """
        self._set_market_type("spot")
        result = await self.get_balance()
        self._set_market_type("swap")  # 恢复默认
        return result

    # ──────────────────────── 持仓模式 ────────────────────────

    async def get_position_mode(self, use_cache: bool = True) -> dict[str, Any]:
        """
        Get the current position mode (hedge/one-way) for futures trading.

        Args:
            use_cache: Whether to use cached result (default: True)

        Returns:
            Dict with 'dual_side_position' (True = hedge mode, False = one-way mode)
            or error dict.

        Note:
            - Binance: GET /fapi/v1/positionSide/dual
            - Hedge mode (dual=True): must specify positionSide as LONG or SHORT
            - One-way mode (dual=False): positionSide should be BOTH or omitted
        """
        # Return cached result if available
        if use_cache and self._position_mode_cache is not None:
            return self._position_mode_cache

        if self.name == "binance":
            try:
                # Try using the ccxt wrapper method if available
                if hasattr(self._ex, "fapiPrivateGetPositionsideDual"):
                    result = await self._call("fapiPrivateGetPositionsideDual")
                elif hasattr(self._ex, "fapiPrivateGetPositionSideDual"):
                    result = await self._call("fapiPrivateGetPositionSideDual")
                else:
                    # Fallback: try to infer from positions
                    logger.warning(f"[{self.name}] Position mode API not available, using default")
                    result = {"dualSidePosition": False}

                if isinstance(result, dict) and "error" in result:
                    return result

                # Binance returns {"dualSidePosition": true/false}
                dual = result.get("dualSidePosition", False)
                mode_info = {"dual_side_position": dual, "mode": "hedge" if dual else "one-way"}

                # Cache the result
                self._position_mode_cache = mode_info
                return mode_info
            except Exception as e:
                logger.warning(f"[{self.name}] Failed to get position mode: {e}")
                # Default to one-way mode on error
                default = {"dual_side_position": False, "mode": "one-way"}
                self._position_mode_cache = default
                return default
        elif self.name == "bitget":
            # Bitget uses different API structure; for now return a sensible default
            # Most Bitget futures accounts default to one-way mode
            default = {"dual_side_position": False, "mode": "one-way"}
            self._position_mode_cache = default
            return default

        # Default for other exchanges
        default = {"dual_side_position": False, "mode": "one-way"}
        self._position_mode_cache = default
        return default

    def _get_position_side(self, side: str, is_reduce_only: bool, is_hedge_mode: bool) -> str | None:
        """
        Determine the correct positionSide parameter for Binance futures.

        Args:
            side: Order side ('buy' or 'sell')
            is_reduce_only: Whether this is a reduce-only order (closing position)
            is_hedge_mode: Whether account is in hedge mode (dual position)

        Returns:
            positionSide value ('LONG', 'SHORT', 'BOTH') or None
        """
        if not is_hedge_mode:
            # One-way mode: positionSide should be BOTH or omitted
            return "BOTH"

        # Hedge mode: need to specify LONG or SHORT
        if is_reduce_only:
            # Closing position: sell closes LONG, buy closes SHORT
            return "LONG" if side.lower() == "sell" else "SHORT"
        else:
            # Opening position: buy opens LONG, sell opens SHORT
            return "LONG" if side.lower() == "buy" else "SHORT"

    # ──────────────────────── 交易接口 ────────────────────────

    async def place_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        amount: float,
        price: float | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Place an order.

        Args:
            symbol: Trading pair.
            side: "buy" or "sell".
            order_type: "market" or "limit".
            amount: Order amount.
            price: Limit price (required for limit orders).
            params: Extra exchange-specific params.

        Returns:
            Order result dict or error.
        """
        # Make a copy of params to avoid modifying the input
        params = dict(params or {})

        # Auto-detect and set positionSide for Binance futures (if not already set)
        if self.name == "binance" and ":" in symbol and "positionSide" not in params:
            try:
                # Get current position mode
                mode_info = await self.get_position_mode()
                if "error" not in mode_info:
                    is_hedge_mode = mode_info.get("dual_side_position", False)
                    # Check if this is a reduce-only order
                    # Stop-loss and take-profit orders are always reduce-only
                    is_reduce_only = (
                        params.get("reduceOnly", False)
                        or order_type.lower() in ("stop_market", "take_profit_market", "stop", "take_profit")
                    )

                    # Determine correct positionSide
                    position_side = self._get_position_side(side, is_reduce_only, is_hedge_mode)
                    if position_side:
                        params["positionSide"] = position_side
                        logger.debug(
                            f"Auto-set positionSide={position_side} "
                            f"(mode={'hedge' if is_hedge_mode else 'one-way'}, "
                            f"side={side}, reduceOnly={is_reduce_only}, type={order_type})"
                        )
            except Exception as e:
                logger.warning(f"Failed to auto-detect positionSide: {e}")
                # Continue without positionSide and let the exchange return an error if needed

        result = await self._call(
            "create_order", symbol, order_type, side, amount, price, params
        )
        if isinstance(result, dict) and "error" in result:
            return result
        return {
            "id": result.get("id"),
            "symbol": result.get("symbol"),
            "side": result.get("side"),
            "type": result.get("type"),
            "amount": result.get("amount"),
            "price": result.get("price"),
            "status": result.get("status"),
            "timestamp": result.get("timestamp"),
        }

    async def cancel_order(self, order_id: str, symbol: str) -> dict[str, Any]:
        """
        Cancel an open order.

        Args:
            order_id: Order ID to cancel.
            symbol: Trading pair of the order.

        Returns:
            Cancellation result or error.
        """
        # First try as-is (user may pass spot/futures explicitly).
        result = await self._call("cancel_order", order_id, symbol)
        if isinstance(result, dict) and "error" not in result:
            return {
                "id": result.get("id"),
                "symbol": result.get("symbol"),
                "status": result.get("status", "cancelled"),
            }

        # If user passed a spot-style symbol but the order is futures, retry with normalized swap symbol.
        if isinstance(result, dict) and "error" in result and ":" not in symbol:
            sym2 = self._normalize_symbol_for_market(symbol, "swap")
            if sym2 != symbol:
                result2 = await self._call("cancel_order", order_id, sym2)
                if isinstance(result2, dict) and "error" not in result2:
                    return {
                        "id": result2.get("id"),
                        "symbol": result2.get("symbol"),
                        "status": result2.get("status", "cancelled"),
                    }
        return result

    # ──────────────────────── 生命周期 ────────────────────────

    async def close(self) -> None:
        """关闭连接, 释放资源."""
        try:
            if self._is_pro and hasattr(self._ex, "close"):
                await self._ex.close()
            logger.debug(f"Exchange adapter closed: {self.name}")
        except Exception as e:
            logger.warning(f"Error closing exchange {self.name}: {e}")
