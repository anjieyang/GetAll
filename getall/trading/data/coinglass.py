"""Coinglass API v4 衍生品 & 市场数据适配器.

直连: https://open-api-v4.coinglass.com/api  (需 CG-API-KEY)
Plan: Startup — 80+ endpoints, 80 req/min

所有方法均为 async, 使用 httpx 发起请求.
Coinglass v4 路径使用 kebab-case (连字符).

Symbol 格式说明:
  - 带 exchange 参数的 history 端点 → 交易对格式 "BTCUSDT"
  - exchange-list / aggregated 端点 → 基础币种 "BTC"
  适配器内部自动转换.
"""

from typing import Any

import httpx
from loguru import logger


_TIMEOUT = 15.0


def _to_pair(symbol: str) -> str:
    """归一化为交易对格式 (BTCUSDT)."""
    s = symbol.upper().replace(" ", "").split(":")[0].replace("/", "")
    if not s.endswith(("USDT", "USDC", "BUSD", "USD")):
        s = s + "USDT"
    return s


def _to_base(symbol: str) -> str:
    """归一化为基础币种 (BTC)."""
    s = symbol.upper().replace(" ", "").split(":")[0].split("/")[0]
    for suffix in ("USDT", "USDC", "BUSD", "USD"):
        if s.endswith(suffix) and len(s) > len(suffix):
            s = s[: -len(suffix)]
            break
    return s


class CoinglassAdapter:
    """Coinglass open-api-v4 adapter (Startup plan).

    Capabilities:
      Futures: OI, funding, long/short ratio, taker/CVD, liquidations,
               basis, orderbook, coin flow
      Spot:    pairs markets, net flow, supported coins
      Options: info, max pain
      Index:   fear & greed, AHR999
      Coin:    token unlocks
      Chain:   whale transfers
    """

    def __init__(self, base_url: str, api_key: str = ""):
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key

    # ──────────────────────── 底层请求 ────────────────────────

    async def _request(
        self, endpoint: str, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Send GET request to Coinglass API."""
        url = f"{self._base_url}/{endpoint.lstrip('/')}"
        headers: dict[str, str] = {"accept": "application/json"}
        if self._api_key:
            headers["CG-API-KEY"] = self._api_key
        query = dict(params or {})

        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                resp = await client.get(url, params=query, headers=headers)
                resp.raise_for_status()
                data = resp.json()
                if data.get("code") not in ("0", 0, None):
                    msg = data.get("msg", "Unknown error")
                    logger.warning(f"Coinglass API error [{endpoint}]: {msg}")
                    return {"error": msg, "endpoint": endpoint}
                return data.get("data") if "data" in data else data
        except httpx.HTTPStatusError as e:
            logger.error(f"Coinglass HTTP {e.response.status_code}: {endpoint}")
            return {"error": f"HTTP {e.response.status_code}", "endpoint": endpoint}
        except httpx.RequestError as e:
            logger.error(f"Coinglass request failed [{endpoint}]: {e}")
            return {"error": str(e), "endpoint": endpoint}

    # ════════════════════════ Futures ════════════════════════

    # ── OI (持仓量) ──

    async def get_oi_history(
        self, symbol: str, exchange: str = "Binance",
        interval: str = "1d", limit: int = 100,
    ) -> Any:
        """OI history for one symbol on one exchange. Symbol: pair."""
        return await self._request(
            "futures/open-interest/history",
            {"symbol": _to_pair(symbol), "exchange": exchange,
             "interval": interval, "limit": limit, "unit": "usd"},
        )

    async def get_aggregated_oi(
        self, symbol: str, interval: str = "1d", limit: int = 50,
    ) -> Any:
        """Aggregated OI across all exchanges. Symbol: base."""
        return await self._request(
            "futures/open-interest/aggregated-history",
            {"symbol": _to_base(symbol), "interval": interval, "limit": limit},
        )

    async def get_exchange_oi(self, symbol: str) -> Any:
        """OI breakdown by exchange. Symbol: base."""
        return await self._request(
            "futures/open-interest/exchange-list",
            {"symbol": _to_base(symbol)},
        )

    # ── 资金费率 ──

    async def get_funding_rate_history(
        self, symbol: str, exchange: str = "Binance", interval: str = "h8",
    ) -> Any:
        """Funding rate history. Symbol: pair."""
        return await self._request(
            "futures/funding-rate/history",
            {"symbol": _to_pair(symbol), "exchange": exchange, "interval": interval},
        )

    async def get_funding_rate_exchange(self, symbol: str) -> Any:
        """Current funding rates across exchanges. Symbol: base."""
        return await self._request(
            "futures/funding-rate/exchange-list",
            {"symbol": _to_base(symbol)},
        )

    # ── 多空比 ──

    async def get_long_short_ratio(
        self, symbol: str, exchange: str = "Binance", interval: str = "h1",
    ) -> Any:
        """Global long/short account ratio. Symbol: pair."""
        return await self._request(
            "futures/global-long-short-account-ratio/history",
            {"symbol": _to_pair(symbol), "exchange": exchange, "interval": interval},
        )

    # ── Taker / CVD ──

    async def get_taker_buy_sell(
        self, symbol: str, exchange: str = "Binance",
        interval: str = "h1", limit: int = 10,
    ) -> Any:
        """Taker buy/sell volume history. Symbol: pair."""
        return await self._request(
            "futures/taker-buy-sell-volume/history",
            {"symbol": _to_pair(symbol), "exchange": exchange,
             "interval": interval, "limit": limit},
        )

    async def get_taker_buy_sell_history(
        self, symbol: str, exchange: str = "Binance", interval: str = "h1",
    ) -> Any:
        """Taker buy/sell volume history (full). Symbol: pair."""
        return await self._request(
            "futures/taker-buy-sell-volume/history",
            {"symbol": _to_pair(symbol), "exchange": exchange, "interval": interval},
        )

    async def get_cvd(
        self, symbol: str, exchange: str = "Binance", interval: str = "h1",
    ) -> Any:
        """Cumulative Volume Delta (CVD). Symbol: pair."""
        return await self._request(
            "futures/cvd/history",
            {"symbol": _to_pair(symbol), "exchange": exchange, "interval": interval},
        )

    async def get_coin_flow(self, symbol: str) -> Any:
        """Futures coin net flow across exchanges. Symbol: base."""
        return await self._request(
            "futures/netflow-list", {"symbol": _to_base(symbol)},
        )

    # ── 清算 ──

    async def get_liquidations(
        self, symbol: str, exchange: str = "Binance", interval: str = "h1",
    ) -> Any:
        """Liquidation history for one symbol. Symbol: pair."""
        return await self._request(
            "futures/liquidation/history",
            {"symbol": _to_pair(symbol), "exchange": exchange, "interval": interval},
        )

    async def get_liquidation_coin_list(self) -> Any:
        """All coins liquidation summary (24h/12h/4h/1h, long/short split).

        Returns ~941 coins. No symbol param needed.
        Great for: "which coins got liquidated the most?"
        """
        return await self._request("futures/liquidation/coin-list", {})

    # ── 期现差价 (Basis) ──

    async def get_basis_history(
        self, symbol: str, exchange: str = "Binance", interval: str = "h1",
    ) -> Any:
        """Futures-spot basis history. Symbol: pair.

        Returns: time, open_basis, close_basis, open_change, close_change.
        Use for: arbitrage analysis, market structure.
        """
        return await self._request(
            "futures/basis/history",
            {"symbol": _to_pair(symbol), "exchange": exchange, "interval": interval},
        )

    # ── 大额挂单 ──

    async def get_large_orderbook(
        self, symbol: str, exchange: str = "Binance",
    ) -> Any:
        """Large open limit orders from futures orderbook. Symbol: pair."""
        return await self._request(
            "futures/orderbook/large-limit-order",
            {"symbol": _to_pair(symbol), "exchange": exchange},
        )

    # ════════════════════════ Spot ════════════════════════

    async def get_spot_netflow(self, symbol: str) -> Any:
        """Spot net flow across exchanges. Symbol: base."""
        return await self._request(
            "spot/netflow-list", {"symbol": _to_base(symbol)},
        )

    async def get_spot_pairs(self, symbol: str) -> Any:
        """Spot trading pairs data by exchange.

        Returns per-exchange: price, 1h/4h/12h/24h volume & change, buy/sell split.
        Symbol: base (e.g. "BTC").
        """
        return await self._request(
            "spot/pairs-markets", {"symbol": _to_base(symbol)},
        )

    async def get_spot_supported_coins(self) -> Any:
        """List all 2500+ supported spot coins."""
        return await self._request("spot/supported-coins", {})

    # ════════════════════════ Options ════════════════════════

    async def get_option_info(self, symbol: str = "BTC") -> Any:
        """Options market overview: OI, volume, OI change by exchange.

        Symbol: base (e.g. "BTC", "ETH").
        """
        return await self._request(
            "option/info", {"symbol": _to_base(symbol)},
        )

    async def get_option_max_pain(
        self, symbol: str = "BTC", exchange: str = "Deribit",
    ) -> Any:
        """Options max pain price by expiry date.

        Returns per-expiry: max_pain_price, call/put OI, notional values.
        Great for: short-term price targets, options expiry impact.
        """
        return await self._request(
            "option/max-pain",
            {"symbol": _to_base(symbol), "exchange": exchange},
        )

    # ════════════════════════ Index ════════════════════════

    async def get_fear_greed(self) -> Any:
        """Fear & Greed index history."""
        return await self._request("index/fear-greed-history", {})

    async def get_ahr999(self) -> Any:
        """AHR999 Bitcoin valuation index history.

        Returns daily: ahr999_value, average_price, current_value.
        < 0.45 = strong buy zone, 0.45-1.2 = DCA zone, > 1.2 = overvalued.
        """
        return await self._request("index/ahr999", {})

    # ════════════════════════ Chain ════════════════════════

    async def get_whale_transfers(self, symbol: str, limit: int = 20) -> Any:
        """Recent whale transfer events. Symbol: base."""
        return await self._request(
            "chain/v2/whale-transfer",
            {"symbol": _to_base(symbol), "limit": limit},
        )

    # ════════════════════════ Coin ════════════════════════

    async def get_token_unlocks(self, per_page: int = 50, page: int = 1) -> Any:
        """Token unlock list with upcoming schedules.

        Returns: symbol, price, market_cap, next_unlock_date/tokens/usd,
                 next_unlock_of_circulating/supply, total_locked, etc.
        """
        return await self._request(
            "coin/unlock-list",
            {"per_page": int(per_page), "page": int(page)},
        )
