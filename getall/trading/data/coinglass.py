"""Coinglass API v4 全量衍生品数据适配器 (代理模式).

代理地址: http://18.176.51.231:5000 (映射到 open-api-v4.coinglass.com/api/)
所有方法均为 async, 使用 httpx 发起请求.

重要: Coinglass v4 API 路径使用 **kebab-case** (连字符), 而非 camelCase.
      示例: futures/open-interest/history (非 futures/openInterest/history)

Symbol 格式说明:
  - 带 exchange 参数的 history 端点 → 使用交易对格式, 如 "BTCUSDT"
  - exchange-list / aggregated 端点 → 使用基础币种, 如 "BTC"
  适配器内部会根据端点类型自动转换.
"""

from typing import Any

import httpx
from loguru import logger


# 默认请求超时 (秒)
_TIMEOUT = 15.0


def _to_pair(symbol: str) -> str:
    """将任意格式的 symbol 归一化为交易对格式 (BTCUSDT).

    接受: "BTC", "BTC/USDT", "BTC/USDT:USDT", "BTCUSDT" → "BTCUSDT"
    """
    s = symbol.upper().replace(" ", "")
    # 去除 :USDT 后缀 (合约标记)
    s = s.split(":")[0]
    # 去除 / 分隔符
    s = s.replace("/", "")
    # 如果只有 base coin, 追加 USDT
    if not s.endswith(("USDT", "USDC", "BUSD", "USD")):
        s = s + "USDT"
    return s


def _to_base(symbol: str) -> str:
    """将任意格式的 symbol 归一化为基础币种 (BTC).

    接受: "BTC", "BTC/USDT", "BTCUSDT", "BTC/USDT:USDT" → "BTC"
    """
    s = symbol.upper().replace(" ", "")
    s = s.split(":")[0]
    s = s.split("/")[0]
    # 去除常见 quote 后缀
    for suffix in ("USDT", "USDC", "BUSD", "USD"):
        if s.endswith(suffix) and len(s) > len(suffix):
            s = s[: -len(suffix)]
            break
    return s


class CoinglassAdapter:
    """
    Coinglass open-api-v4 adapter.

    Provides derivatives data: OI, funding, long/short ratios,
    taker buy/sell, liquidations, on-chain indicators, etc.

    代理端点路径使用 kebab-case, 如 futures/open-interest/history.
    """

    def __init__(self, base_url: str, api_key: str = ""):
        """
        Initialize Coinglass adapter.

        Args:
            base_url: Proxy or direct base URL (no trailing slash).
            api_key: CG-API-KEY for authentication.
        """
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key

    # ──────────────────────── 底层请求 ────────────────────────

    async def _request(
        self, endpoint: str, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Send GET request to Coinglass API.

        Args:
            endpoint: API endpoint path (e.g. "futures/open-interest/history").
            params: Query parameters.

        Returns:
            Parsed JSON response dict or error dict.
        """
        url = f"{self._base_url}/{endpoint.lstrip('/')}"
        headers = {
            "CG-API-KEY": self._api_key,
            "accept": "application/json",
            "accept-encoding": "identity",
        }
        # 代理兼容: api_key 同时作为 query param
        query = dict(params or {})
        if self._api_key:
            query["api_key"] = self._api_key

        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                resp = await client.get(url, params=query, headers=headers)
                resp.raise_for_status()
                data = resp.json()
                # Coinglass 标准响应: {"code": 0, "msg": "success", "data": ...}
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

    # ──────────────────────── OI (持仓量) ────────────────────────

    async def get_oi_history(
        self,
        symbol: str,
        exchange: str = "Binance",
        interval: str = "1d",
        limit: int = 100,
    ) -> Any:
        """Get open interest history for a symbol on a specific exchange.

        Symbol format: pair (e.g. "BTCUSDT").
        """
        return await self._request(
            "futures/open-interest/history",
            {
                "symbol": _to_pair(symbol),
                "exchange": exchange,
                "interval": interval,
                "limit": limit,
                "unit": "usd",
            },
        )

    async def get_aggregated_oi(
        self, symbol: str, interval: str = "1d", limit: int = 50
    ) -> Any:
        """Get aggregated OI across all exchanges.

        Symbol format: base coin (e.g. "BTC").
        """
        return await self._request(
            "futures/open-interest/aggregated-history",
            {"symbol": _to_base(symbol), "interval": interval, "limit": limit},
        )

    async def get_exchange_oi(self, symbol: str) -> Any:
        """Get OI breakdown by exchange.

        Symbol format: base coin (e.g. "BTC").
        """
        return await self._request(
            "futures/open-interest/exchange-list",
            {"symbol": _to_base(symbol)},
        )

    # ──────────────────────── 资金费率 ────────────────────────

    async def get_funding_rate_history(
        self,
        symbol: str,
        exchange: str = "Binance",
        interval: str = "h8",
    ) -> Any:
        """Get funding rate history.

        Symbol format: pair (e.g. "BTCUSDT").
        """
        return await self._request(
            "futures/funding-rate/history",
            {"symbol": _to_pair(symbol), "exchange": exchange, "interval": interval},
        )

    async def get_funding_rate_exchange(self, symbol: str) -> Any:
        """Get current funding rates across exchanges.

        Symbol format: base coin (e.g. "BTC").
        """
        return await self._request(
            "futures/funding-rate/exchange-list",
            {"symbol": _to_base(symbol)},
        )

    # ──────────────────────── 多空比 ────────────────────────

    async def get_long_short_ratio(
        self,
        symbol: str,
        exchange: str = "Binance",
        interval: str = "h1",
    ) -> Any:
        """Get global long/short account ratio.

        Symbol format: pair (e.g. "BTCUSDT").
        """
        return await self._request(
            "futures/global-long-short-account-ratio/history",
            {"symbol": _to_pair(symbol), "exchange": exchange, "interval": interval},
        )

    async def get_top_long_short_account(
        self, symbol: str, exchange: str = "Binance", interval: str = "h1"
    ) -> Any:
        """Get top trader long/short ratio (accounts).

        Symbol format: pair (e.g. "BTCUSDT").
        """
        return await self._request(
            "futures/top-long-short-account-ratio/history",
            {"symbol": _to_pair(symbol), "exchange": exchange, "interval": interval},
        )

    async def get_top_long_short_position(
        self, symbol: str, exchange: str = "Binance", interval: str = "h1"
    ) -> Any:
        """Get top trader long/short ratio (positions).

        Symbol format: pair (e.g. "BTCUSDT").
        """
        return await self._request(
            "futures/top-long-short-position-ratio/history",
            {"symbol": _to_pair(symbol), "exchange": exchange, "interval": interval},
        )

    async def get_net_long_short(self, symbol: str) -> Any:
        """Get net long/short ratio overview.

        ⚠️ 当前代理不支持 chart 端点, 回退到 global-long-short-account-ratio/history.
        """
        return await self._request(
            "futures/global-long-short-account-ratio/history",
            {"symbol": _to_pair(symbol), "exchange": "Binance", "interval": "h1"},
        )

    async def get_net_position_history(
        self,
        symbol: str,
        exchange: str = "Binance",
        interval: str = "h1",
    ) -> Any:
        """Get net long/short position history (真实净多头/净空头头寸变化).

        Symbol format: pair (e.g. "BTCUSDT").
        """
        return await self._request(
            "futures/v2/net-position/history",
            {"symbol": _to_pair(symbol), "exchange": exchange, "interval": interval},
        )

    # ──────────────────────── Taker / CVD ────────────────────────

    async def get_taker_buy_sell(
        self,
        symbol: str,
        exchange: str = "Binance",
        interval: str = "h1",
        limit: int = 10,
    ) -> Any:
        """Get taker buy/sell volume history (改用 history 端点, exchange-list 数据异常).

        Symbol format: pair (e.g. "BTCUSDT").
        interval: 时间间隔, 如 "m5", "m15", "h1", "h4".
        limit: 返回条数.
        """
        return await self._request(
            "futures/taker-buy-sell-volume/history",
            {
                "symbol": _to_pair(symbol),
                "exchange": exchange,
                "interval": interval,
                "limit": limit,
            },
        )

    async def get_taker_buy_sell_history(
        self,
        symbol: str,
        exchange: str = "Binance",
        interval: str = "h1",
    ) -> Any:
        """Get taker buy/sell volume history.

        Symbol format: pair (e.g. "BTCUSDT").
        """
        return await self._request(
            "futures/taker-buy-sell-volume/history",
            {"symbol": _to_pair(symbol), "exchange": exchange, "interval": interval},
        )

    async def get_cvd(
        self,
        symbol: str,
        exchange: str = "Binance",
        interval: str = "h1",
    ) -> Any:
        """Get cumulative volume delta (CVD).

        Symbol format: pair (e.g. "BTCUSDT").
        """
        return await self._request(
            "futures/cvd/history",
            {"symbol": _to_pair(symbol), "exchange": exchange, "interval": interval},
        )

    async def get_coin_flow(self, symbol: str) -> Any:
        """Get futures coin net flow data (inflow/outflow) across exchanges.

        Coinglass v4 路径: futures/netflow-list (非 futures/coin-flow)
        Symbol format: base coin (e.g. "BTC").
        """
        return await self._request(
            "futures/netflow-list",
            {"symbol": _to_base(symbol)},
        )

    async def get_spot_netflow(self, symbol: str) -> Any:
        """Get spot net flow across exchanges (现货资金流入流出).

        Symbol format: base coin (e.g. "BTC").
        """
        return await self._request(
            "spot/netflow-list",
            {"symbol": _to_base(symbol)},
        )

    # ──────────────────────── 爆仓 / 清算 ────────────────────────

    async def get_liquidations(
        self, symbol: str, exchange: str = "Binance", interval: str = "h1"
    ) -> Any:
        """Get aggregated liquidation data.

        Symbol format: pair (e.g. "BTCUSDT").
        """
        return await self._request(
            "futures/liquidation/history",
            {"symbol": _to_pair(symbol), "exchange": exchange, "interval": interval},
        )

    async def get_liquidation_orders(self, symbol: str, limit: int = 50) -> Any:
        """Get recent large liquidation orders.

        Symbol format: base coin (e.g. "BTC").
        """
        return await self._request(
            "futures/liquidation/order",
            {"symbol": _to_base(symbol), "limit": limit},
        )

    # ──────────────────────── 大额挂单 / 链上 / 指标 ────────────────────────

    async def get_large_orderbook(
        self, symbol: str, exchange: str = "Binance"
    ) -> Any:
        """Get large open limit orders from futures orderbook.

        Coinglass v4 路径: futures/orderbook/large-limit-order
        """
        return await self._request(
            "futures/orderbook/large-limit-order",
            {"symbol": _to_pair(symbol), "exchange": exchange},
        )

    async def get_whale_transfers(self, symbol: str, limit: int = 20) -> Any:
        """Get recent whale transfer events.

        Coinglass v4 路径: chain/v2/whale-transfer
        """
        return await self._request(
            "chain/v2/whale-transfer",
            {"symbol": _to_base(symbol), "limit": limit},
        )

    async def get_fear_greed(self) -> Any:
        """Get Fear & Greed index history.

        Coinglass v4 路径: index/fear-greed-history
        """
        return await self._request("index/fear-greed-history", {})

    async def get_coin_change(self, symbol: str, interval: str = "h24") -> Any:
        """Get coin price change statistics.

        ⚠️ Coinglass v4 API 没有 indicator/coin-change 端点.
        可使用 futures/coins-markets 或 spot/coins-markets 替代.
        """
        return {"error": "Endpoint not available in Coinglass v4 API", "endpoint": "indicator/coin-change"}

    async def get_coins_markets(
        self,
        exchange_list: str | None = None,
        per_page: int | None = None,
        page: int | None = None,
    ) -> Any:
        """Get futures coins market overview (全市场合约数据概览).

        返回所有币种的: 价格、OI、资金费率、多空比、成交量、清算等汇总数据.
        可用于: 资金费率排行、OI 排行、清算排行等全市场扫描.

        Args:
            exchange_list: Optional comma-separated exchanges, e.g. "Binance,OKX".
            per_page: Pagination size (Coinglass uses per_page/page).
            page: Pagination index (1-based).
        """
        params: dict[str, Any] = {}
        if exchange_list:
            params["exchange_list"] = exchange_list
        if per_page is not None:
            params["per_page"] = int(per_page)
        if page is not None:
            params["page"] = int(page)
        return await self._request("futures/coins-markets", params)

    async def get_spot_coins_markets(
        self,
        per_page: int | None = None,
        page: int | None = None,
    ) -> Any:
        """Get spot coins market overview (全市场现货数据概览)."""
        params: dict[str, Any] = {}
        if per_page is not None:
            params["per_page"] = int(per_page)
        if page is not None:
            params["page"] = int(page)
        return await self._request("spot/coins-markets", params)
