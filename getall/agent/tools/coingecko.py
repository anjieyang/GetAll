"""CoinGecko tool — general-purpose crypto data source.

Covers 28M+ tokens across 1700+ exchanges including DeFi / small-cap coins.
This is a NATIVE data capability for the agent, not limited to backtesting.

Actions:
  - search:   Find CoinGecko coin ID for a symbol or name.
  - price:    Get current price + 24h change for one or more coins.
  - ohlcv:    Fetch OHLCV candle data for a coin.
  - markets:  Get top coins by market cap (or filtered list).
  - trending: Currently trending coins on CoinGecko (what people are searching).
  - global:   Global crypto market stats (total mcap, volume, BTC dominance, etc.).
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import httpx
from loguru import logger

from getall.agent.tools.base import Tool

_BASE = "https://api.coingecko.com/api/v3"

# Simple rate limiter: max 25 req/min (stay under 30 limit)
_MIN_INTERVAL = 2.5  # seconds between requests
_last_request_ts: float = 0.0


async def _throttle() -> None:
    global _last_request_ts
    now = time.monotonic()
    wait = _MIN_INTERVAL - (now - _last_request_ts)
    if wait > 0:
        await asyncio.sleep(wait)
    _last_request_ts = time.monotonic()


async def _get(path: str, params: dict[str, Any] | None = None, api_key: str = "") -> Any:
    """GET with throttle + error handling."""
    await _throttle()
    headers: dict[str, str] = {"accept": "application/json"}
    if api_key:
        headers["x-cg-demo-api-key"] = api_key
    url = f"{_BASE}{path}"
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(url, params=params, headers=headers)
        if resp.status_code == 429:
            logger.warning("CoinGecko rate limited, waiting 60s")
            await asyncio.sleep(60)
            resp = await client.get(url, params=params, headers=headers)
        resp.raise_for_status()
        return resp.json()


class CoinGeckoTool(Tool):
    """CoinGecko — general-purpose crypto data (price, OHLCV, market overview, coin search)."""

    def __init__(self, api_key: str = ""):
        self._api_key = api_key

    @property
    def name(self) -> str:
        return "coingecko"

    @property
    def description(self) -> str:
        return (
            "General-purpose crypto data from CoinGecko (28M+ tokens, 1700+ exchanges, "
            "10+ years history). Use for: price checks, OHLCV candles, market overview, "
            "coin search, trending coins, global market stats. "
            "Covers DeFi tokens, small-cap coins, and aggregated cross-exchange data. "
            "Actions: 'search' (find coin ID), 'price' (current price), 'ohlcv' (candles), "
            "'markets' (top coins by market cap), 'trending' (trending coins right now), "
            "'global' (total market cap, BTC dominance, etc.). "
            "Note: CoinGecko uses its own coin IDs (e.g. 'bitcoin', 'ethereum'). "
            "Use 'search' first if you only know the ticker symbol."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "Action to perform",
                    "enum": ["search", "price", "ohlcv", "markets", "trending", "global"],
                },
                "query": {
                    "type": "string",
                    "description": "For 'search': coin name or ticker (e.g. 'bitcoin', 'BTC', 'uniswap')",
                },
                "coin_ids": {
                    "type": "string",
                    "description": (
                        "For 'price': comma-separated CoinGecko IDs "
                        "(e.g. 'bitcoin,ethereum,solana')"
                    ),
                },
                "coin_id": {
                    "type": "string",
                    "description": (
                        "For 'ohlcv': single CoinGecko coin ID (e.g. 'bitcoin'). "
                        "Use 'search' first if you only know the ticker."
                    ),
                },
                "vs_currency": {
                    "type": "string",
                    "description": "Quote currency. Default: 'usd'",
                },
                "days": {
                    "type": "string",
                    "description": "For 'ohlcv': history length — '1','7','30','90','180','365','max'. Default: '180'",
                },
                "per_page": {
                    "type": "integer",
                    "description": "For 'markets': number of coins. Default: 20, max: 250",
                },
                "page": {
                    "type": "integer",
                    "description": "For 'markets': page number. Default: 1",
                },
            },
            "required": ["action"],
        }

    async def execute(self, **kwargs: Any) -> str:
        action = kwargs["action"]
        try:
            if action == "search":
                return await self._search(**kwargs)
            if action == "price":
                return await self._price(**kwargs)
            if action == "ohlcv":
                return await self._ohlcv(**kwargs)
            if action == "markets":
                return await self._markets(**kwargs)
            if action == "trending":
                return await self._trending()
            if action == "global":
                return await self._global()
            return json.dumps({"error": f"Unknown action '{action}'"})
        except httpx.HTTPStatusError as e:
            logger.error(f"CoinGecko HTTP error: {e}")
            return json.dumps({"error": f"CoinGecko API error: {e.response.status_code}"})
        except Exception as e:
            logger.error(f"CoinGecko tool error: {e}")
            return json.dumps({"error": str(e)})

    # ── search ──

    async def _search(self, **kwargs: Any) -> str:
        query = kwargs.get("query", "")
        if not query:
            return json.dumps({"error": "query is required for search"})
        data = await _get("/search", {"query": query}, self._api_key)
        coins = data.get("coins", [])[:10]
        results = [
            {
                "id": c["id"],
                "symbol": c["symbol"],
                "name": c["name"],
                "market_cap_rank": c.get("market_cap_rank"),
            }
            for c in coins
        ]
        return json.dumps({"results": results, "query": query})

    # ── price ──

    async def _price(self, **kwargs: Any) -> str:
        coin_ids = kwargs.get("coin_ids", "")
        if not coin_ids:
            return json.dumps({"error": "coin_ids is required (e.g. 'bitcoin,ethereum')"})
        vs = kwargs.get("vs_currency", "usd").lower()
        data = await _get(
            "/simple/price",
            {
                "ids": coin_ids,
                "vs_currencies": vs,
                "include_24hr_change": "true",
                "include_24hr_vol": "true",
                "include_market_cap": "true",
            },
            self._api_key,
        )
        return json.dumps(data)

    # ── ohlcv ──

    async def _ohlcv(self, **kwargs: Any) -> str:
        coin_id = kwargs.get("coin_id", "")
        if not coin_id:
            return json.dumps({"error": "coin_id is required. Use action='search' first."})
        vs = kwargs.get("vs_currency", "usd").lower()
        days = kwargs.get("days", "180")

        data = await _get(
            f"/coins/{coin_id}/ohlc",
            {"vs_currency": vs, "days": days},
            self._api_key,
        )
        if not isinstance(data, list) or not data:
            return json.dumps({"error": f"No OHLCV data for {coin_id}", "coin_id": coin_id})

        # CoinGecko returns [[timestamp_ms, open, high, low, close], ...]
        candles = []
        for row in data:
            if len(row) >= 5:
                candles.append({
                    "timestamp": int(row[0]),
                    "open": float(row[1]),
                    "high": float(row[2]),
                    "low": float(row[3]),
                    "close": float(row[4]),
                    "volume": 0.0,  # CoinGecko OHLC doesn't include volume
                })

        return json.dumps({
            "coin_id": coin_id,
            "symbol": coin_id,
            "vs_currency": vs,
            "days": days,
            "candles": len(candles),
            "ohlcv": candles,
        })

    # ── markets ──

    async def _markets(self, **kwargs: Any) -> str:
        vs = kwargs.get("vs_currency", "usd").lower()
        per_page = min(kwargs.get("per_page", 20), 250)
        page = kwargs.get("page", 1)
        data = await _get(
            "/coins/markets",
            {
                "vs_currency": vs,
                "order": "market_cap_desc",
                "per_page": per_page,
                "page": page,
                "sparkline": "false",
            },
            self._api_key,
        )
        results = [
            {
                "id": c["id"],
                "symbol": c["symbol"],
                "name": c["name"],
                "current_price": c.get("current_price"),
                "market_cap": c.get("market_cap"),
                "market_cap_rank": c.get("market_cap_rank"),
                "total_volume": c.get("total_volume"),
                "price_change_24h_pct": c.get("price_change_percentage_24h"),
            }
            for c in (data if isinstance(data, list) else [])
        ]
        return json.dumps({"coins": results, "page": page, "per_page": per_page})

    # ── trending ──

    async def _trending(self) -> str:
        data = await _get("/search/trending", api_key=self._api_key)
        coins = data.get("coins", [])
        results = []
        for item in coins[:15]:
            c = item.get("item", {})
            results.append({
                "id": c.get("id"),
                "symbol": c.get("symbol"),
                "name": c.get("name"),
                "market_cap_rank": c.get("market_cap_rank"),
                "price_btc": c.get("price_btc"),
                "score": c.get("score"),
            })
        return json.dumps({"action": "trending", "count": len(results), "coins": results})

    # ── global ──

    async def _global(self) -> str:
        data = await _get("/global", api_key=self._api_key)
        gd = data.get("data", {})
        mcap_pct = gd.get("market_cap_percentage", {})
        return json.dumps({
            "action": "global",
            "active_cryptocurrencies": gd.get("active_cryptocurrencies"),
            "total_market_cap_usd": round(float(
                (gd.get("total_market_cap") or {}).get("usd", 0)
            ), 2),
            "total_volume_24h_usd": round(float(
                (gd.get("total_volume") or {}).get("usd", 0)
            ), 2),
            "btc_dominance": round(float(mcap_pct.get("btc", 0)), 2),
            "eth_dominance": round(float(mcap_pct.get("eth", 0)), 2),
            "market_cap_change_24h_pct": gd.get("market_cap_change_percentage_24h_usd"),
            "markets": gd.get("markets"),
        })
