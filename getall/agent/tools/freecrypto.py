"""FreeCryptoAPI tool — real-time crypto prices from Binance aggregator.

Requires API key (free tier: 100K req/month).
Free tier supports: getData (live prices), getCryptoList (supported symbols).
Note: Technical analysis, OHLC, Fear&Greed, etc. require a paid plan.

This is a NATIVE data capability for the agent.

Actions:
  - price:    Real-time price + 24h change for one or more symbols (Binance source).
  - symbols:  List all 3000+ supported cryptocurrency symbols.
"""

from __future__ import annotations

import json
import os
from typing import Any

import httpx
from loguru import logger

from getall.agent.tools.base import Tool

_BASE = "https://api.freecryptoapi.com/v1"
_TIMEOUT = 15.0


class FreeCryptoTool(Tool):
    """FreeCryptoAPI — real-time crypto prices (3000+ coins, Binance source). Requires API key."""

    def __init__(self, api_key: str = ""):
        self._api_key = api_key or os.environ.get("FREECRYPTO_API_KEY", "")

    @property
    def name(self) -> str:
        return "freecrypto"

    @property
    def description(self) -> str:
        return (
            "Real-time cryptocurrency prices from FreeCryptoAPI (3000+ coins, Binance source). "
            "Millisecond latency, aggregated data. "
            "Actions: 'price' (live price + 24h change for one or more symbols), "
            "'symbols' (list all supported cryptocurrencies). "
            "Symbol format: 'BTC', 'ETH', 'SOL' (just the base symbol, no quote). "
            "Use as a fast price check alternative when other sources are slow or rate-limited."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "Action to perform",
                    "enum": ["price", "symbols"],
                },
                "symbol": {
                    "type": "string",
                    "description": (
                        "For 'price': one or more symbols, comma-separated "
                        "(e.g. 'BTC', 'BTC,ETH,SOL'). "
                        "Use base symbol only, no quote currency."
                    ),
                },
                "limit": {
                    "type": "integer",
                    "description": "For 'symbols': max results to return. Default: 50",
                    "minimum": 1,
                    "maximum": 200,
                },
            },
            "required": ["action"],
        }

    async def _get(self, endpoint: str, params: dict[str, Any] | None = None) -> Any:
        """GET with auth + error handling."""
        if not self._api_key:
            return {"status": False, "error": "FREECRYPTO_API_KEY not configured"}
        p = dict(params or {})
        p["token"] = self._api_key
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.get(f"{_BASE}/{endpoint}", params=p)
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, dict) and data.get("status") is False:
                return {"error": data.get("error", "Unknown error")}
            return data

    async def execute(self, **kwargs: Any) -> str:
        action = kwargs.get("action", "")
        try:
            if action == "price":
                return await self._price(**kwargs)
            if action == "symbols":
                return await self._symbols(**kwargs)
            return json.dumps({"error": f"Unknown action '{action}'"})
        except httpx.HTTPStatusError as e:
            logger.error(f"FreeCrypto HTTP error: {e}")
            return json.dumps({"error": f"FreeCryptoAPI error: {e.response.status_code}"})
        except Exception as e:
            logger.error(f"FreeCrypto tool error: {e}")
            return json.dumps({"error": str(e)})

    # ── price ──

    async def _price(self, **kwargs: Any) -> str:
        symbol = (kwargs.get("symbol") or "").strip().upper()
        if not symbol:
            return json.dumps({"error": "symbol is required (e.g. 'BTC' or 'BTC,ETH,SOL')"})

        # FreeCryptoAPI only supports single-symbol queries in free tier
        # Split and query each individually if multiple
        symbols = [s.strip() for s in symbol.split(",") if s.strip()]
        results = []

        for sym in symbols[:10]:  # Cap at 10 to avoid excessive API calls
            data = await self._get("getData", {"symbol": sym})
            if isinstance(data, dict) and "error" in data:
                results.append({"symbol": sym, "error": data["error"]})
                continue

            entries = data.get("symbols", [])
            if not entries:
                results.append({"symbol": sym, "error": "No data"})
                continue

            for entry in entries:
                results.append({
                    "symbol": entry.get("symbol"),
                    "price": float(entry.get("last") or 0),
                    "high_24h": float(entry.get("highest") or 0),
                    "low_24h": float(entry.get("lowest") or 0),
                    "change_24h_pct": round(float(entry.get("daily_change_percentage") or 0), 2),
                    "source": entry.get("source_exchange"),
                    "timestamp": entry.get("date"),
                })

        return json.dumps({"action": "price", "count": len(results), "prices": results})

    # ── symbols ──

    async def _symbols(self, **kwargs: Any) -> str:
        limit = min(kwargs.get("limit", 50) or 50, 200)

        data = await self._get("getCryptoList")
        if isinstance(data, dict) and "error" in data:
            return json.dumps(data)

        entries = data.get("result", [])
        total = data.get("resultset_size", len(entries))

        results = []
        for entry in entries[:limit]:
            results.append({
                "symbol": entry.get("symbol"),
                "name": entry.get("name"),
                "source": entry.get("source"),
            })

        return json.dumps({
            "action": "symbols",
            "total_supported": total,
            "returned": len(results),
            "symbols": results,
        })
