"""Finnhub tool — financial data: earnings calendar, market news, company profiles.

Requires API key (free tier: 60 calls/min).
This is a NATIVE data capability for the agent.

Actions:
  - earnings:     Upcoming/recent earnings reports with EPS estimates.
  - news:         Market news (general, forex, crypto categories).
  - crypto_news:  Crypto-specific news headlines.
  - company:      Company profile (for stock tickers).
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from typing import Any

import httpx
from loguru import logger

from getall.agent.tools.base import Tool

_BASE = "https://finnhub.io/api/v1"
_TIMEOUT = 15.0


class FinnhubTool(Tool):
    """Finnhub — earnings calendar, market news, company profiles. Requires API key."""

    def __init__(self, api_key: str = ""):
        self._api_key = api_key or os.environ.get("FINNHUB_API_KEY", "")

    @property
    def name(self) -> str:
        return "finnhub"

    @property
    def description(self) -> str:
        return (
            "Financial data from Finnhub: earnings calendar, market news (general/crypto/forex), "
            "and company profiles. "
            "Actions: 'earnings' (upcoming/recent earnings with EPS estimates), "
            "'news' (general market news), 'crypto_news' (crypto-specific headlines), "
            "'company' (company profile for stock tickers like AAPL, COIN, MSTR). "
            "Use for: tracking earnings dates that may impact crypto (e.g. COIN, MSTR), "
            "reading latest market/crypto news, researching companies."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "Action to perform",
                    "enum": ["earnings", "news", "crypto_news", "company"],
                },
                "symbol": {
                    "type": "string",
                    "description": (
                        "For 'company': stock ticker (e.g. 'AAPL', 'COIN', 'MSTR'). "
                        "For 'earnings': optional filter by symbol."
                    ),
                },
                "from_date": {
                    "type": "string",
                    "description": "For 'earnings': start date YYYY-MM-DD. Default: today.",
                },
                "to_date": {
                    "type": "string",
                    "description": "For 'earnings': end date YYYY-MM-DD. Default: 7 days from now.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results. Default: 20, max: 50",
                    "minimum": 1,
                    "maximum": 50,
                },
            },
            "required": ["action"],
        }

    async def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        """GET with auth + error handling."""
        if not self._api_key:
            return {"error": "FINNHUB_API_KEY not configured"}
        p = dict(params or {})
        p["token"] = self._api_key
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            resp = await client.get(f"{_BASE}{path}", params=p)
            if resp.status_code == 429:
                return {"error": "Finnhub rate limited. Try again in a minute."}
            resp.raise_for_status()
            return resp.json()

    async def execute(self, **kwargs: Any) -> str:
        action = kwargs.get("action", "")
        try:
            if action == "earnings":
                return await self._earnings(**kwargs)
            if action == "news":
                return await self._news(**kwargs)
            if action == "crypto_news":
                return await self._crypto_news(**kwargs)
            if action == "company":
                return await self._company(**kwargs)
            return json.dumps({"error": f"Unknown action '{action}'"})
        except httpx.HTTPStatusError as e:
            logger.error(f"Finnhub HTTP error: {e}")
            return json.dumps({"error": f"Finnhub API error: {e.response.status_code}"})
        except Exception as e:
            logger.error(f"Finnhub tool error: {e}")
            return json.dumps({"error": str(e)})

    # ── earnings ──

    async def _earnings(self, **kwargs: Any) -> str:
        today = datetime.utcnow().date()
        from_date = kwargs.get("from_date") or str(today)
        to_date = kwargs.get("to_date") or str(today + timedelta(days=7))
        limit = min(kwargs.get("limit", 20) or 20, 50)
        symbol_filter = (kwargs.get("symbol") or "").strip().upper()

        data = await self._get("/calendar/earnings", {"from": from_date, "to": to_date})
        if isinstance(data, dict) and "error" in data:
            return json.dumps(data)

        entries = data.get("earningsCalendar", [])

        # Filter by symbol if specified
        if symbol_filter:
            entries = [e for e in entries if (e.get("symbol") or "").upper() == symbol_filter]

        # Sort by date
        entries.sort(key=lambda e: e.get("date", ""), reverse=True)

        results = []
        for e in entries[:limit]:
            results.append({
                "symbol": e.get("symbol"),
                "date": e.get("date"),
                "hour": e.get("hour"),  # bmo=before market open, amc=after market close
                "quarter": e.get("quarter"),
                "year": e.get("year"),
                "eps_estimate": e.get("epsEstimate"),
                "eps_actual": e.get("epsActual"),
                "revenue_estimate": e.get("revenueEstimate"),
                "revenue_actual": e.get("revenueActual"),
            })

        return json.dumps({
            "action": "earnings",
            "from": from_date,
            "to": to_date,
            "count": len(results),
            "earnings": results,
        })

    # ── news ──

    async def _news(self, **kwargs: Any) -> str:
        limit = min(kwargs.get("limit", 20) or 20, 50)

        data = await self._get("/news", {"category": "general"})
        if isinstance(data, dict) and "error" in data:
            return json.dumps(data)
        if not isinstance(data, list):
            return json.dumps({"error": "Unexpected response format"})

        results = []
        for article in data[:limit]:
            results.append({
                "headline": article.get("headline"),
                "source": article.get("source"),
                "summary": (article.get("summary") or "")[:300],
                "url": article.get("url"),
                "datetime": article.get("datetime"),
                "category": article.get("category"),
            })

        return json.dumps({"action": "news", "category": "general", "count": len(results), "articles": results})

    # ── crypto_news ──

    async def _crypto_news(self, **kwargs: Any) -> str:
        limit = min(kwargs.get("limit", 20) or 20, 50)

        data = await self._get("/news", {"category": "crypto"})
        if isinstance(data, dict) and "error" in data:
            return json.dumps(data)
        if not isinstance(data, list):
            return json.dumps({"error": "Unexpected response format"})

        results = []
        for article in data[:limit]:
            results.append({
                "headline": article.get("headline"),
                "source": article.get("source"),
                "summary": (article.get("summary") or "")[:300],
                "url": article.get("url"),
                "datetime": article.get("datetime"),
            })

        return json.dumps({"action": "crypto_news", "count": len(results), "articles": results})

    # ── company ──

    async def _company(self, **kwargs: Any) -> str:
        symbol = (kwargs.get("symbol") or "").strip().upper()
        if not symbol:
            return json.dumps({"error": "symbol is required (e.g. 'AAPL', 'COIN')"})

        data = await self._get("/stock/profile2", {"symbol": symbol})
        if isinstance(data, dict) and "error" in data:
            return json.dumps(data)
        if not data or not isinstance(data, dict) or not data.get("name"):
            return json.dumps({"error": f"Company not found: {symbol}"})

        return json.dumps({
            "action": "company",
            "symbol": symbol,
            "name": data.get("name"),
            "country": data.get("country"),
            "exchange": data.get("exchange"),
            "industry": data.get("finnhubIndustry"),
            "ipo_date": data.get("ipo"),
            "market_cap_m": round(float(data.get("marketCapitalization") or 0), 2),
            "shares_outstanding_m": round(float(data.get("shareOutstanding") or 0), 2),
            "logo": data.get("logo"),
            "url": data.get("weburl"),
        })
