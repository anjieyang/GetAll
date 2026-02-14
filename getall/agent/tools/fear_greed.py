"""Fear & Greed Index tool — crypto market sentiment.

Free, no API key required.
Sources:
  - Alternative.me: Classic Crypto Fear & Greed Index (daily granularity).
  - CoinyBubble:    Higher-frequency sentiment (5-min updates).

This is a NATIVE data capability for the agent.

Actions:
  - current:  Latest Fear & Greed value + classification.
  - history:  Historical daily values (Alternative.me).
  - realtime: Recent high-frequency data (CoinyBubble, 5-min intervals).
"""

from __future__ import annotations

import json
from typing import Any

import httpx
from loguru import logger

from getall.agent.tools.base import Tool

_ALT_ME_URL = "https://api.alternative.me/fng/"
_COINY_BASE = "https://api.coinybubble.com/v1"
_TIMEOUT = 15.0


async def _get(url: str, params: dict[str, Any] | None = None) -> Any:
    """GET with error handling."""
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        return resp.json()


class FearGreedTool(Tool):
    """Fear & Greed Index — crypto market sentiment from Alternative.me + CoinyBubble. No API key."""

    @property
    def name(self) -> str:
        return "fear_greed"

    @property
    def description(self) -> str:
        return (
            "Crypto Fear & Greed Index — market sentiment indicator (0=Extreme Fear, 100=Extreme Greed). "
            "No API key needed. "
            "Actions: 'current' (latest value + classification), "
            "'history' (daily historical data from Alternative.me, up to 365 days), "
            "'realtime' (high-frequency 5-min data from CoinyBubble, up to 8 days). "
            "Use for: market sentiment analysis, timing entries/exits, risk assessment. "
            "Values: 0-24 Extreme Fear, 25-49 Fear, 50 Neutral, 51-74 Greed, 75-100 Extreme Greed."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "Action to perform",
                    "enum": ["current", "history", "realtime"],
                },
                "days": {
                    "type": "integer",
                    "description": "For 'history': number of days (1-365). Default: 30",
                    "minimum": 1,
                    "maximum": 365,
                },
                "hours": {
                    "type": "integer",
                    "description": "For 'realtime': hours of data (1-192). Default: 24",
                    "minimum": 1,
                    "maximum": 192,
                },
            },
            "required": ["action"],
        }

    async def execute(self, **kwargs: Any) -> str:
        action = kwargs.get("action", "current")
        try:
            if action == "current":
                return await self._current()
            if action == "history":
                return await self._history(**kwargs)
            if action == "realtime":
                return await self._realtime(**kwargs)
            return json.dumps({"error": f"Unknown action '{action}'"})
        except httpx.HTTPStatusError as e:
            logger.error(f"Fear&Greed HTTP error: {e}")
            return json.dumps({"error": f"API error: {e.response.status_code}"})
        except Exception as e:
            logger.error(f"Fear&Greed tool error: {e}")
            return json.dumps({"error": str(e)})

    # ── current ──

    async def _current(self) -> str:
        """Get latest value from both sources, prefer CoinyBubble for freshness."""
        results: dict[str, Any] = {"action": "current"}

        # Alternative.me (daily)
        try:
            alt_data = await _get(_ALT_ME_URL, {"limit": "1"})
            entry = alt_data.get("data", [{}])[0]
            results["value"] = int(entry.get("value", 0))
            results["classification"] = entry.get("value_classification", "")
            results["timestamp"] = int(entry.get("timestamp", 0))
            results["source"] = "alternative.me"
        except Exception as e:
            logger.warning(f"Alternative.me failed: {e}")

        # CoinyBubble (realtime, may have fresher data)
        try:
            coiny_data = await _get(f"{_COINY_BASE}/latest")
            if isinstance(coiny_data, dict) and "value" in coiny_data:
                results["realtime_value"] = coiny_data.get("value")
                results["realtime_classification"] = coiny_data.get("value_classification", "")
                results["btc_price"] = coiny_data.get("btc_price")
                results["realtime_source"] = "coinybubble"
        except Exception as e:
            logger.debug(f"CoinyBubble latest failed (optional): {e}")

        if "value" not in results and "realtime_value" not in results:
            return json.dumps({"error": "Failed to fetch Fear & Greed Index from all sources"})

        return json.dumps(results)

    # ── history ──

    async def _history(self, **kwargs: Any) -> str:
        days = min(max(kwargs.get("days", 30) or 30, 1), 365)

        data = await _get(_ALT_ME_URL, {"limit": str(days)})
        entries = data.get("data", [])

        results = []
        for entry in entries:
            results.append({
                "value": int(entry.get("value", 0)),
                "classification": entry.get("value_classification", ""),
                "timestamp": int(entry.get("timestamp", 0)),
            })

        return json.dumps({
            "action": "history",
            "days": days,
            "count": len(results),
            "source": "alternative.me",
            "data": results,
        })

    # ── realtime ──

    async def _realtime(self, **kwargs: Any) -> str:
        hours = min(max(kwargs.get("hours", 24) or 24, 1), 192)

        data = await _get(f"{_COINY_BASE}/history/5min", {"hours": str(hours)})

        if isinstance(data, list):
            entries = data
        elif isinstance(data, dict):
            entries = data.get("data", data.get("history", []))
        else:
            return json.dumps({"error": "Unexpected response format from CoinyBubble"})

        results = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            results.append({
                "value": entry.get("value"),
                "timestamp": entry.get("timestamp"),
                "btc_price": entry.get("btc_price"),
            })

        return json.dumps({
            "action": "realtime",
            "hours": hours,
            "interval": "5min",
            "count": len(results),
            "source": "coinybubble",
            "data": results,
        })
