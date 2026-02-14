"""yfinance tool — zero-config market data for mainstream assets.

No API key required. Covers major cryptocurrencies + stocks + forex via Yahoo Finance.
This is a NATIVE data capability, not limited to backtesting.

Symbol format: BTC-USD, ETH-USD, AAPL, EURUSD=X, GC=F (gold futures), etc.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from loguru import logger

from getall.agent.tools.base import Tool


class YFinanceTool(Tool):
    """yfinance — market data for mainstream crypto, stocks, forex. No API key needed."""

    @property
    def name(self) -> str:
        return "yfinance"

    @property
    def description(self) -> str:
        return (
            "Fetch market data from Yahoo Finance. No API key needed. "
            "Covers: mainstream crypto (BTC-USD, ETH-USD), stocks (AAPL, TSLA), "
            "forex (EURUSD=X), commodities (GC=F gold). "
            "Actions: 'ohlcv' (candles), 'price' (current quote). "
            "5+ years of daily history for most assets. "
            "Note: each data source uses its own symbol format. "
            "Yahoo uses 'BTC-USD' not 'BTCUSDT'. You decide the correct format."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "Action: 'ohlcv' or 'price'",
                    "enum": ["ohlcv", "price"],
                },
                "symbol": {
                    "type": "string",
                    "description": (
                        "Yahoo Finance symbol. Crypto: 'BTC-USD', 'ETH-USD'. "
                        "Stocks: 'AAPL'. Forex: 'EURUSD=X'. Gold: 'GC=F'. "
                        "YOU decide the correct format for this data source."
                    ),
                },
                "period": {
                    "type": "string",
                    "description": "For 'ohlcv': '1mo','3mo','6mo','1y','2y','5y','max'. Default: '1y'",
                },
                "interval": {
                    "type": "string",
                    "description": "For 'ohlcv': '1h','1d','1wk','1mo'. Default: '1d'",
                },
            },
            "required": ["action", "symbol"],
        }

    async def execute(self, **kwargs: Any) -> str:
        action = kwargs.get("action", "ohlcv")
        symbol = kwargs.get("symbol", "")
        if not symbol:
            return json.dumps({"error": "symbol is required"})
        try:
            if action == "ohlcv":
                return await asyncio.to_thread(
                    self._fetch_ohlcv, symbol,
                    kwargs.get("period", "1y"), kwargs.get("interval", "1d"),
                )
            if action == "price":
                return await asyncio.to_thread(self._fetch_price, symbol)
            return json.dumps({"error": f"Unknown action '{action}'"})
        except ImportError:
            return json.dumps({"error": "yfinance not installed"})
        except Exception as e:
            logger.error(f"yfinance error for {symbol}: {e}")
            return json.dumps({"error": str(e), "symbol": symbol})

    # ── ohlcv ──

    @staticmethod
    def _fetch_ohlcv(symbol: str, period: str, interval: str) -> str:
        import yfinance as yf

        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)

        if df.empty:
            return json.dumps({"error": f"No data for {symbol}", "symbol": symbol})

        candles = []
        for ts, row in df.iterrows():
            candles.append({
                "timestamp": int(ts.timestamp() * 1000),
                "open": round(float(row["Open"]), 6),
                "high": round(float(row["High"]), 6),
                "low": round(float(row["Low"]), 6),
                "close": round(float(row["Close"]), 6),
                "volume": round(float(row.get("Volume", 0)), 2),
            })

        return json.dumps({
            "symbol": symbol,
            "period": period,
            "interval": interval,
            "candles": len(candles),
            "ohlcv": candles,
        })

    # ── price ──

    @staticmethod
    def _fetch_price(symbol: str) -> str:
        import yfinance as yf

        ticker = yf.Ticker(symbol)
        info = ticker.fast_info
        try:
            price = float(info.last_price)
        except Exception:
            return json.dumps({"error": f"Could not get price for {symbol}", "symbol": symbol})

        result: dict[str, Any] = {
            "symbol": symbol,
            "price": round(price, 6),
        }
        try:
            result["previous_close"] = round(float(info.previous_close), 6)
            result["change_pct"] = round((price / float(info.previous_close) - 1) * 100, 2)
        except Exception:
            pass
        try:
            result["market_cap"] = int(info.market_cap)
        except Exception:
            pass

        return json.dumps(result)
