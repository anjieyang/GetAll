"""AKShare tool — A股/港股/全球市场数据，免费无需 API key.

Covers Chinese A-shares, Hong Kong stocks, major indices, and more.
Data source: https://akshare.akfamily.xyz/

Symbol formats:
  A股: "000001" (平安银行), "600519" (贵州茅台)
  港股: "00700" (腾讯), "09988" (阿里巴巴)
  指数: "sh000001" (上证指数), "sz399001" (深证成指), "sz399006" (创业板指)
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta
from typing import Any

from loguru import logger

from getall.agent.tools.base import Tool


class AKShareTool(Tool):
    """AKShare — A股/港股/指数行情数据，免费无需 API key."""

    @property
    def name(self) -> str:
        return "akshare"

    @property
    def description(self) -> str:
        return (
            "Fetch Chinese A-share (A股), Hong Kong (港股), and index data via AKShare. "
            "No API key needed. Free and open-source. "
            "Actions: 'ohlcv' (历史K线), 'price' (实时行情), 'search' (搜索股票), "
            "'index' (指数行情). "
            "A-share symbols: '000001' (平安银行), '600519' (贵州茅台). "
            "HK symbols: '00700' (腾讯), '09988' (阿里巴巴). "
            "Index symbols: 'sh000001' (上证指数), 'sz399001' (深证成指), "
            "'sz399006' (创业板指), 'sh000300' (沪深300). "
            "This tool is specialized for Chinese/HK markets. "
            "For US stocks use yfinance instead."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": (
                        "'ohlcv': historical candles for A-share/HK stock. "
                        "'price': real-time quote for A-share/HK stock. "
                        "'search': search stocks by keyword or code. "
                        "'index': major index OHLCV data."
                    ),
                    "enum": ["ohlcv", "price", "search", "index"],
                },
                "symbol": {
                    "type": "string",
                    "description": (
                        "Stock/index code. "
                        "A-share: '000001', '600519'. "
                        "HK: '00700', '09988'. "
                        "Index: 'sh000001', 'sz399001', 'sh000300'. "
                        "For 'search' action: keyword like '茅台' or '银行'."
                    ),
                },
                "market": {
                    "type": "string",
                    "description": "Market: 'a' (A股, default), 'hk' (港股).",
                    "enum": ["a", "hk"],
                },
                "period": {
                    "type": "string",
                    "description": (
                        "K-line period: 'daily' (default), 'weekly', 'monthly'. "
                        "For intraday: '1', '5', '15', '30', '60' (minutes)."
                    ),
                },
                "days": {
                    "type": "integer",
                    "description": "Number of calendar days of history to fetch. Default: 365.",
                },
                "adjust": {
                    "type": "string",
                    "description": "Price adjust: '' (不复权, default), 'qfq' (前复权), 'hfq' (后复权).",
                    "enum": ["", "qfq", "hfq"],
                },
            },
            "required": ["action"],
        }

    async def execute(self, **kwargs: Any) -> str:
        action = kwargs.get("action", "ohlcv")
        symbol = kwargs.get("symbol", "")

        if action != "search" and not symbol:
            return json.dumps({"error": "symbol is required for this action"})

        try:
            if action == "ohlcv":
                market = kwargs.get("market", "a")
                period = kwargs.get("period", "daily")
                days = kwargs.get("days", 365)
                adjust = kwargs.get("adjust", "")
                return await asyncio.to_thread(
                    self._fetch_ohlcv, symbol, market, period, days, adjust,
                )
            if action == "price":
                market = kwargs.get("market", "a")
                return await asyncio.to_thread(self._fetch_price, symbol, market)
            if action == "search":
                return await asyncio.to_thread(self._search, symbol or "")
            if action == "index":
                days = kwargs.get("days", 365)
                return await asyncio.to_thread(self._fetch_index, symbol, days)
            return json.dumps({"error": f"Unknown action '{action}'"})
        except ImportError:
            return json.dumps({"error": "akshare not installed. Run: pip install akshare"})
        except Exception as e:
            logger.error(f"akshare error ({action}, {symbol}): {e}")
            return json.dumps({"error": str(e), "symbol": symbol, "action": action})

    # ── ohlcv: 历史K线 ──

    @staticmethod
    def _fetch_ohlcv(
        symbol: str,
        market: str,
        period: str,
        days: int,
        adjust: str,
    ) -> str:
        import akshare as ak

        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=max(1, days))).strftime("%Y%m%d")

        if market == "hk":
            df = ak.stock_hk_hist(
                symbol=symbol,
                period=period,
                start_date=start_date,
                end_date=end_date,
                adjust=adjust or "",
            )
        else:
            df = ak.stock_zh_a_hist(
                symbol=symbol,
                period=period,
                start_date=start_date,
                end_date=end_date,
                adjust=adjust or "",
            )

        if df is None or df.empty:
            return json.dumps({"error": f"No data for {symbol}", "symbol": symbol})

        # Normalize column names (akshare uses Chinese column names)
        col_map = _build_column_map(df.columns.tolist())

        candles: list[dict[str, Any]] = []
        for _, row in df.iterrows():
            candle: dict[str, Any] = {}
            if "date" in col_map:
                candle["date"] = str(row[col_map["date"]])
            if "open" in col_map:
                candle["open"] = round(float(row[col_map["open"]]), 4)
            if "high" in col_map:
                candle["high"] = round(float(row[col_map["high"]]), 4)
            if "low" in col_map:
                candle["low"] = round(float(row[col_map["low"]]), 4)
            if "close" in col_map:
                candle["close"] = round(float(row[col_map["close"]]), 4)
            if "volume" in col_map:
                candle["volume"] = int(row[col_map["volume"]])
            if "turnover" in col_map:
                candle["turnover"] = round(float(row[col_map["turnover"]]), 2)
            candles.append(candle)

        # Limit output to avoid context overflow
        if len(candles) > 500:
            candles = candles[-500:]

        return json.dumps({
            "symbol": symbol,
            "market": market,
            "period": period,
            "adjust": adjust or "不复权",
            "candles": len(candles),
            "ohlcv": candles,
        }, ensure_ascii=False)

    # ── price: 实时行情 ──

    @staticmethod
    def _fetch_price(symbol: str, market: str) -> str:
        import akshare as ak

        if market == "hk":
            df = ak.stock_hk_spot_em()
            if df is None or df.empty:
                return json.dumps({"error": "Failed to fetch HK market data"})
            # Filter by symbol
            row = df[df["代码"] == symbol]
            if row.empty:
                return json.dumps({"error": f"HK stock {symbol} not found", "symbol": symbol})
            row = row.iloc[0]
            return json.dumps({
                "symbol": symbol,
                "market": "hk",
                "name": str(row.get("名称", "")),
                "price": round(float(row.get("最新价", 0)), 4),
                "change_pct": round(float(row.get("涨跌幅", 0)), 2),
                "change_amount": round(float(row.get("涨跌额", 0)), 4),
                "volume": int(row.get("成交量", 0)),
                "turnover": round(float(row.get("成交额", 0)), 2),
                "high": round(float(row.get("最高", 0)), 4),
                "low": round(float(row.get("最低", 0)), 4),
                "open": round(float(row.get("今开", 0)), 4),
                "previous_close": round(float(row.get("昨收", 0)), 4),
            }, ensure_ascii=False)
        else:
            df = ak.stock_zh_a_spot_em()
            if df is None or df.empty:
                return json.dumps({"error": "Failed to fetch A-share market data"})
            row = df[df["代码"] == symbol]
            if row.empty:
                return json.dumps({"error": f"A-share {symbol} not found", "symbol": symbol})
            row = row.iloc[0]
            result: dict[str, Any] = {
                "symbol": symbol,
                "market": "a",
                "name": str(row.get("名称", "")),
                "price": _safe_float(row, "最新价"),
                "change_pct": _safe_float(row, "涨跌幅"),
                "change_amount": _safe_float(row, "涨跌额"),
                "volume": _safe_int(row, "成交量"),
                "turnover": _safe_float(row, "成交额"),
                "high": _safe_float(row, "最高"),
                "low": _safe_float(row, "最低"),
                "open": _safe_float(row, "今开"),
                "previous_close": _safe_float(row, "昨收"),
                "pe_ratio": _safe_float(row, "市盈率-动态"),
                "pb_ratio": _safe_float(row, "市净率"),
                "total_market_cap": _safe_float(row, "总市值"),
                "circulating_market_cap": _safe_float(row, "流通市值"),
                "turnover_rate": _safe_float(row, "换手率"),
            }
            return json.dumps(result, ensure_ascii=False)

    # ── search: 搜索股票 ──

    @staticmethod
    def _search(keyword: str) -> str:
        import akshare as ak

        if not keyword:
            return json.dumps({"error": "keyword is required for search"})

        df = ak.stock_zh_a_spot_em()
        if df is None or df.empty:
            return json.dumps({"error": "Failed to fetch stock list"})

        # Filter by code or name
        mask = df["代码"].str.contains(keyword, na=False) | df["名称"].str.contains(
            keyword, na=False
        )
        matches = df[mask]

        if matches.empty:
            return json.dumps({"keyword": keyword, "results": [], "total": 0}, ensure_ascii=False)

        results: list[dict[str, Any]] = []
        for _, row in matches.head(20).iterrows():
            results.append({
                "symbol": str(row.get("代码", "")),
                "name": str(row.get("名称", "")),
                "price": _safe_float(row, "最新价"),
                "change_pct": _safe_float(row, "涨跌幅"),
                "total_market_cap": _safe_float(row, "总市值"),
            })

        return json.dumps({
            "keyword": keyword,
            "results": results,
            "total": len(matches),
            "showing": len(results),
        }, ensure_ascii=False)

    # ── index: 指数行情 ──

    @staticmethod
    def _fetch_index(symbol: str, days: int) -> str:
        import akshare as ak

        end_date = datetime.now().strftime("%Y%m%d")
        start_date = (datetime.now() - timedelta(days=max(1, days))).strftime("%Y%m%d")

        df = ak.stock_zh_index_daily_em(symbol=symbol, start_date=start_date, end_date=end_date)

        if df is None or df.empty:
            return json.dumps({"error": f"No index data for {symbol}", "symbol": symbol})

        col_map = _build_column_map(df.columns.tolist())

        candles: list[dict[str, Any]] = []
        for _, row in df.iterrows():
            candle: dict[str, Any] = {}
            if "date" in col_map:
                candle["date"] = str(row[col_map["date"]])
            if "open" in col_map:
                candle["open"] = round(float(row[col_map["open"]]), 2)
            if "high" in col_map:
                candle["high"] = round(float(row[col_map["high"]]), 2)
            if "low" in col_map:
                candle["low"] = round(float(row[col_map["low"]]), 2)
            if "close" in col_map:
                candle["close"] = round(float(row[col_map["close"]]), 2)
            if "volume" in col_map:
                candle["volume"] = int(row[col_map["volume"]])
            candles.append(candle)

        if len(candles) > 500:
            candles = candles[-500:]

        return json.dumps({
            "symbol": symbol,
            "type": "index",
            "candles": len(candles),
            "ohlcv": candles,
        }, ensure_ascii=False)


# ── Helpers ──

# Mapping from Chinese column names to standardized keys.
# AKShare DataFrames use Chinese headers; we normalize them.
_CN_COL_ALIASES: dict[str, list[str]] = {
    "date": ["日期", "date", "Date"],
    "open": ["开盘", "开盘价", "open", "Open"],
    "high": ["最高", "最高价", "high", "High"],
    "low": ["最低", "最低价", "low", "Low"],
    "close": ["收盘", "收盘价", "close", "Close"],
    "volume": ["成交量", "volume", "Volume"],
    "turnover": ["成交额", "turnover", "Turnover", "Amount"],
}


def _build_column_map(columns: list[str]) -> dict[str, str]:
    """Map standardized keys to actual DataFrame column names."""
    result: dict[str, str] = {}
    for key, aliases in _CN_COL_ALIASES.items():
        for alias in aliases:
            if alias in columns:
                result[key] = alias
                break
    return result


def _safe_float(row: Any, col: str, decimals: int = 4) -> float:
    """Safely extract a float value from a DataFrame row."""
    try:
        val = row.get(col, 0)
        if val is None or str(val) in ("", "-", "nan", "None"):
            return 0.0
        return round(float(val), decimals)
    except (ValueError, TypeError):
        return 0.0


def _safe_int(row: Any, col: str) -> int:
    """Safely extract an int value from a DataFrame row."""
    try:
        val = row.get(col, 0)
        if val is None or str(val) in ("", "-", "nan", "None"):
            return 0
        return int(float(val))
    except (ValueError, TypeError):
        return 0
