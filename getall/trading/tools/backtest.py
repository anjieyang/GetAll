"""Backtest tool — atomic capability for the agent.

Design (per Anthropic skill guide):
  Tool = raw capability → returns structured JSON facts.
  Skill = domain intelligence → interprets, formats, iterates.

Actions:
  - run:    Run a VectorBT backtest. Returns JSON metrics dict.
  - chart:  Generate equity curve chart from previous metrics.
"""

import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from getall.agent.tools.base import Tool
from getall.trading.backtest.engine import (
    ccxt_to_dataframe,
    estimate_candle_count,
    generate_chart,
    run_backtest,
    timeframe_to_seconds,
)
from getall.trading.data.exchange import ExchangeAdapter
from getall.trading.data.hub import DataHub


class BacktestTool(Tool):
    """VectorBT backtest tool — returns structured data for agent interpretation."""

    def __init__(self, hub: DataHub, workspace: Path):
        self.hub = hub
        self.workspace = workspace
        self._public_exchanges: dict[str, ExchangeAdapter] = {}
        self._report_root = workspace / "reports" / "backtest"

    @property
    def name(self) -> str:
        return "backtest"

    @property
    def description(self) -> str:
        return (
            "Run historical backtests on trading strategies. This tool handles EVERYTHING: "
            "data fetching from exchanges, indicator computation, signal evaluation, "
            "portfolio simulation, metrics calculation, AND professional dashboard chart generation. "
            "ALWAYS use this tool for ANY backtest — NEVER write backtest code via exec. "
            "Returns structured JSON metrics + chart_path. Read backtest-runner skill for report format."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "Action to perform",
                    "enum": ["run", "chart"],
                },
                "strategy_config": {
                    "type": "string",
                    "description": (
                        "JSON string of strategy config for 'run'. Keys: "
                        "name, symbols (list), timeframe, indicators (list of "
                        '{name, params, key}), entry_conditions (list of '
                        "{indicator, field, operator, value}), exit_conditions, "
                        "direction (long/short/both), stop_loss_pct, take_profit_pct, "
                        "trade_size_pct, fees."
                    ),
                },
                "period": {
                    "type": "string",
                    "description": "Lookback period e.g. '6m', '1y', '30d'. Default: '6m'",
                },
                "exchange": {
                    "type": "string",
                    "description": (
                        "Exchange for OHLCV data. Options: binance (widest coverage, default), "
                        "bitget, okx, bybit, coinbase, kraken, kucoin. "
                        "If result contains failed_symbols, retry those on another exchange."
                    ),
                    "enum": ["binance", "bitget", "okx", "bybit", "coinbase", "kraken", "kucoin"],
                },
                "starting_balance": {
                    "type": "number",
                    "description": "Initial cash. Default: 100000",
                },
                "ohlcv_json": {
                    "type": "string",
                    "description": (
                        "For action='run': pre-fetched OHLCV data as JSON string. "
                        "Use when data comes from coingecko or yfinance_ohlcv tools. "
                        "Format: {\"SYMBOL\": [{\"timestamp\":ms,\"open\":...,\"high\":...,\"low\":...,\"close\":...,\"volume\":...}, ...]} "
                        "When provided, skips exchange data fetching entirely."
                    ),
                },
                "metrics_json": {
                    "type": "string",
                    "description": (
                        "For action='chart': the JSON metrics string returned "
                        "by a previous 'run' call."
                    ),
                },
            },
            "required": ["action"],
        }

    async def execute(self, **kwargs: Any) -> str:
        action = kwargs["action"]
        try:
            if action == "run":
                return await self._run(**kwargs)
            if action == "chart":
                return await self._chart(**kwargs)
            return f"Error: unknown action '{action}'"
        except Exception as e:
            logger.error(f"Backtest tool error ({action}): {e}")
            return json.dumps({"error": str(e)})

    # ── run ──────────────────────────────────────────────

    async def _run(self, **kwargs: Any) -> str:
        config_str = kwargs.get("strategy_config")
        if not config_str:
            return json.dumps({"error": "strategy_config is required"})

        try:
            config = json.loads(config_str)
        except json.JSONDecodeError as e:
            return json.dumps({"error": f"Invalid JSON: {e}"})

        symbols = config.get("symbols", [])
        if not symbols:
            return json.dumps({"error": "symbols list is required in strategy_config"})

        period = kwargs.get("period", "6m")
        starting_balance = kwargs.get("starting_balance", 100_000)
        exchange_name = kwargs.get("exchange")

        # ── Option A: Agent supplies pre-fetched OHLCV (from coingecko/yfinance) ──
        ohlcv_json_str = kwargs.get("ohlcv_json")
        if ohlcv_json_str:
            ohlcv_data, parse_err = self._parse_external_ohlcv(ohlcv_json_str)
            if parse_err:
                return json.dumps({"error": parse_err})
            failed_symbols: list[str] = []
        else:
            # ── Option B: Fetch from exchange via ccxt ──
            ex = await self._resolve_exchange(exchange_name)
            if ex is None:
                return json.dumps({
                    "error": "No exchange available. Configure exchanges.yaml or pass exchange='binance'."
                })

            ohlcv_data: dict[str, pd.DataFrame] = {}
            failed_symbols: list[str] = []
            timeframe = config.get("timeframe", "4h")
            for symbol in symbols:
                normalized = self._normalize_symbol(symbol)
                df, err = await self._load_ohlcv(ex, normalized, timeframe, period)
                if err or df.empty:
                    logger.warning(f"Skipping {symbol} ({normalized}): {err or 'empty data'}")
                    failed_symbols.append(symbol)
                    continue
                ohlcv_data[normalized] = df

        if not ohlcv_data:
            return json.dumps({
                "error": f"Could not fetch data for any symbol. Failed: {failed_symbols}"
            })

        # Run backtest (sync engine, offload to thread)
        import asyncio
        metrics = await asyncio.to_thread(
            run_backtest, ohlcv_data, config, starting_balance
        )

        if failed_symbols:
            metrics["failed_symbols"] = failed_symbols
            metrics["loaded_symbols"] = list(ohlcv_data.keys())

        # Generate chart by default (uses internal _equity_curve / _benchmark_curve)
        chart_path = await generate_chart(
            metrics,
            save_dir=self._chart_dir(),
        )
        if chart_path:
            metrics["chart_path"] = chart_path
            metrics["chart_path_marker"] = f"[GENERATED_IMAGE:{chart_path}]"

        # Strip internal/heavy keys before returning to agent (keep JSON small)
        for key in ("_equity_curve", "_benchmark_curve", "_equity_symbol",
                     "equity_curve", "benchmark_curve", "config"):
            metrics.pop(key, None)
        # Also strip per_symbol equity curves if present
        for item in metrics.get("per_symbol", []):
            if isinstance(item, dict):
                for k in ("equity_curve", "benchmark_curve", "monthly_pnl",
                           "top_trades", "bottom_trades"):
                    item.pop(k, None)

        return json.dumps(metrics, default=str)

    # ── chart ────────────────────────────────────────────

    async def _chart(self, **kwargs: Any) -> str:
        metrics_str = kwargs.get("metrics_json")
        if not metrics_str:
            return json.dumps({"error": "metrics_json is required for chart action"})
        try:
            metrics = json.loads(metrics_str)
        except json.JSONDecodeError as e:
            return json.dumps({"error": f"Invalid metrics JSON: {e}"})

        chart_path = await generate_chart(metrics, save_dir=self._chart_dir())
        if chart_path:
            return json.dumps({"chart_path": chart_path})
        return json.dumps({"error": "No equity curve data to chart"})

    # ── helpers ──────────────────────────────────────────

    def _chart_dir(self) -> Path:
        now = datetime.now(timezone.utc)
        d = self._report_root / now.strftime("%Y%m%d")
        d.mkdir(parents=True, exist_ok=True)
        return d

    async def _resolve_exchange(self, exchange_name: str | None) -> ExchangeAdapter | None:
        preferred = (exchange_name or "").strip().lower()
        if preferred:
            ex = await self.hub.get_exchange(preferred)
            if ex is not None:
                return ex
            return self._public_exchange(preferred)
        default = self.hub.exchange
        if default is not None:
            return default
        return self._public_exchange(self._default_exchange_name())

    def _public_exchange(self, name: str) -> ExchangeAdapter | None:
        name = (name or "").strip().lower()
        if not name:
            return None
        if name in self._public_exchanges:
            return self._public_exchanges[name]
        try:
            adapter = ExchangeAdapter(exchange_name=name)
            self._public_exchanges[name] = adapter
            return adapter
        except Exception as e:
            logger.warning(f"Failed to create public exchange '{name}': {e}")
            return None

    _VALID_EXCHANGES = {"binance", "bitget", "okx", "bybit", "coinbase", "kraken", "kucoin"}

    def _default_exchange_name(self) -> str:
        cfg = getattr(self.hub, "_config", None)
        name = str(getattr(cfg, "default_exchange", "") or "").strip().lower()
        return name if name in self._VALID_EXCHANGES else "binance"

    @staticmethod
    def _parse_external_ohlcv(raw: str) -> tuple[dict[str, pd.DataFrame], str | None]:
        """Parse agent-supplied OHLCV JSON into {symbol: DataFrame}.

        Accepts two formats:
        1. Direct from coingecko/yfinance tool: {"ohlcv": [{...}, ...], "symbol": "X"}
        2. Multi-symbol dict: {"SYM1": [{...}], "SYM2": [{...}]}
        """
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            return {}, f"Invalid ohlcv_json: {e}"

        result: dict[str, pd.DataFrame] = {}

        # Format 1: single-symbol output from coingecko/yfinance tools
        if "ohlcv" in data and isinstance(data["ohlcv"], list):
            symbol = data.get("symbol") or data.get("coin_id") or "EXTERNAL"
            rows = data["ohlcv"]
            df = ccxt_to_dataframe(rows, symbol)
            if not df.empty:
                result[symbol] = df
            return result, None if result else f"ohlcv array was empty for {symbol}"

        # Format 2: multi-symbol dict {"BTC/USDT": [...], ...}
        for symbol, rows in data.items():
            if isinstance(rows, list) and rows:
                df = ccxt_to_dataframe(rows, symbol)
                if not df.empty:
                    result[symbol] = df

        if not result:
            return {}, "No valid OHLCV data found in ohlcv_json"
        return result, None

    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        """Normalize symbol to ccxt unified format.

        Handles common agent inputs:
          BTCUSDT        → BTC/USDT:USDT   (futures)
          BTC/USDT       → BTC/USDT:USDT   (add settlement suffix)
          BTC/USDT:USDT  → BTC/USDT:USDT   (already correct)
          1MCHEEMSUSDT   → 1MCHEEMS/USDT:USDT
        """
        s = symbol.strip()
        # Already has settlement suffix — done
        if ":" in s:
            return s
        # Already has slash — add :USDT suffix for futures
        if "/" in s:
            quote = s.split("/")[1].split(":")[0]
            return f"{s}:{quote}"
        # Raw concatenated: BTCUSDT, AGIUSDT, 1MCHEEMSUSDT
        upper = s.upper()
        for quote in ("USDT", "USDC", "USD"):
            if upper.endswith(quote) and len(upper) > len(quote):
                base = upper[:-len(quote)]
                return f"{base}/{quote}:{quote}"
        # Fallback: treat as base, assume USDT
        return f"{upper}/USDT:USDT"

    async def _load_ohlcv(
        self,
        ex: ExchangeAdapter,
        symbol: str,
        timeframe: str,
        period: str,
    ) -> tuple[pd.DataFrame, str | None]:
        """Load OHLCV with pagination."""
        target = max(estimate_candle_count(period, timeframe), 1)
        tf_sec = timeframe_to_seconds(timeframe)
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        since = now_ms - target * tf_sec * 1000 - tf_sec * 1000

        rows: list[dict[str, Any]] = []
        last_ts = -1
        max_batches = max((target // 1000) + 3, 3)

        for _ in range(max_batches):
            remaining = max(target - len(rows), 0)
            limit = min(1000, max(remaining, 200))
            klines = await ex.get_klines(symbol, timeframe, limit=limit, since=since)
            if isinstance(klines, dict) and "error" in klines:
                return pd.DataFrame(), f"Error fetching {symbol}: {klines['error']}"
            if not klines:
                break
            fresh = sorted(
                (k for k in klines if int(k.get("timestamp", 0)) > last_ts),
                key=lambda x: int(x.get("timestamp", 0)),
            )
            if not fresh:
                break
            rows.extend(fresh)
            last_ts = int(fresh[-1]["timestamp"])
            since = last_ts + 1
            if len(rows) >= target and len(fresh) < limit:
                break

        if not rows:
            return pd.DataFrame(), f"No kline data for {symbol}"
        if len(rows) > target:
            rows = rows[-target:]
        return ccxt_to_dataframe(rows, symbol), None
