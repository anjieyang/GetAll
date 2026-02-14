"""VectorBT-native backtest engine.

Atomic capability: takes OHLCV DataFrame + strategy config dict →
runs vectorbt.Portfolio.from_signals() → returns structured metrics dict.

Design principles (per Anthropic skill guide):
  - Tool = raw capability, returns structured facts (JSON-serialisable dict)
  - No report formatting, no interpretation — that's the agent's job via skills
  - Thin, composable, easy to test
"""

from __future__ import annotations

import asyncio
import math
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pandas_ta as ta
import vectorbt as vbt
from loguru import logger


# ═══════════════════════════════════════════════════════
# Indicator computation
# ═══════════════════════════════════════════════════════

_INDICATOR_REGISTRY: dict[str, dict[str, Any]] = {
    # Moving averages
    "sma": {"fn": lambda df, p: ta.sma(df["close"], length=p.get("period", 20)), "defaults": {"period": 20}},
    "ema": {"fn": lambda df, p: ta.ema(df["close"], length=p.get("period", 20)), "defaults": {"period": 20}},
    "dema": {"fn": lambda df, p: ta.dema(df["close"], length=p.get("period", 20)), "defaults": {"period": 20}},
    "hma": {"fn": lambda df, p: ta.hma(df["close"], length=p.get("period", 20)), "defaults": {"period": 20}},
    "wma": {"fn": lambda df, p: ta.wma(df["close"], length=p.get("period", 20)), "defaults": {"period": 20}},
    # Momentum
    "rsi": {"fn": lambda df, p: ta.rsi(df["close"], length=p.get("period", 14)), "defaults": {"period": 14}},
    "roc": {"fn": lambda df, p: ta.roc(df["close"], length=p.get("period", 10)), "defaults": {"period": 10}},
    "cci": {"fn": lambda df, p: ta.cci(df["high"], df["low"], df["close"], length=p.get("period", 20)), "defaults": {"period": 20}},
    # MACD — multi-output
    "macd": {
        "fn": lambda df, p: ta.macd(
            df["close"],
            fast=p.get("fast_period", 12),
            slow=p.get("slow_period", 26),
            signal=p.get("signal_period", 9),
        ),
        "defaults": {"fast_period": 12, "slow_period": 26, "signal_period": 9},
        "multi": True,  # returns DataFrame with columns [MACD_*, MACDs_*, MACDh_*]
    },
    # Bollinger Bands — multi-output
    "bollinger": {
        "fn": lambda df, p: ta.bbands(df["close"], length=p.get("period", 20), std=p.get("std", 2.0)),
        "defaults": {"period": 20, "std": 2.0},
        "multi": True,
    },
    # Stochastics — multi-output
    "stoch": {
        "fn": lambda df, p: ta.stoch(df["high"], df["low"], df["close"], k=p.get("period_k", 14), d=p.get("period_d", 3)),
        "defaults": {"period_k": 14, "period_d": 3},
        "multi": True,
    },
    # Volatility
    "atr": {"fn": lambda df, p: ta.atr(df["high"], df["low"], df["close"], length=p.get("period", 14)), "defaults": {"period": 14}},
}

# Canonical field aliases for multi-output indicators
_MULTI_FIELD_MAP: dict[str, dict[str, int]] = {
    "macd": {"value": 0, "signal": 1, "histogram": 2},
    "bollinger": {"lower": 0, "mid": 1, "upper": 2, "bandwidth": 3, "percent": 4},
    "stoch": {"k": 0, "d": 1},
}


def compute_indicators(
    df: pd.DataFrame,
    indicator_configs: list[dict[str, Any]],
) -> dict[str, pd.Series]:
    """Compute all requested indicators and return a flat {key: Series} dict."""
    result: dict[str, pd.Series] = {}
    for cfg in indicator_configs:
        name = cfg["name"].lower()
        params = {**_INDICATOR_REGISTRY.get(name, {}).get("defaults", {}), **cfg.get("params", {})}
        key = cfg.get("key", name)
        reg = _INDICATOR_REGISTRY.get(name)
        if reg is None:
            logger.warning(f"Unknown indicator '{name}', skipping")
            continue
        try:
            out = reg["fn"](df, params)
        except Exception as e:
            logger.warning(f"Error computing indicator {name}: {e}")
            continue
        if reg.get("multi") and isinstance(out, pd.DataFrame) and out is not None:
            field_map = _MULTI_FIELD_MAP.get(name, {})
            for field_name, col_idx in field_map.items():
                if col_idx < len(out.columns):
                    result[f"{key}_{field_name}"] = out.iloc[:, col_idx]
            # Also store first column as plain key for simple references
            result[key] = out.iloc[:, 0]
        elif isinstance(out, pd.Series):
            result[key] = out
    # Always include OHLCV as referenceable series
    for col in ("open", "high", "low", "close", "volume"):
        if col in df.columns:
            result[col] = df[col]
    return result


# ═══════════════════════════════════════════════════════
# Condition evaluation → boolean signal arrays
# ═══════════════════════════════════════════════════════

def _resolve_value(
    raw: float | int | str,
    indicators: dict[str, pd.Series],
) -> pd.Series | float:
    """Resolve a condition threshold — either a literal number or an indicator reference."""
    if isinstance(raw, (int, float)):
        return float(raw)
    s = str(raw)
    # "indicator.field" e.g. "macd.signal" → key "macd_signal"
    if "." in s:
        ref_key, ref_field = s.split(".", 1)
        lookup = f"{ref_key}_{ref_field}"
        if lookup in indicators:
            return indicators[lookup]
    # Direct key
    if s in indicators:
        return indicators[s]
    # Try as number
    try:
        return float(s)
    except ValueError:
        pass
    return float("nan")


def _eval_op(current: pd.Series, op: str, threshold: pd.Series | float) -> pd.Series:
    """Apply a comparison operator and return a boolean Series."""
    if op == "lt":
        return current < threshold
    if op == "gt":
        return current > threshold
    if op == "lte":
        return current <= threshold
    if op == "gte":
        return current >= threshold
    if op == "eq":
        if isinstance(threshold, pd.Series):
            return (current - threshold).abs() < 1e-8
        return (current - threshold).abs() < 1e-8
    if op == "cross_above":
        prev = current.shift(1)
        if isinstance(threshold, pd.Series):
            return (prev <= threshold.shift(1)) & (current > threshold)
        return (prev <= threshold) & (current > threshold)
    if op == "cross_below":
        prev = current.shift(1)
        if isinstance(threshold, pd.Series):
            return (prev >= threshold.shift(1)) & (current < threshold)
        return (prev >= threshold) & (current > threshold)
    logger.warning(f"Unknown operator '{op}', returning False")
    return pd.Series(False, index=current.index)


def evaluate_conditions(
    conditions: list[dict[str, Any]],
    indicators: dict[str, pd.Series],
    logic: str = "and",
) -> pd.Series:
    """Evaluate a list of conditions and combine with AND or OR logic.

    Each condition: {"indicator": str, "field": str, "operator": str, "value": ...}
    """
    if not conditions:
        return pd.Series(False, index=next(iter(indicators.values())).index)

    signals: list[pd.Series] = []
    for cond in conditions:
        ind_key = cond["indicator"]
        field = cond.get("field", "value")
        # Resolve current value series
        lookup = ind_key if field == "value" else f"{ind_key}_{field}"
        current = indicators.get(lookup)
        if current is None and field == "value":
            current = indicators.get(f"{ind_key}_value")
        if current is None:
            current = indicators.get(ind_key)
        if current is None:
            logger.warning(f"Indicator '{lookup}' not found, condition skipped")
            continue

        threshold = _resolve_value(cond["value"], indicators)
        sig = _eval_op(current, cond["operator"], threshold)
        signals.append(sig)

    if not signals:
        return pd.Series(False, index=next(iter(indicators.values())).index)

    combined = signals[0]
    for s in signals[1:]:
        combined = (combined & s) if logic == "and" else (combined | s)
    return combined.fillna(False)


# ═══════════════════════════════════════════════════════
# Core backtest runner
# ═══════════════════════════════════════════════════════

def run_backtest(
    ohlcv_data: dict[str, pd.DataFrame],
    config: dict[str, Any],
    starting_balance: float = 100_000.0,
) -> dict[str, Any]:
    """Run a VectorBT backtest and return structured metrics.

    Args:
        ohlcv_data: {symbol: DataFrame} with OHLCV columns + DatetimeIndex.
        config: Strategy configuration dict with keys:
            name, symbols, timeframe, indicators, entry_conditions,
            exit_conditions, direction, stop_loss_pct, take_profit_pct,
            trade_size_pct, leverage.
        starting_balance: Initial cash.

    Returns:
        JSON-serialisable dict with metrics, equity curve, and trade list.
    """
    direction = config.get("direction", "long")
    sl_pct = config.get("stop_loss_pct")
    tp_pct = config.get("take_profit_pct")
    size_pct = config.get("trade_size_pct", 100.0) / 100.0
    fees = config.get("fees", 0.0006)

    all_results: list[dict[str, Any]] = []

    for symbol, df in ohlcv_data.items():
        if df.empty:
            continue

        # 1. Compute indicators
        indicators = compute_indicators(df, config.get("indicators", []))

        # 2. Evaluate entry / exit signals
        entry_signals = evaluate_conditions(config.get("entry_conditions", []), indicators, logic="and")
        exit_signals = evaluate_conditions(config.get("exit_conditions", []), indicators, logic="or")

        # 3. Build VectorBT portfolio
        close = df["close"]
        pf_kwargs: dict[str, Any] = {
            "close": close,
            "init_cash": starting_balance,
            "size": size_pct,
            "size_type": "percent",
            "fees": fees,
        }
        if sl_pct is not None and sl_pct > 0:
            pf_kwargs["sl_stop"] = sl_pct / 100.0
        if tp_pct is not None and tp_pct > 0:
            pf_kwargs["tp_stop"] = tp_pct / 100.0

        if direction == "short":
            pf_kwargs["short_entries"] = entry_signals
            pf_kwargs["short_exits"] = exit_signals
        elif direction == "both":
            # For "both": use entry_conditions with direction field filtering
            long_entry = _filter_directional(config.get("entry_conditions", []), "long", indicators)
            long_exit = _filter_directional(config.get("exit_conditions", []), "long", indicators, logic="or")
            short_entry = _filter_directional(config.get("entry_conditions", []), "short", indicators)
            short_exit = _filter_directional(config.get("exit_conditions", []), "short", indicators, logic="or")
            pf_kwargs["entries"] = long_entry
            pf_kwargs["exits"] = long_exit
            pf_kwargs["short_entries"] = short_entry
            pf_kwargs["short_exits"] = short_exit
        else:
            pf_kwargs["entries"] = entry_signals
            pf_kwargs["exits"] = exit_signals

        try:
            pf = vbt.Portfolio.from_signals(**pf_kwargs)
        except Exception as e:
            logger.error(f"VectorBT portfolio creation failed for {symbol}: {e}")
            all_results.append({"symbol": symbol, "error": str(e)})
            continue

        # 4. Extract structured metrics
        metrics = _extract_metrics(pf, symbol, df, starting_balance)
        all_results.append(metrics)

    # Merge multi-symbol or return single
    if len(all_results) == 1:
        result = all_results[0]
    else:
        result = _merge_results(all_results)

    result["strategy_name"] = config.get("name", "unnamed")
    result["timeframe"] = config.get("timeframe", "")
    result["direction"] = direction
    result["config"] = config
    return result


def _filter_directional(
    conditions: list[dict[str, Any]],
    target_dir: str,
    indicators: dict[str, pd.Series],
    logic: str = "and",
) -> pd.Series:
    """Filter conditions by direction tag and evaluate."""
    filtered = []
    for c in conditions:
        d = str(c.get("direction", "")).lower()
        if d and d != target_dir:
            continue
        filtered.append(c)
    return evaluate_conditions(filtered, indicators, logic=logic)


def _extract_metrics(
    pf: Any,
    symbol: str,
    df: pd.DataFrame,
    starting_balance: float,
) -> dict[str, Any]:
    """Extract comprehensive structured metrics from a VectorBT Portfolio."""
    trades = pf.trades
    equity = pf.value()
    total_trades = int(trades.count())

    # Period info
    period_start = str(df.index[0])[:10] if len(df) > 0 else ""
    period_end = str(df.index[-1])[:10] if len(df) > 0 else ""
    total_days = max((df.index[-1] - df.index[0]).days, 1) if len(df) > 1 else 0
    years = max(total_days / 365.0, 1 / 365.0)
    ending_bal = round(float(equity.iloc[-1]), 2) if len(equity) > 0 else starting_balance
    total_return = float(pf.total_return())

    # Annualized return
    ann_return = 0.0
    if starting_balance > 0 and ending_bal > 0 and years > 0:
        ann_return = ((ending_bal / starting_balance) ** (1 / years) - 1) * 100

    # Max drawdown duration
    dd_duration_days = 0.0
    if len(equity) > 1:
        peaks = equity.cummax()
        in_dd = equity < peaks
        groups = (~in_dd).cumsum()
        if in_dd.any():
            dd_groups = in_dd.groupby(groups)
            for _, grp in dd_groups:
                if grp.any():
                    dd_len = (grp.index[-1] - grp.index[0]).total_seconds() / 86400
                    dd_duration_days = max(dd_duration_days, dd_len)

    metrics: dict[str, Any] = {
        "symbol": symbol,
        "period_start": period_start,
        "period_end": period_end,
        "total_days": total_days,
        # ── Portfolio performance ──
        "starting_balance": starting_balance,
        "ending_balance": ending_bal,
        "net_profit": round(ending_bal - starting_balance, 2),
        "total_return_pct": round(total_return * 100, 4),
        "annualized_return_pct": round(ann_return, 2),
        # ── Risk metrics ──
        "max_drawdown_pct": round(abs(float(pf.drawdowns.max_drawdown())) * 100, 2),
        "max_drawdown_duration_days": round(dd_duration_days, 1),
        "sharpe_ratio": _safe_float(pf.sharpe_ratio()),
        "sortino_ratio": _safe_float(pf.sortino_ratio()),
        "calmar_ratio": round(ann_return / max(abs(float(pf.drawdowns.max_drawdown())) * 100, 0.01), 2),
        # ── Trade quality ──
        "total_trades": total_trades,
        "win_rate_pct": round(float(trades.win_rate()) * 100, 2) if total_trades > 0 else 0.0,
        "profit_factor": _safe_float(trades.profit_factor()) if total_trades > 0 else 0.0,
        "expectancy": round(float(trades.expectancy()), 4) if total_trades > 0 else 0.0,
    }

    # Trade-level detail
    if total_trades > 0:
        records = trades.records_readable
        pnl_col = records["PnL"] if "PnL" in records.columns else pd.Series(dtype=float)
        win_count = int(trades.winning.count())
        loss_count = int(trades.losing.count())
        avg_win = round(float(trades.winning.pnl.mean()), 2) if win_count > 0 else 0.0
        avg_loss = round(float(trades.losing.pnl.mean()), 2) if loss_count > 0 else 0.0
        payoff_ratio = round(abs(avg_win / avg_loss), 2) if avg_loss != 0 else 0.0

        metrics["avg_win"] = avg_win
        metrics["avg_loss"] = avg_loss
        metrics["payoff_ratio"] = payoff_ratio
        metrics["best_trade"] = round(float(pnl_col.max()), 2) if len(pnl_col) > 0 else 0.0
        metrics["worst_trade"] = round(float(pnl_col.min()), 2) if len(pnl_col) > 0 else 0.0

        # Avg hold time
        if "Entry Timestamp" in records.columns and "Exit Timestamp" in records.columns:
            closed = records[records["Status"] == "Closed"] if "Status" in records.columns else records
            if len(closed) > 0:
                durations = pd.to_datetime(closed["Exit Timestamp"]) - pd.to_datetime(closed["Entry Timestamp"])
                avg_hold_sec = durations.dt.total_seconds().mean()
                if avg_hold_sec >= 86400:
                    metrics["avg_hold_time"] = f"{avg_hold_sec / 86400:.1f}d"
                else:
                    metrics["avg_hold_time"] = f"{avg_hold_sec / 3600:.1f}h"

        # Max consecutive losses
        if len(pnl_col) > 0:
            is_loss = (pnl_col < 0).astype(int)
            streaks = is_loss.groupby((is_loss != is_loss.shift()).cumsum())
            max_consec = max((g.sum() for _, g in streaks), default=0)
            metrics["max_consecutive_losses"] = int(max_consec)

        # Top 5 / Bottom 5 trades
        if len(pnl_col) >= 3:
            sorted_pnl = pnl_col.sort_values()
            n = min(5, len(sorted_pnl))
            metrics["top_trades"] = [round(float(v), 2) for v in sorted_pnl.iloc[-n:][::-1]]
            metrics["bottom_trades"] = [round(float(v), 2) for v in sorted_pnl.iloc[:n]]

        # Monthly P&L attribution
        if "Exit Timestamp" in records.columns:
            closed = records[records["Status"] == "Closed"] if "Status" in records.columns else records
            if len(closed) > 0:
                monthly_df = closed.copy()
                monthly_df["month"] = pd.to_datetime(monthly_df["Exit Timestamp"]).dt.to_period("M")
                monthly_pnl = monthly_df.groupby("month")["PnL"].sum()
                metrics["monthly_pnl"] = {
                    str(k): round(float(v), 2) for k, v in monthly_pnl.items()
                }

    # Equity curve (sampled for transport — max 500 points)
    if len(equity) > 0:
        step = max(1, len(equity) // 500)
        sampled = equity.iloc[::step]
        metrics["equity_curve"] = [
            {"t": str(ts)[:19], "v": round(float(v), 2)}
            for ts, v in zip(sampled.index, sampled.values)
        ]

    # Benchmark (buy & hold)
    if len(df) > 1:
        bh_return = (float(df["close"].iloc[-1]) / float(df["close"].iloc[0]) - 1) * 100
        metrics["benchmark_return_pct"] = round(bh_return, 4)
        metrics["excess_return_pct"] = round(metrics["total_return_pct"] - bh_return, 4)
        # Benchmark equity curve for chart overlay
        base_price = float(df["close"].iloc[0])
        bh_equity = starting_balance * (df["close"] / base_price)
        step = max(1, len(bh_equity) // 500)
        sampled_bh = bh_equity.iloc[::step]
        metrics["benchmark_curve"] = [
            {"t": str(ts)[:19], "v": round(float(v), 2)}
            for ts, v in zip(sampled_bh.index, sampled_bh.values)
        ]

    return metrics


def _merge_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge per-symbol results into a combined summary.

    IMPORTANT: per_symbol only contains compact summaries (no equity_curve,
    no benchmark_curve, no monthly_pnl) to keep the JSON small enough for
    LLM context windows.
    """
    valid = [r for r in results if "error" not in r]
    failed = [r for r in results if "error" in r]
    if not valid:
        return {"error": "All symbols failed", "details": results}

    n = len(valid)
    _avg = lambda key: round(sum(r.get(key, 0) for r in valid) / n, 4)
    _max = lambda key: round(max(r.get(key, 0) for r in valid), 4)
    _sum = lambda key: sum(r.get(key, 0) for r in valid)

    # Compact per-symbol table (only key metrics, no large arrays)
    _COMPACT_KEYS = [
        "symbol", "total_return_pct", "win_rate_pct", "profit_factor",
        "max_drawdown_pct", "total_trades", "sharpe_ratio",
    ]
    compact_per_symbol = [
        {k: r.get(k) for k in _COMPACT_KEYS if r.get(k) is not None}
        for r in valid
    ]
    # Sort by return descending for easy reading
    compact_per_symbol.sort(key=lambda x: x.get("total_return_pct", 0), reverse=True)

    merged: dict[str, Any] = {
        "symbols_count": n,
        "per_symbol": compact_per_symbol,
        # Portfolio (aggregate)
        "starting_balance": valid[0].get("starting_balance", 100_000),
        "total_return_pct": _avg("total_return_pct"),
        "annualized_return_pct": _avg("annualized_return_pct"),
        "net_profit": round(_sum("net_profit"), 2),
        # Risk
        "max_drawdown_pct": _max("max_drawdown_pct"),
        "sharpe_ratio": _avg("sharpe_ratio"),
        "sortino_ratio": _avg("sortino_ratio"),
        "calmar_ratio": _avg("calmar_ratio"),
        # Trades
        "total_trades": _sum("total_trades"),
        "win_rate_pct": _avg("win_rate_pct"),
        "profit_factor": _avg("profit_factor"),
        "expectancy": _avg("expectancy"),
        "payoff_ratio": _avg("payoff_ratio"),
        # Benchmark
        "benchmark_return_pct": _avg("benchmark_return_pct"),
        "excess_return_pct": _avg("excess_return_pct"),
        # Period
        "period_start": min(r.get("period_start", "") for r in valid),
        "period_end": max(r.get("period_end", "") for r in valid),
        "total_days": max(r.get("total_days", 0) for r in valid),
    }

    # Top 3 / Bottom 3 performers
    by_return = sorted(valid, key=lambda r: r.get("total_return_pct", 0), reverse=True)
    merged["top_performers"] = [
        {"symbol": r["symbol"], "return_pct": r.get("total_return_pct", 0)}
        for r in by_return[:3]
    ]
    merged["worst_performers"] = [
        {"symbol": r["symbol"], "return_pct": r.get("total_return_pct", 0)}
        for r in by_return[-3:]
    ]

    # Use first symbol's equity curve for the chart (kept internally, NOT in JSON response)
    for r in valid:
        if r.get("equity_curve"):
            merged["_equity_curve"] = r["equity_curve"]
            merged["_equity_symbol"] = r["symbol"]
            if r.get("benchmark_curve"):
                merged["_benchmark_curve"] = r["benchmark_curve"]
            break

    # Aggregate monthly P&L across symbols
    all_months: dict[str, float] = {}
    for r in valid:
        for month, pnl in r.get("monthly_pnl", {}).items():
            all_months[month] = all_months.get(month, 0) + pnl
    if all_months:
        merged["monthly_pnl"] = {k: round(v, 2) for k, v in sorted(all_months.items())}

    if failed:
        merged["failed_symbols"] = [r.get("symbol", "?") for r in failed]

    return merged


def _safe_float(val: Any) -> float:
    """Convert to float, handling inf/nan."""
    try:
        f = float(val)
        if math.isinf(f) or math.isnan(f):
            return 0.0
        return round(f, 4)
    except (TypeError, ValueError):
        return 0.0


# ═══════════════════════════════════════════════════════
# Chart generation (optional artifact)
# ═══════════════════════════════════════════════════════

def generate_chart(
    metrics: dict[str, Any],
    save_dir: Path | None = None,
) -> str | None:
    """Generate a professional multi-panel backtest dashboard.

    Panels:
      1. Equity curve + benchmark overlay + peak line
      2. Drawdown area chart
      3. Monthly P&L heatmap
      4. Key metrics summary box

    Returns file path or None if no equity data.
    """
    # Support both single-symbol (equity_curve) and multi-symbol (_equity_curve) layouts
    curve = metrics.get("equity_curve") or metrics.get("_equity_curve")
    if not curve:
        return None

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        logger.warning("matplotlib not installed, skipping chart")
        return None

    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except Exception:
        pass

    timestamps = pd.to_datetime([p["t"] for p in curve])
    equity_vals = np.array([p["v"] for p in curve], dtype=float)
    start_val = float(metrics.get("starting_balance", equity_vals[0]))

    # Benchmark curve
    bench_curve = metrics.get("benchmark_curve") or metrics.get("_benchmark_curve", [])
    has_bench = len(bench_curve) > 0

    # Monthly P&L
    monthly_pnl = metrics.get("monthly_pnl", {})
    has_monthly = len(monthly_pnl) > 0

    # Layout: 3 rows x 2 cols
    fig = plt.figure(figsize=(16, 10))
    grid = fig.add_gridspec(
        3, 2,
        height_ratios=[3.0, 1.5, 2.0],
        hspace=0.35, wspace=0.25,
    )
    ax_eq = fig.add_subplot(grid[0, :])      # full-width equity
    ax_dd = fig.add_subplot(grid[1, :])      # full-width drawdown
    ax_heat = fig.add_subplot(grid[2, 0])    # monthly heatmap
    ax_stats = fig.add_subplot(grid[2, 1])   # metrics box

    # ── Panel 1: Equity + Benchmark + Peak ──
    eq_series = pd.Series(equity_vals, index=timestamps)
    peak = eq_series.cummax()

    ax_eq.plot(timestamps, equity_vals, color="#1d4ed8", linewidth=2.0, label="Strategy")
    ax_eq.plot(timestamps, peak.values, color="#94a3b8", linewidth=1.2, linestyle="--", label="Peak", alpha=0.7)
    ax_eq.axhline(start_val, color="#6b7280", linestyle=":", linewidth=1.0, alpha=0.6)

    if has_bench:
        bench_ts = pd.to_datetime([p["t"] for p in bench_curve])
        bench_vals = [p["v"] for p in bench_curve]
        ax_eq.plot(bench_ts, bench_vals, color="#f59e0b", linewidth=1.6, linestyle="-.", label="Buy & Hold")

    ax_eq.fill_between(timestamps, equity_vals, start_val,
                       where=equity_vals >= start_val,
                       color="#22c55e", alpha=0.12, interpolate=True)
    ax_eq.fill_between(timestamps, equity_vals, start_val,
                       where=equity_vals < start_val,
                       color="#ef4444", alpha=0.10, interpolate=True)

    # Mark max drawdown point
    dd_vals = (eq_series / peak - 1) * 100
    dd_min_idx = dd_vals.idxmin()
    if pd.notna(dd_min_idx):
        ax_eq.scatter([dd_min_idx], [eq_series.loc[dd_min_idx]], color="#dc2626", s=40, zorder=5)
        ax_eq.annotate(
            f"Max DD {dd_vals.min():.1f}%",
            xy=(dd_min_idx, eq_series.loc[dd_min_idx]),
            xytext=(10, -18), textcoords="offset points", fontsize=8, color="#991b1b",
            bbox={"boxstyle": "round,pad=0.2", "fc": "#fee2e2", "ec": "#fecaca"},
        )

    # Build a short symbols label (max 3 shown, rest counted)
    all_symbols = metrics.get("symbols", [metrics.get("symbol", "")])
    if isinstance(all_symbols, list) and len(all_symbols) > 3:
        symbols_str = ", ".join(all_symbols[:3]) + f" +{len(all_symbols) - 3} more"
    elif isinstance(all_symbols, list):
        symbols_str = ", ".join(all_symbols)
    else:
        symbols_str = str(all_symbols)
    n_symbols = metrics.get("symbols_count", 1)
    title_suffix = f" ({n_symbols} symbols)" if n_symbols > 1 else ""

    ax_eq.set_title(
        f"{metrics.get('strategy_name', '')}{title_suffix} | "
        f"{metrics.get('period_start', '')} → {metrics.get('period_end', '')}",
        fontsize=12, fontweight="bold",
    )
    ax_eq.set_ylabel("Equity", fontsize=10)
    ax_eq.legend(loc="upper left", fontsize=9, ncol=4)
    ax_eq.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    # ── Panel 2: Drawdown ──
    ax_dd.fill_between(timestamps, dd_vals.values, 0, color="#ef4444", alpha=0.35)
    ax_dd.plot(timestamps, dd_vals.values, color="#b91c1c", linewidth=1.0)
    ax_dd.axhline(0, color="#6b7280", linestyle=":", linewidth=0.8)
    ax_dd.set_ylabel("Drawdown %", fontsize=10)
    ax_dd.set_title("Drawdown", fontsize=10, fontweight="bold")
    ax_dd.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    # ── Panel 3: Monthly heatmap ──
    ax_heat.set_title("Monthly P&L", fontsize=10, fontweight="bold")
    if has_monthly:
        try:
            month_series = pd.Series(monthly_pnl, dtype=float)
            month_series.index = pd.PeriodIndex(month_series.index, freq="M")
            pivot = month_series.groupby([month_series.index.year, month_series.index.month]).sum().unstack()
            pivot = pivot.reindex(columns=range(1, 13))

            vmax = float(np.nanmax(np.abs(pivot.values[np.isfinite(pivot.values)]))) if np.isfinite(pivot.values).any() else 1.0
            vmax = max(vmax, 1.0)
            heat = ax_heat.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=-vmax, vmax=vmax)
            month_labels = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]
            ax_heat.set_xticks(range(12))
            ax_heat.set_xticklabels(month_labels, fontsize=8)
            ax_heat.set_yticks(range(len(pivot.index)))
            ax_heat.set_yticklabels([str(y) for y in pivot.index], fontsize=8)
            for r in range(pivot.shape[0]):
                for c in range(pivot.shape[1]):
                    val = pivot.iloc[r, c]
                    if pd.notna(val):
                        ax_heat.text(c, r, f"{val:.0f}", ha="center", va="center", fontsize=7,
                                     color="#111827" if abs(val) < vmax * 0.6 else "white")
            fig.colorbar(heat, ax=ax_heat, fraction=0.046, pad=0.04).ax.tick_params(labelsize=7)
        except Exception as e:
            logger.warning(f"Monthly heatmap failed: {e}")
            ax_heat.text(0.5, 0.5, "Insufficient monthly data", ha="center", va="center", fontsize=10)
            ax_heat.set_axis_off()
    else:
        ax_heat.text(0.5, 0.5, "No monthly data", ha="center", va="center", fontsize=10)
        ax_heat.set_axis_off()

    # ── Panel 4: Metrics summary box ──
    ax_stats.axis("off")
    pf_val = metrics.get("profit_factor", 0)
    pf_text = f"{pf_val:.2f}" if pf_val < 100 else "∞"

    excess = metrics.get("excess_return_pct", 0)
    bench_line = f"Benchmark Ret : {metrics.get('benchmark_return_pct', 0):+.2f}%"
    excess_line = f"Excess Return : {excess:+.2f}%"

    summary = "\n".join([
        "PERFORMANCE",
        "─" * 32,
        f"Total Return  : {metrics.get('total_return_pct', 0):+.2f}%",
        f"Annualized    : {metrics.get('annualized_return_pct', 0):+.2f}%",
        f"Net Profit    : {metrics.get('net_profit', 0):+,.0f}",
        bench_line,
        excess_line,
        "",
        "RISK",
        "─" * 32,
        f"Max Drawdown  : {metrics.get('max_drawdown_pct', 0):.2f}%",
        f"DD Duration   : {metrics.get('max_drawdown_duration_days', 0):.0f}d",
        f"Sharpe        : {metrics.get('sharpe_ratio', 0):.2f}",
        f"Sortino       : {metrics.get('sortino_ratio', 0):.2f}",
        f"Calmar        : {metrics.get('calmar_ratio', 0):.2f}",
        "",
        "TRADES",
        "─" * 32,
        f"Total         : {metrics.get('total_trades', 0)}",
        f"Win Rate      : {metrics.get('win_rate_pct', 0):.1f}%",
        f"Profit Factor : {pf_text}",
        f"Payoff Ratio  : {metrics.get('payoff_ratio', 0):.2f}",
        f"Expectancy    : {metrics.get('expectancy', 0):.4f}",
        f"Avg Hold      : {metrics.get('avg_hold_time', 'N/A')}",
        f"Max Consec Loss: {metrics.get('max_consecutive_losses', 'N/A')}",
    ])
    ax_stats.text(
        0.02, 0.98, summary,
        transform=ax_stats.transAxes, va="top", ha="left",
        fontsize=9, family="monospace",
        bbox={"boxstyle": "round,pad=0.5", "fc": "#f8fafc", "ec": "#cbd5e1"},
    )

    # Suptitle
    fig.suptitle(
        f"Backtest Dashboard — {metrics.get('strategy_name', '')}",
        fontsize=14, fontweight="bold", y=0.99,
    )
    plt.tight_layout(rect=[0, 0.01, 1, 0.97])

    # Save
    if save_dir is None:
        import tempfile
        save_dir = Path(tempfile.gettempdir()) / "getall_charts"
    save_dir.mkdir(parents=True, exist_ok=True)

    name_token = re.sub(r"[^A-Za-z0-9._-]+", "_", metrics.get("strategy_name", "bt"))
    filename = f"{name_token}_{uuid.uuid4().hex[:8]}.png"
    path = save_dir / filename
    plt.savefig(str(path), dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info(f"Dashboard chart saved: {path}")
    return str(path)


# ═══════════════════════════════════════════════════════
# Data helpers (kept from old data_loader, slimmed down)
# ═══════════════════════════════════════════════════════

def ccxt_to_dataframe(ohlcv_list: list[dict[str, Any]], symbol: str = "") -> pd.DataFrame:
    """Convert ccxt OHLCV list[dict] to a standard DataFrame."""
    if not ohlcv_list:
        return pd.DataFrame()
    df = pd.DataFrame(ohlcv_list)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("timestamp", inplace=True)
    for col in ("open", "high", "low", "close", "volume"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df.attrs["symbol"] = symbol
    return df


def timeframe_to_seconds(tf: str) -> int:
    """Convert timeframe string (e.g. '4h') to seconds."""
    multipliers = {"m": 60, "h": 3600, "d": 86400, "w": 604800}
    unit = tf[-1].lower()
    try:
        value = int(tf[:-1])
    except ValueError:
        return 3600
    return value * multipliers.get(unit, 60)


def estimate_candle_count(period_str: str, timeframe: str) -> int:
    """Estimate number of candles needed for a period + timeframe combo."""
    unit = period_str[-1].lower()
    try:
        value = int(period_str[:-1])
    except ValueError:
        return 1000
    period_days = {"d": value, "w": value * 7, "m": value * 30, "y": value * 365}.get(unit, value)
    return int(period_days * 86400 / max(timeframe_to_seconds(timeframe), 1))
