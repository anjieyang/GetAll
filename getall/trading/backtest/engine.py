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


# pandas-ta can fail for some edge lengths (e.g., 1) on SMA.
def _to_int(value: Any, default: int) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _to_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    raw = value
    if isinstance(raw, str):
        raw = raw.strip().replace("%", "")
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _safe_sma(df: pd.DataFrame, params: dict[str, Any]) -> pd.Series:
    period = _to_int(params.get("period", 20), 20)
    if period <= 1:
        return df["close"].copy()
    out = ta.sma(df["close"], length=period)
    if isinstance(out, pd.Series) and len(out) == len(df.index):
        return out
    return df["close"].rolling(window=period, min_periods=period).mean()


def _normalize_token(raw: Any) -> str:
    text = str(raw or "").strip().lower()
    if not text:
        return ""
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


_INDICATOR_ALIASES: dict[str, str] = {
    "bbands": "bollinger",
    "bband": "bollinger",
    "bollinger_bands": "bollinger",
    "bollinger_band": "bollinger",
    "bollingerbands": "bollinger",
    "stochastic": "stoch",
    "stochastics": "stoch",
    "moving_average": "sma",
    "movingaverage": "sma",
    "ma": "sma",
}

_FIELD_ALIASES: dict[str, str] = {
    "val": "value",
    "current": "value",
    "middle": "mid",
    "midline": "mid",
    "center": "mid",
    "upper_band": "upper",
    "lower_band": "lower",
    "upperband": "upper",
    "lowerband": "lower",
    "signal_line": "signal",
    "hist": "histogram",
    "pct": "percent",
    "percent_b": "percent",
}

_OPERATOR_ALIASES: dict[str, str] = {
    "<": "lt",
    ">": "gt",
    "<=": "lte",
    ">=": "gte",
    "==": "eq",
    "=": "eq",
    "!=": "ne",
    "<>": "ne",
    "less_than": "lt",
    "greater_than": "gt",
    "less_or_equal": "lte",
    "greater_or_equal": "gte",
    "crossabove": "cross_above",
    "cross_over": "cross_above",
    "crossover": "cross_above",
    "crossbelow": "cross_below",
    "cross_under": "cross_below",
    "crossunder": "cross_below",
}

_DIRECTION_ALIASES: dict[str, str] = {
    "long_only": "long",
    "longonly": "long",
    "spot": "long",
    "short_only": "short",
    "shortonly": "short",
    "long_short": "both",
    "longshort": "both",
}

_SUPPORTED_OPERATORS = {"lt", "gt", "lte", "gte", "eq", "ne", "cross_above", "cross_below"}


def _normalize_indicator_name(raw: Any) -> str:
    token = _normalize_token(raw)
    if not token:
        return ""
    return _INDICATOR_ALIASES.get(token, token)


def _normalize_field_name(raw: Any) -> str:
    token = _normalize_token(raw)
    if not token:
        return "value"
    return _FIELD_ALIASES.get(token, token)


def _normalize_direction(raw: Any) -> str:
    token = _normalize_token(raw)
    if not token:
        return "long"
    return _DIRECTION_ALIASES.get(token, token)


def _dedupe_non_empty(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for v in values:
        if not v or v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out


def _indicator_candidates(raw: Any) -> list[str]:
    text = str(raw or "").strip()
    token = _normalize_token(text)
    canonical = _normalize_indicator_name(text)
    return _dedupe_non_empty([text, text.lower(), token, canonical])


def _field_candidates(raw: Any) -> list[str]:
    text = str(raw or "").strip()
    token = _normalize_token(text)
    canonical = _normalize_field_name(text)
    return _dedupe_non_empty([text, text.lower(), token, canonical])


def _normalize_indicator_params(name: str, raw_params: Any) -> dict[str, Any]:
    if not isinstance(raw_params, dict):
        return {}

    token_params = {_normalize_token(k): v for k, v in raw_params.items()}
    normalized: dict[str, Any] = dict(token_params)

    period = token_params.get("period", token_params.get("length", token_params.get("window")))
    if period is not None:
        normalized["period"] = _to_int(period, 20)

    if name == "bollinger":
        std = token_params.get("std", token_params.get("stddev", token_params.get("deviation")))
        if std is not None:
            normalized["std"] = _to_float(std, 2.0)
    elif name == "macd":
        fast = token_params.get("fast_period", token_params.get("fast"))
        slow = token_params.get("slow_period", token_params.get("slow"))
        signal = token_params.get("signal_period", token_params.get("signal"))
        if fast is not None:
            normalized["fast_period"] = _to_int(fast, 12)
        if slow is not None:
            normalized["slow_period"] = _to_int(slow, 26)
        if signal is not None:
            normalized["signal_period"] = _to_int(signal, 9)
    elif name == "stoch":
        k = token_params.get("period_k", token_params.get("k"))
        d = token_params.get("period_d", token_params.get("d"))
        if k is not None:
            normalized["period_k"] = _to_int(k, 14)
        if d is not None:
            normalized["period_d"] = _to_int(d, 3)

    return normalized


def _lookup_indicator_series(
    indicators: dict[str, pd.Series],
    indicator_raw: Any,
    field_raw: Any,
) -> pd.Series | None:
    base_keys = _indicator_candidates(indicator_raw)
    field_keys = _field_candidates(field_raw)
    field_norm = _normalize_field_name(field_raw)

    lookup_keys: list[str] = []
    if field_norm == "value":
        for base in base_keys:
            lookup_keys.extend([base, f"{base}_value"])
    else:
        for base in base_keys:
            for field in field_keys:
                lookup_keys.append(f"{base}_{field}")
            lookup_keys.append(base)

    for key in _dedupe_non_empty(lookup_keys):
        series = indicators.get(key)
        if isinstance(series, pd.Series):
            return series
    return None


# ═══════════════════════════════════════════════════════
# Indicator computation
# ═══════════════════════════════════════════════════════

_INDICATOR_REGISTRY: dict[str, dict[str, Any]] = {
    # Moving averages
    "sma": {"fn": _safe_sma, "defaults": {"period": 20}},
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
        if not isinstance(cfg, dict):
            logger.warning(f"Invalid indicator config type {type(cfg).__name__}, skipping")
            continue

        name = _normalize_indicator_name(cfg.get("name"))
        if not name:
            logger.warning("Indicator config missing 'name', skipping")
            continue

        reg = _INDICATOR_REGISTRY.get(name)
        if reg is None:
            logger.warning(f"Unknown indicator '{name}', skipping")
            continue

        params = {
            **reg.get("defaults", {}),
            **_normalize_indicator_params(name, cfg.get("params", {})),
        }
        key = _normalize_token(cfg.get("key", name)) or name

        try:
            out = reg["fn"](df, params)
        except Exception as e:
            logger.warning(f"Error computing indicator {name}: {e}")
            continue

        if reg.get("multi") and isinstance(out, pd.DataFrame) and out is not None:
            field_map = _MULTI_FIELD_MAP.get(name, {})
            for field_name, col_idx in field_map.items():
                if col_idx < len(out.columns):
                    series = out.iloc[:, col_idx]
                    result[f"{key}_{field_name}"] = series
                    result.setdefault(f"{name}_{field_name}", series)
            # Also store first column as plain key for simple references
            first_col = out.iloc[:, 0]
            result[key] = first_col
            result.setdefault(name, first_col)
        elif isinstance(out, pd.Series):
            result[key] = out
            result.setdefault(name, out)

    # Always include OHLCV as referenceable series
    for col in ("open", "high", "low", "close", "volume"):
        if col in df.columns:
            result[col] = df[col]
    # Common alias used by strategy prompts.
    if "close" in df.columns:
        result["price"] = df["close"]
    return result


# ═══════════════════════════════════════════════════════
# Condition evaluation → boolean signal arrays
# ═══════════════════════════════════════════════════════

def _resolve_value(
    raw: Any,
    indicators: dict[str, pd.Series],
) -> pd.Series | float:
    """Resolve a condition threshold — either a literal number or an indicator reference."""
    if isinstance(raw, pd.Series):
        return raw
    if isinstance(raw, (int, float)):
        return float(raw)
    s = str(raw).strip()
    if not s:
        return float("nan")

    # "indicator.field" e.g. "macd.signal" → key "macd_signal"
    if "." in s:
        ref_key, ref_field = s.split(".", 1)
        ref_series = _lookup_indicator_series(indicators, ref_key, ref_field)
        if ref_series is not None:
            return ref_series

    # Direct key
    for key in _indicator_candidates(s):
        if key in indicators:
            return indicators[key]

    # Try as number
    try:
        return float(s)
    except ValueError:
        pass
    return float("nan")


def _normalize_operator(op: str) -> str:
    """Normalize operator aliases to canonical engine operator names."""
    raw = str(op or "").strip().lower()
    if raw in _OPERATOR_ALIASES:
        return _OPERATOR_ALIASES[raw]
    token = _normalize_token(raw)
    return _OPERATOR_ALIASES.get(token, raw)


def _eval_op(current: pd.Series, op: str, threshold: pd.Series | float) -> pd.Series:
    """Apply a comparison operator and return a boolean Series."""
    op = _normalize_operator(op)
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
    if op == "ne":
        if isinstance(threshold, pd.Series):
            return (current - threshold).abs() >= 1e-8
        return (current - threshold).abs() >= 1e-8
    if op == "cross_above":
        prev = current.shift(1)
        if isinstance(threshold, pd.Series):
            return (prev <= threshold.shift(1)) & (current > threshold)
        return (prev <= threshold) & (current > threshold)
    if op == "cross_below":
        prev = current.shift(1)
        if isinstance(threshold, pd.Series):
            return (prev >= threshold.shift(1)) & (current < threshold)
        return (prev >= threshold) & (current < threshold)
    logger.warning(f"Unknown operator '{op}', returning False")
    return pd.Series(False, index=current.index)


def _describe_condition(cond: dict[str, Any]) -> str:
    """Human-readable one-liner for a condition dict."""
    ind = cond.get("indicator", "?")
    field = cond.get("field", "value")
    op = cond.get("operator", "?")
    val = cond.get("value", "?")
    if isinstance(val, pd.Series):
        val = "<series>"
    return f"{ind}.{field} {op} {val}"


def evaluate_conditions(
    conditions: list[dict[str, Any]],
    indicators: dict[str, pd.Series],
    logic: str = "and",
    diagnostics: list[dict[str, Any]] | None = None,
) -> pd.Series:
    """Evaluate a list of conditions and combine with AND or OR logic.

    Each condition: {"indicator": str, "field": str, "operator": str, "value": ...}

    Args:
        diagnostics: Mutable list — per-condition hit stats are always appended
            when provided.  Each entry contains:
            condition, hits, hit_pct, valid_bars, nan_bars, status.
    """
    index = next(iter(indicators.values())).index if indicators else pd.RangeIndex(0)
    total_bars = len(index)
    if not conditions:
        return pd.Series(False, index=index)

    signals: list[pd.Series] = []
    for idx, cond in enumerate(conditions):
        if not isinstance(cond, dict):
            logger.warning(f"Condition #{idx} is not an object, skipped")
            if diagnostics is not None:
                diagnostics.append({
                    "condition": f"#{idx} (not a dict)",
                    "hits": 0, "hit_pct": 0.0, "valid_bars": 0,
                    "nan_bars": total_bars, "status": "invalid",
                })
            continue

        indicator_raw = cond.get("indicator")
        if indicator_raw is None:
            logger.warning(f"Condition #{idx} missing 'indicator', skipped")
            if diagnostics is not None:
                diagnostics.append({
                    "condition": _describe_condition(cond),
                    "hits": 0, "hit_pct": 0.0, "valid_bars": 0,
                    "nan_bars": total_bars, "status": "missing_indicator",
                })
            continue
        field_raw = cond.get("field", "value")

        # Allow shorthand: {"indicator": "macd.signal", ...}
        if isinstance(indicator_raw, str) and "." in indicator_raw and _normalize_field_name(field_raw) == "value":
            indicator_raw, field_raw = indicator_raw.split(".", 1)

        current = _lookup_indicator_series(indicators, indicator_raw, field_raw)
        if current is None:
            logger.warning(f"Indicator '{indicator_raw}.{field_raw}' not found, condition skipped")
            if diagnostics is not None:
                diagnostics.append({
                    "condition": _describe_condition(cond),
                    "hits": 0, "hit_pct": 0.0, "valid_bars": 0,
                    "nan_bars": total_bars, "status": "not_found",
                })
            continue

        if "value" not in cond:
            logger.warning(f"Condition #{idx} missing 'value', skipped")
            if diagnostics is not None:
                diagnostics.append({
                    "condition": _describe_condition(cond),
                    "hits": 0, "hit_pct": 0.0, "valid_bars": 0,
                    "nan_bars": total_bars, "status": "missing_value",
                })
            continue
        threshold = _resolve_value(cond.get("value"), indicators)
        if isinstance(threshold, float) and math.isnan(threshold):
            logger.warning(f"Condition #{idx} has unresolved threshold '{cond.get('value')}', skipped")
            if diagnostics is not None:
                diagnostics.append({
                    "condition": _describe_condition(cond),
                    "hits": 0, "hit_pct": 0.0, "valid_bars": 0,
                    "nan_bars": total_bars, "status": "unresolved_threshold",
                })
            continue

        op = _normalize_operator(cond.get("operator"))
        if op not in _SUPPORTED_OPERATORS:
            logger.warning(f"Condition #{idx} has unsupported operator '{cond.get('operator')}', skipped")
            if diagnostics is not None:
                diagnostics.append({
                    "condition": _describe_condition(cond),
                    "hits": 0, "hit_pct": 0.0, "valid_bars": 0,
                    "nan_bars": total_bars, "status": "bad_operator",
                })
            continue

        sig = _eval_op(current, op, threshold)
        signals.append(sig)

        # Record per-condition diagnostics (with NaN tracking)
        if diagnostics is not None:
            nan_count = int(current.isna().sum())
            if isinstance(threshold, pd.Series):
                nan_count = max(nan_count, int(threshold.isna().sum()))
            valid_bars = total_bars - nan_count
            hits = int(sig.sum())
            hit_pct = round(hits / max(valid_bars, 1) * 100, 2)
            diagnostics.append({
                "condition": f"{indicator_raw}.{field_raw} {op} {cond.get('value')}",
                "hits": hits,
                "hit_pct": hit_pct,
                "valid_bars": valid_bars,
                "nan_bars": nan_count,
                "status": "ok",
            })

    if not signals:
        if diagnostics is not None:
            diagnostics.append({
                "condition": f"combined ({_normalize_token(logic)})",
                "hits": 0, "hit_pct": 0.0, "valid_bars": 0,
                "nan_bars": total_bars, "status": "combined",
            })
        return pd.Series(False, index=index)

    combined = signals[0]
    logic_norm = _normalize_token(logic)
    for s in signals[1:]:
        combined = (combined & s) if logic_norm != "or" else (combined | s)
    result = combined.fillna(False)

    if diagnostics is not None:
        combined_hits = int(result.sum())
        combined_pct = round(combined_hits / max(total_bars, 1) * 100, 2)
        diagnostics.append({
            "condition": f"combined ({logic_norm})",
            "hits": combined_hits,
            "hit_pct": combined_pct,
            "valid_bars": total_bars,
            "nan_bars": 0,
            "status": "combined",
        })

    return result


# ═══════════════════════════════════════════════════════
# Zero-signal diagnosis
# ═══════════════════════════════════════════════════════

def _diagnose_zero_signals(
    conditions: list[dict[str, Any]],
    indicators: dict[str, pd.Series],
    entry_diag: list[dict[str, Any]],
    total_bars: int,
) -> dict[str, Any]:
    """Analyze WHY combined entry signals = 0 and return structured diagnosis.

    Runs a *drop-one* analysis: for each condition, removes it and checks how
    many bars the remaining conditions match.  This pinpoints which condition
    (or pair) is the bottleneck.

    Returns a dict with keys: problem, detail, bottleneck, drop_one, suggestions.
    """
    ok_entries = [d for d in entry_diag if d.get("status") == "ok"]
    skipped = [d for d in entry_diag if d.get("status") not in ("ok", "combined")]

    analysis: dict[str, Any] = {
        "total_bars": total_bars,
        "conditions_ok": len(ok_entries),
        "conditions_skipped": len(skipped),
    }

    # ── Case B4: all conditions skipped ──
    if not ok_entries:
        skipped_details = [
            {"condition": d["condition"], "reason": d["status"]} for d in skipped
        ]
        # Distinguish: unresolved thresholds (likely warmup) vs bad names
        unresolved = [d for d in skipped if d.get("status") == "unresolved_threshold"]
        not_found = [d for d in skipped if d.get("status") == "not_found"]

        if unresolved and not not_found:
            # All failures are threshold resolution → likely indicator warmup / data too short
            analysis["problem"] = "unresolved_thresholds"
            refs = [d["condition"] for d in unresolved]
            analysis["detail"] = (
                f"Threshold reference(s) could not be resolved: {refs}. "
                "Most likely the referenced indicator produced all NaN because "
                "the data period is too short for the indicator's lookback window."
            )
            analysis["suggestions"] = [
                "Use a longer period (e.g. 1m → 6m) to provide enough bars for indicator warmup.",
                "Reduce indicator period (e.g. EMA200 → EMA50).",
            ]
        else:
            analysis["problem"] = "all_conditions_skipped"
            analysis["detail"] = (
                f"None of the {len(conditions)} entry conditions could be evaluated. "
                "Check indicator names, operator spelling, and threshold references."
            )
            analysis["suggestions"] = [
                "Fix condition config — no signals can be generated as-is.",
            ]
        analysis["skipped"] = skipped_details
        return analysis

    # ── Case C1: warmup eats most data ──
    worst_nan = max(ok_entries, key=lambda d: d.get("nan_bars", 0))
    nan_ratio = worst_nan.get("nan_bars", 0) / max(total_bars, 1)
    if nan_ratio > 0.8:
        analysis["problem"] = "insufficient_data_after_warmup"
        analysis["detail"] = (
            f"Indicator warmup consumes {worst_nan['nan_bars']}/{total_bars} bars "
            f"({nan_ratio:.0%}). Only {worst_nan.get('valid_bars', 0)} bars are "
            f"evaluable for condition '{worst_nan['condition']}'."
        )
        analysis["suggestions"] = [
            "Use a longer period (e.g. 3m → 6m) to provide more data after warmup.",
            "Reduce indicator period (e.g. EMA200 → EMA50).",
        ]
        return analysis

    # ── Case A2: one condition individually has 0 hits ──
    zero_hit = [d for d in ok_entries if d["hits"] == 0]
    if zero_hit:
        bottleneck = zero_hit[0]
        analysis["problem"] = "impossible_condition"
        analysis["detail"] = (
            f"Condition '{bottleneck['condition']}' never triggers across "
            f"{bottleneck.get('valid_bars', total_bars)} valid bars."
        )
        analysis["bottleneck"] = bottleneck["condition"]
        analysis["suggestions"] = [
            f"Relax the threshold for '{bottleneck['condition']}'.",
            "Check if the threshold is realistic for this timeframe / coin set.",
        ]
        return analysis

    # ── Case A1/A3: conditions individually fire but AND = 0 → contradictory ──
    # Drop-one analysis: remove each condition, re-evaluate the rest.
    drop_one: list[dict[str, Any]] = []
    for i, cond in enumerate(conditions):
        if not isinstance(cond, dict):
            continue
        subset = conditions[:i] + conditions[i + 1:]
        if not subset:
            continue
        # Re-evaluate without diagnostics (fast path)
        remaining_sig = evaluate_conditions(subset, indicators, logic="and")
        remaining_hits = int(remaining_sig.sum())
        drop_one.append({
            "dropped": _describe_condition(cond),
            "remaining_hits": remaining_hits,
            "remaining_pct": round(remaining_hits / max(total_bars, 1) * 100, 2),
        })

    analysis["drop_one"] = drop_one

    if drop_one:
        best = max(drop_one, key=lambda d: d["remaining_hits"])
        if best["remaining_hits"] > 0:
            analysis["problem"] = "contradictory_conditions"
            analysis["detail"] = (
                f"Each condition fires individually, but they never overlap (AND = 0). "
                f"Removing '{best['dropped']}' would produce {best['remaining_hits']} "
                f"entry signals ({best['remaining_pct']}%)."
            )
            analysis["bottleneck"] = best["dropped"]
            analysis["suggestions"] = [
                f"Remove or relax '{best['dropped']}' — it contradicts the other conditions.",
                "Switch from AND to OR logic if conditions represent alternative signals.",
                "Change timeframe (shorter TF = more candles = more chances for overlap).",
            ]
        else:
            # Even dropping one condition still yields 0 — deeply contradictory
            analysis["problem"] = "all_pairs_contradictory"
            analysis["detail"] = (
                "Removing any single condition still yields 0 signals. "
                "Multiple condition pairs are mutually exclusive."
            )
            analysis["suggestions"] = [
                "Redesign entry logic — current conditions cannot co-occur.",
                "Try keeping only 1-2 conditions and test individually.",
            ]
    else:
        analysis["problem"] = "unknown"
        analysis["detail"] = "Could not determine cause of zero signals."

    return analysis


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
    direction = _normalize_direction(config.get("direction", "long"))
    sl_pct = _to_float(config.get("stop_loss_pct"))
    tp_pct = _to_float(config.get("take_profit_pct"))
    fees = _to_float(config.get("fees"), 0.0006) or 0.0006

    size_raw = _to_float(config.get("trade_size_pct"), 100.0)
    if size_raw is None or size_raw <= 0:
        logger.warning(f"Invalid trade_size_pct '{config.get('trade_size_pct')}', fallback to 100")
        size_raw = 100.0
    size_pct = size_raw / 100.0
    timeframe = str(config.get("timeframe", "1h") or "1h")

    all_results: list[dict[str, Any]] = []

    # ── Diagnostics tracking (across all symbols) ──
    first_entry_diag: list[dict[str, Any]] = []
    first_exit_diag: list[dict[str, Any]] = []
    first_diag_symbol: str = ""
    first_diag_indicators: dict[str, pd.Series] = {}
    first_diag_total_bars: int = 0
    symbols_with_zero_entries: int = 0
    symbols_total: int = 0

    for symbol, df in ohlcv_data.items():
        if df.empty:
            continue
        symbols_total += 1

        # 1. Compute indicators
        indicators = compute_indicators(df, config.get("indicators", []))

        # 2. Evaluate entry / exit signals (always with diagnostics)
        entry_diag: list[dict[str, Any]] = []
        exit_diag: list[dict[str, Any]] = []
        entry_signals = evaluate_conditions(
            config.get("entry_conditions", []), indicators, logic="and", diagnostics=entry_diag,
        )
        exit_signals = evaluate_conditions(
            config.get("exit_conditions", []), indicators, logic="or", diagnostics=exit_diag,
        )

        entry_hits = int(entry_signals.sum())
        if entry_hits == 0:
            symbols_with_zero_entries += 1

        # Save first symbol's diagnostics + indicators for deep analysis later
        if not first_diag_symbol:
            first_entry_diag = entry_diag
            first_exit_diag = exit_diag
            first_diag_symbol = symbol
            first_diag_indicators = indicators
            first_diag_total_bars = len(df)

        # 3. Build VectorBT portfolio
        close = df["close"]
        pf_kwargs: dict[str, Any] = {
            "close": close,
            "init_cash": starting_balance,
            "size": size_pct,
            "size_type": "percent",
            "fees": fees,
            "freq": pd.Timedelta(seconds=max(timeframe_to_seconds(timeframe), 1)),
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

    # ── Always include signal diagnostics ──
    if first_entry_diag:
        result["entry_signal_diagnostics"] = [
            {"symbol": first_diag_symbol, **d} for d in first_entry_diag
        ]
    if first_exit_diag:
        result["exit_signal_diagnostics"] = [
            {"symbol": first_diag_symbol, **d} for d in first_exit_diag
        ]

    # ── Deep diagnosis when zero trades detected ──
    total_trades = result.get("total_trades", 0)
    if isinstance(total_trades, float):
        total_trades = int(total_trades)
    if total_trades == 0 and symbols_with_zero_entries > 0:
        result["signal_analysis"] = _diagnose_zero_signals(
            config.get("entry_conditions", []),
            first_diag_indicators,
            first_entry_diag,
            first_diag_total_bars,
        )
        result["signal_analysis"]["symbols_with_zero_entries"] = symbols_with_zero_entries
        result["signal_analysis"]["symbols_total"] = symbols_total

    # ── Warn when conditions were silently skipped (even if trades > 0) ──
    skipped_diags = [
        d for d in first_entry_diag
        if d.get("status") not in ("ok", "combined")
    ]
    if skipped_diags:
        result["skipped_conditions_warning"] = [
            {"condition": d["condition"], "reason": d["status"]}
            for d in skipped_diags
        ]

    return result


def _filter_directional(
    conditions: list[dict[str, Any]],
    target_dir: str,
    indicators: dict[str, pd.Series],
    logic: str = "and",
) -> pd.Series:
    """Filter conditions by direction tag and evaluate."""
    filtered = []
    target = _normalize_direction(target_dir)
    for c in conditions:
        raw_direction = str(c.get("direction", "") or "").strip()
        if raw_direction:
            d = _normalize_direction(raw_direction)
            if d != target:
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
    max_dd_pct = abs(_safe_metric_float(lambda: pf.drawdowns.max_drawdown())) * 100

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
        "max_drawdown_pct": round(max_dd_pct, 2),
        "max_drawdown_duration_days": round(dd_duration_days, 1),
        "sharpe_ratio": _safe_float(_safe_metric_float(lambda: pf.sharpe_ratio())),
        "sortino_ratio": _safe_float(_safe_metric_float(lambda: pf.sortino_ratio())),
        "calmar_ratio": round(ann_return / max(max_dd_pct, 0.01), 2),
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
                exit_ts = pd.to_datetime(monthly_df["Exit Timestamp"], errors="coerce", utc=True)
                monthly_df["month"] = exit_ts.dt.tz_localize(None).dt.to_period("M")
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


def _safe_metric_float(metric_fn: Any, default: float = 0.0) -> float:
    """Safely evaluate a portfolio metric function that may raise."""
    try:
        return float(metric_fn())
    except Exception as e:
        logger.warning(f"Metric evaluation failed, fallback to {default}: {e}")
        return default


# ═══════════════════════════════════════════════════════
# Chart generation (optional artifact)
# ═══════════════════════════════════════════════════════

async def generate_chart(
    metrics: dict[str, Any],
    save_dir: Path | None = None,
) -> str | None:
    """Generate a professional multi-panel backtest dashboard.

    Tries the Playwright/ECharts renderer first for modern, high-quality output.
    Falls back to matplotlib if Playwright is unavailable or fails.

    Returns file path or *None* if no equity data.
    """
    curve = metrics.get("equity_curve") or metrics.get("_equity_curve")
    if not curve:
        return None

    # ── Try modern renderer first ──
    try:
        from getall.charts import PLAYWRIGHT_AVAILABLE, render_chart

        if PLAYWRIGHT_AVAILABLE:
            path = await render_chart(
                "backtest_dashboard",
                metrics,
                save_dir=save_dir,
            )
            logger.info(f"Dashboard chart saved (ECharts): {path}")
            return str(path)
    except Exception as exc:
        logger.warning(f"ECharts renderer failed, falling back to matplotlib: {exc}")

    # ── Matplotlib fallback ──
    return await asyncio.to_thread(_generate_chart_matplotlib, metrics, save_dir)


def _generate_chart_matplotlib(
    metrics: dict[str, Any],
    save_dir: Path | None = None,
) -> str | None:
    """Matplotlib fallback chart generator (legacy).

    Panels:
      1. Equity curve + benchmark overlay + peak line
      2. Drawdown area chart
      3. Monthly P&L heatmap
      4. Key metrics summary box
    """
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
    logger.info(f"Dashboard chart saved (matplotlib fallback): {path}")
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
