"""Ephemeral watch scope for proactive monitoring.

This module maintains a small, TTL-based list of coins the system should
actively monitor even if they are not yet in positions.json (e.g. user
just placed an entry order that has not filled).

Storage: {workspace}/memory/trading/watch_scope.json
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any


DEFAULT_TTL_HOURS = 24


@dataclass
class WatchCoin:
    symbol: str  # base coin symbol, e.g. "BTC"
    source: str  # e.g. "ws:order_open", "ws:position_change"
    added_at: str
    expires_at: str


def watch_scope_path(workspace_path: Path) -> Path:
    return workspace_path / "memory" / "trading" / "watch_scope.json"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.isoformat()


def _normalize_coin(raw: str) -> str:
    """
    Normalize any of:
    - base coin: "eth"
    - ccxt symbol: "ETH/USDT:USDT"
    - pair: "ETH/USDT"
    into base coin: "ETH"
    """
    s = (raw or "").strip()
    if not s:
        return ""
    if "/" in s:
        s = s.split("/", 1)[0]
    if ":" in s:
        s = s.split(":", 1)[0]
    return s.strip().upper()


def load_watch_scope(workspace_path: Path) -> dict[str, Any]:
    path = watch_scope_path(workspace_path)
    if not path.exists():
        return {"updated_at": None, "coins": []}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return {"updated_at": None, "coins": []}
        data.setdefault("updated_at", None)
        data.setdefault("coins", [])
        if not isinstance(data["coins"], list):
            data["coins"] = []
        return data
    except (OSError, json.JSONDecodeError):
        return {"updated_at": None, "coins": []}


def _prune_expired(coins: list[dict[str, Any]]) -> list[dict[str, Any]]:
    now = _now()
    out: list[dict[str, Any]] = []
    for c in coins:
        if not isinstance(c, dict):
            continue
        sym = _normalize_coin(str(c.get("symbol") or ""))
        if not sym:
            continue
        exp = c.get("expires_at")
        if not isinstance(exp, str) or not exp:
            continue
        try:
            exp_dt = datetime.fromisoformat(exp)
        except ValueError:
            continue
        if exp_dt.tzinfo is None:
            exp_dt = exp_dt.replace(tzinfo=timezone.utc)
        if exp_dt > now:
            c2 = dict(c)
            c2["symbol"] = sym
            out.append(c2)
    return out


def upsert_watch_coins(
    workspace_path: Path,
    coins: list[str],
    *,
    source: str,
    ttl_hours: int = DEFAULT_TTL_HOURS,
) -> list[str]:
    """
    Add or refresh coins in watch_scope.json.

    Returns:
        List of normalized symbols that were upserted.
    """
    normalized = [_normalize_coin(c) for c in coins]
    normalized = [c for c in normalized if c]
    if not normalized:
        return []

    data = load_watch_scope(workspace_path)
    existing = _prune_expired(list(data.get("coins") or []))

    now = _now()
    exp = now + timedelta(hours=ttl_hours)

    by_symbol: dict[str, dict[str, Any]] = {str(c.get("symbol")): c for c in existing}
    for sym in normalized:
        by_symbol[sym] = asdict(
            WatchCoin(
                symbol=sym,
                source=source,
                added_at=_iso(now),
                expires_at=_iso(exp),
            )
        )

    path = watch_scope_path(workspace_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = {
        "updated_at": _iso(now),
        "coins": sorted(by_symbol.values(), key=lambda x: str(x.get("symbol", ""))),
    }
    try:
        path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        # Best-effort: ignore persistence failure
        pass

    return normalized


def get_active_watch_coins(workspace_path: Path) -> set[str]:
    """Return active coin symbols (base, uppercase) from watch_scope.json."""
    data = load_watch_scope(workspace_path)
    coins = _prune_expired(list(data.get("coins") or []))
    return {str(c.get("symbol")) for c in coins if isinstance(c, dict) and c.get("symbol")}


def extract_coin(raw_symbol: str) -> str:
    """Public helper: extract base coin from a ccxt symbol/pair/base symbol."""
    return _normalize_coin(raw_symbol)

