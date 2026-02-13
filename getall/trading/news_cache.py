"""News cache persistence for trading module.

Design goals:
- Always keep a *latest* snapshot for debugging (even if empty).
- Keep a history log for time-window queries.
- Keep a last-non-empty snapshot so user queries can avoid returning empties.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from loguru import logger


_LATEST_FILE = "news_cache_latest.json"
_HISTORY_FILE = "news_cache_history.jsonl"
_LAST_NON_EMPTY_FILE = "news_cache_last_non_empty.json"


@dataclass(frozen=True)
class NewsCachePaths:
    trading_dir: Path
    latest: Path
    history: Path
    last_non_empty: Path


def _ensure_trading_dir(workspace_path: Path) -> Path:
    trading_dir = workspace_path / "memory" / "trading"
    trading_dir.mkdir(parents=True, exist_ok=True)
    return trading_dir


def get_news_cache_paths(workspace_path: Path) -> NewsCachePaths:
    trading_dir = _ensure_trading_dir(workspace_path)
    return NewsCachePaths(
        trading_dir=trading_dir,
        latest=trading_dir / _LATEST_FILE,
        history=trading_dir / _HISTORY_FILE,
        last_non_empty=trading_dir / _LAST_NON_EMPTY_FILE,
    )


def _atomic_write_text(path: Path, content: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)


def snapshot_has_content(snapshot: dict[str, Any]) -> bool:
    """Return True if snapshot contains any meaningful news/topic/opinion items."""
    if not isinstance(snapshot, dict):
        return False
    data = snapshot.get("data")
    if not isinstance(data, dict):
        return False

    trending_news = data.get("trending_news")
    if isinstance(trending_news, list) and len(trending_news) > 0:
        return True

    trending_topics = data.get("trending_topics")
    if isinstance(trending_topics, list) and len(trending_topics) > 0:
        return True

    kol = data.get("kol_opinions")
    if isinstance(kol, dict):
        for v in kol.values():
            if isinstance(v, list) and len(v) > 0:
                return True

    return False


def parse_timestamp(value: Any) -> datetime | None:
    """Parse timestamps used in cache snapshots. Accepts ISO or unix seconds/millis."""
    if value is None:
        return None
    try:
        if isinstance(value, (int, float)):
            ts = float(value)
            if ts > 1e12:
                ts = ts / 1000.0
            return datetime.fromtimestamp(ts, tz=timezone.utc)
        if isinstance(value, str):
            v = value.strip()
            if not v:
                return None
            return datetime.fromisoformat(v.replace("Z", "+00:00"))
    except Exception:
        return None
    return None


def snapshot_ts(snapshot: dict[str, Any]) -> datetime | None:
    return parse_timestamp(snapshot.get("ts"))


def persist_news_snapshot(
    workspace_path: Path,
    snapshot: dict[str, Any],
    *,
    write_latest: bool = True,
    append_history: bool = True,
    write_last_non_empty: bool = True,
) -> NewsCachePaths:
    """Persist snapshot to latest/history/last_non_empty files.

    - latest: always overwritten (pretty JSON)
    - history: JSONL append (one snapshot per line)
    - last_non_empty: overwritten only when snapshot has content
    """
    paths = get_news_cache_paths(workspace_path)
    try:
        encoded_pretty = json.dumps(snapshot, ensure_ascii=False, indent=2)
    except TypeError as e:
        # Ensure we don't crash cron runs on non-serializable values.
        logger.warning(f"[news_cache] snapshot not JSON-serializable, skipping persist: {e}")
        return paths

    if write_latest:
        try:
            _atomic_write_text(paths.latest, encoded_pretty + "\n")
        except OSError as e:
            logger.warning(f"[news_cache] failed to write latest snapshot: {e}")

    if append_history:
        try:
            line = json.dumps(snapshot, ensure_ascii=False)
            with paths.history.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError as e:
            logger.warning(f"[news_cache] failed to append history: {e}")

    if write_last_non_empty and snapshot_has_content(snapshot):
        try:
            _atomic_write_text(paths.last_non_empty, encoded_pretty + "\n")
        except OSError as e:
            logger.warning(f"[news_cache] failed to write last_non_empty: {e}")

    return paths


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def load_latest_snapshot(workspace_path: Path) -> dict[str, Any] | None:
    return _load_json(get_news_cache_paths(workspace_path).latest)


def load_last_non_empty_snapshot(workspace_path: Path) -> dict[str, Any] | None:
    return _load_json(get_news_cache_paths(workspace_path).last_non_empty)


def iter_history_snapshots(
    workspace_path: Path,
    *,
    since: datetime | None = None,
) -> Iterable[dict[str, Any]]:
    path = get_news_cache_paths(workspace_path).history
    if not path.exists():
        return []

    def _iter() -> Iterable[dict[str, Any]]:
        try:
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    raw = line.strip()
                    if not raw:
                        continue
                    try:
                        obj = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(obj, dict):
                        continue
                    if since is not None:
                        ts = snapshot_ts(obj)
                        if ts is None or ts < since:
                            continue
                    yield obj
        except OSError as e:
            logger.debug(f"[news_cache] failed to read history: {e}")
            return

    return _iter()


def pick_best_snapshot(
    snapshots: list[dict[str, Any]],
    *,
    prefer_non_empty: bool = True,
) -> dict[str, Any] | None:
    """Pick the best snapshot from a list (usually a time window)."""
    if not snapshots:
        return None

    def _key(s: dict[str, Any]) -> float:
        dt = snapshot_ts(s)
        return dt.timestamp() if dt else 0.0

    ordered = sorted(snapshots, key=_key, reverse=True)
    if not prefer_non_empty:
        return ordered[0]
    for s in ordered:
        if snapshot_has_content(s):
            return s
    return ordered[0]

