"""简单缓存层: L1 内存 (TTL), L2 文件缓存."""

import json
import time
from pathlib import Path
from typing import Any

from loguru import logger


class DataCache:
    """
    两级缓存实现.

    L1: 内存字典, 带 TTL 过期
    L2: 文件系统持久化 (可选)
    """

    def __init__(self, cache_dir: Path | None = None):
        """
        Initialize the cache.

        Args:
            cache_dir: Optional directory for L2 file cache. Disabled if None.
        """
        # L1: 内存缓存 {key: (value, expire_timestamp)}
        self._mem: dict[str, tuple[Any, float]] = {}
        # L2: 文件缓存目录
        self._cache_dir = cache_dir
        if self._cache_dir:
            self._cache_dir.mkdir(parents=True, exist_ok=True)

    def get(self, key: str) -> Any | None:
        """
        Get a value from cache (L1 -> L2 fallback).

        Args:
            key: Cache key.

        Returns:
            Cached value or None if not found / expired.
        """
        # L1: 内存查找
        if key in self._mem:
            value, expire_at = self._mem[key]
            if time.time() < expire_at:
                return value
            # 已过期, 清除
            del self._mem[key]

        # L2: 文件查找
        if self._cache_dir:
            path = self._cache_dir / f"{_safe_key(key)}.json"
            if path.exists():
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    if time.time() < data.get("expire_at", 0):
                        # 回填 L1
                        self._mem[key] = (data["value"], data["expire_at"])
                        return data["value"]
                    # 文件过期, 删除
                    path.unlink(missing_ok=True)
                except (json.JSONDecodeError, KeyError):
                    logger.warning(f"Corrupt cache file removed: {path}")
                    path.unlink(missing_ok=True)

        return None

    def set(self, key: str, value: Any, ttl_seconds: int = 300) -> None:
        """
        Set a value in cache.

        Args:
            key: Cache key.
            value: Value to cache (must be JSON-serializable for L2).
            ttl_seconds: Time-to-live in seconds, default 5 minutes.
        """
        expire_at = time.time() + ttl_seconds

        # L1: 写入内存
        self._mem[key] = (value, expire_at)

        # L2: 写入文件
        if self._cache_dir:
            path = self._cache_dir / f"{_safe_key(key)}.json"
            try:
                path.write_text(
                    json.dumps({"value": value, "expire_at": expire_at}, ensure_ascii=False),
                    encoding="utf-8",
                )
            except (TypeError, OSError) as e:
                logger.debug(f"L2 cache write skipped for '{key}': {e}")

    def invalidate(self, key: str) -> None:
        """
        Remove a key from all cache levels.

        Args:
            key: Cache key to invalidate.
        """
        self._mem.pop(key, None)
        if self._cache_dir:
            path = self._cache_dir / f"{_safe_key(key)}.json"
            path.unlink(missing_ok=True)

    def clear(self) -> None:
        """清空全部缓存."""
        self._mem.clear()
        if self._cache_dir:
            for p in self._cache_dir.glob("*.json"):
                p.unlink(missing_ok=True)

    @property
    def size(self) -> int:
        """返回 L1 缓存条目数 (含已过期但未清理的)."""
        return len(self._mem)


def _safe_key(key: str) -> str:
    """将缓存 key 转为安全文件名."""
    return key.replace("/", "_").replace(":", "_").replace(" ", "_")[:128]
