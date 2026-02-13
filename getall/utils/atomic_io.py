"""Atomic file I/O utilities to prevent data corruption from concurrent writes.

Uses file-level asyncio locks + temp-rename pattern to guarantee atomicity.
"""

import asyncio
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from loguru import logger


class AtomicFileWriter:
    """Thread-safe and async-safe atomic file writer.

    Prevents concurrent writes via per-path asyncio locks and uses atomic
    rename (write-to-temp then ``Path.replace``) to avoid partial writes.

    Usage::

        writer = AtomicFileWriter()
        await writer.write_json(path, data)
        await writer.append_line(path, line)
    """

    def __init__(self) -> None:
        self._locks: dict[Path, asyncio.Lock] = {}

    def _get_lock(self, path: Path) -> asyncio.Lock:
        """Get or create a lock for *path* (resolved to canonical form)."""
        resolved = path.resolve()
        if resolved not in self._locks:
            self._locks[resolved] = asyncio.Lock()
        return self._locks[resolved]

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    async def write_text(self, path: Path, content: str, encoding: str = "utf-8") -> bool:
        """Atomically write *content* to *path*.

        Returns ``True`` on success, ``False`` on failure.
        """
        lock = self._get_lock(path)
        async with lock:
            temp_path: str | None = None
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                fd, temp_path = tempfile.mkstemp(
                    dir=path.parent,
                    prefix=f".{path.name}.",
                    suffix=".tmp",
                )
                try:
                    os.write(fd, content.encode(encoding))
                finally:
                    os.close(fd)

                Path(temp_path).replace(path)
                return True
            except OSError as exc:
                logger.error(f"Atomic write failed for {path}: {exc}")
                if temp_path:
                    try:
                        Path(temp_path).unlink(missing_ok=True)
                    except Exception:
                        pass
                return False

    async def write_json(
        self,
        path: Path,
        data: Any,
        indent: int = 2,
        ensure_ascii: bool = False,
    ) -> bool:
        """Atomically serialize *data* as JSON and write to *path*."""
        try:
            content = json.dumps(data, indent=indent, ensure_ascii=ensure_ascii)
        except (TypeError, ValueError) as exc:
            logger.error(f"JSON serialization failed for {path}: {exc}")
            return False
        return await self.write_text(path, content)

    async def append_line(self, path: Path, line: str, encoding: str = "utf-8") -> bool:
        """Append a single *line* with locking (not fully atomic, but serialized)."""
        lock = self._get_lock(path)
        async with lock:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                if not line.endswith("\n"):
                    line += "\n"
                with path.open("a", encoding=encoding) as fh:
                    fh.write(line)
                return True
            except OSError as exc:
                logger.error(f"Append failed for {path}: {exc}")
                return False

    async def append_json_line(self, path: Path, data: Any) -> bool:
        """Append *data* as a single JSON line to a ``.jsonl`` file."""
        try:
            line = json.dumps(data, ensure_ascii=False)
        except (TypeError, ValueError) as exc:
            logger.error(f"JSON serialization failed for append to {path}: {exc}")
            return False
        return await self.append_line(path, line)

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def cleanup_stale_locks(self, max_locks: int = 100) -> None:
        """Evict lock references to bound memory in long-running processes."""
        if len(self._locks) > max_locks:
            old_count = len(self._locks)
            self._locks.clear()
            logger.debug(f"Cleared {old_count} stale file locks")


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_atomic_writer: AtomicFileWriter | None = None


def get_atomic_writer() -> AtomicFileWriter:
    """Return the global :class:`AtomicFileWriter` singleton."""
    global _atomic_writer
    if _atomic_writer is None:
        _atomic_writer = AtomicFileWriter()
    return _atomic_writer
