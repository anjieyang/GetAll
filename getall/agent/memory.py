"""Memory system for persistent agent memory."""

from pathlib import Path

from getall.utils.helpers import ensure_dir, safe_filename


class MemoryStore:
    """Two-layer memory: MEMORY.md (long-term facts) + HISTORY.md (grep-searchable log)."""

    _SOUL_OVERLAY_TEMPLATE = """# Scoped SOUL Overlay

This file stores scope-specific behavior adjustments for this principal/group.

Rules:
- Additive only: refine tone, preferences, and workflow hints for this scope.
- Never weaken or contradict global workspace SOUL hard rules.
- If uncertain, keep this file minimal.
"""

    def __init__(self, workspace: Path, scope: str = "global"):
        self.scope = (scope or "global").strip() or "global"
        if self.scope == "global":
            self.memory_dir = ensure_dir(workspace / "memory")
        else:
            safe_scope = safe_filename(self.scope.replace("/", "__"))
            self.memory_dir = ensure_dir(workspace / "memory" / "scopes" / safe_scope)
        self.memory_file = self.memory_dir / "MEMORY.md"
        self.history_file = self.memory_dir / "HISTORY.md"
        self.soul_overlay_file = self.memory_dir / "SOUL.md"

    @property
    def has_scoped_overlay(self) -> bool:
        """Return True when this scope supports a dedicated SOUL overlay."""
        return self.scope != "global"

    def ensure_soul_overlay(self) -> Path | None:
        """Create scoped SOUL overlay file if missing."""
        if not self.has_scoped_overlay:
            return None
        if not self.soul_overlay_file.exists():
            self.soul_overlay_file.write_text(
                self._SOUL_OVERLAY_TEMPLATE,
                encoding="utf-8",
            )
        return self.soul_overlay_file

    def read_soul_overlay(self, create_if_missing: bool = False) -> str:
        """Read scoped SOUL overlay content."""
        if not self.has_scoped_overlay:
            return ""
        if create_if_missing:
            self.ensure_soul_overlay()
        if self.soul_overlay_file.exists():
            return self.soul_overlay_file.read_text(encoding="utf-8")
        return ""

    def read_long_term(self) -> str:
        if self.memory_file.exists():
            return self.memory_file.read_text(encoding="utf-8")
        return ""

    def write_long_term(self, content: str) -> None:
        self.memory_file.write_text(content, encoding="utf-8")

    def append_history(self, entry: str) -> None:
        with open(self.history_file, "a", encoding="utf-8") as f:
            f.write(entry.rstrip() + "\n\n")

    def get_memory_context(self) -> str:
        long_term = self.read_long_term()
        return f"## Long-term Memory\n{long_term}" if long_term else ""
