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

    @property
    def is_group_scope(self) -> bool:
        return self.scope.startswith("group")

    def read_recent_history(
        self,
        max_entries: int | None = None,
        max_chars: int | None = None,
    ) -> str:
        """Return the most recent consolidation entries from HISTORY.md.

        Each entry is separated by a blank line.  We read from the end
        of the file and return up to *max_entries* entries (capped at
        *max_chars* total characters) so the agent has actionable
        context about recent conversations without needing to grep.

        Group scopes get significantly larger defaults so the agent
        retains rich per-user context across many speakers.
        """
        if max_entries is None:
            max_entries = 12 if self.is_group_scope else 5
        if max_chars is None:
            max_chars = 8000 if self.is_group_scope else 3000
        if not self.history_file.exists():
            return ""
        try:
            raw = self.history_file.read_text(encoding="utf-8").rstrip()
        except Exception:
            return ""
        if not raw:
            return ""

        # Entries are separated by double-newline
        entries = [e.strip() for e in raw.split("\n\n") if e.strip()]
        if not entries:
            return ""

        recent = entries[-max_entries:]
        # Trim to max_chars from the *end* (newest first)
        result_parts: list[str] = []
        total = 0
        for entry in reversed(recent):
            if total + len(entry) > max_chars:
                break
            result_parts.append(entry)
            total += len(entry)
        result_parts.reverse()
        return "\n\n".join(result_parts)

    def get_memory_context(self) -> str:
        parts: list[str] = []

        long_term = self.read_long_term()
        if long_term:
            parts.append(f"## Long-term Memory\n{long_term}")

        recent_history = self.read_recent_history()
        if recent_history:
            parts.append(
                f"## Recent Conversation History (auto-loaded)\n"
                f"These are the most recent conversation summaries. "
                f"For older history, grep the HISTORY.md file.\n\n"
                f"{recent_history}"
            )

        return "\n\n".join(parts)
