"""Memory system for persistent agent memory."""

from pathlib import Path

from getall.utils.helpers import ensure_dir, safe_filename


class MemoryStore:
    """Two-layer memory: MEMORY.md (long-term facts) + HISTORY.md (grep-searchable log)."""

    def __init__(self, workspace: Path, scope: str = "global"):
        self.scope = (scope or "global").strip() or "global"
        if self.scope == "global":
            self.memory_dir = ensure_dir(workspace / "memory")
        else:
            safe_scope = safe_filename(self.scope.replace("/", "__"))
            self.memory_dir = ensure_dir(workspace / "memory" / "scopes" / safe_scope)
        self.memory_file = self.memory_dir / "MEMORY.md"
        self.history_file = self.memory_dir / "HISTORY.md"

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
