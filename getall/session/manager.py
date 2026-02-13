"""Session management for conversation history."""

import json
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from loguru import logger

from getall.utils.helpers import ensure_dir, safe_filename


@dataclass
class Session:
    """A conversation session.

    Stores messages in JSONL format for easy reading and persistence.
    Also tracks recent tool executions for context awareness.
    """

    key: str  # channel:chat_id
    messages: list[dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)
    tool_executions: list[dict[str, Any]] = field(default_factory=list)

    def add_message(self, role: str, content: str, **kwargs: Any) -> None:
        """Add a message to the session."""
        msg = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            **kwargs,
        }
        self.messages.append(msg)
        self.updated_at = datetime.now()

    def add_tool_execution(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        result_summary: str,
    ) -> None:
        """Record a tool execution for context awareness.

        Args:
            tool_name: Name of the tool executed.
            arguments: Tool arguments dict.
            result_summary: Brief summary of result (truncated to 200 chars).
        """
        self.tool_executions.append({
            "tool": tool_name,
            "args": arguments,
            "result": result_summary[:200],
            "timestamp": datetime.now().isoformat(),
        })
        # Keep only last 20 to prevent bloat
        if len(self.tool_executions) > 20:
            self.tool_executions = self.tool_executions[-20:]
        self.updated_at = datetime.now()

    def get_history(self, max_messages: int = 50) -> list[dict[str, Any]]:
        """Get message history for LLM context.

        Args:
            max_messages: Maximum messages to return.

        Returns:
            List of messages in LLM format, optionally prefixed with
            recent tool execution context.
        """
        recent = self.messages[-max_messages:] if len(self.messages) > max_messages else self.messages
        history = [{"role": m["role"], "content": m["content"]} for m in recent]

        # Prepend tool-execution context if available
        if self.tool_executions:
            lines = ["## Recent Tool Executions (context)"]
            for te in self.tool_executions[-10:]:
                tool = te.get("tool", "?")
                args = te.get("args", {})
                result = te.get("result", "")
                ts = te.get("timestamp", "")
                # Surface the most useful argument keys
                key_parts = [f"{k}={v}" for k, v in args.items()
                             if k in {"symbol", "side", "amount", "action", "dry_run", "paper_trade"}]
                args_str = ", ".join(key_parts) if key_parts else str(args)
                lines.append(f"- [{ts}] {tool}({args_str}) -> {result}")
            history.insert(0, {"role": "system", "content": "\n".join(lines)})

        return history

    def clear(self) -> None:
        """Clear all messages in the session."""
        self.messages = []
        self.tool_executions = []
        self.updated_at = datetime.now()


class SessionManager:
    """
    Manages conversation sessions.
    
    Sessions are stored as JSONL files in the sessions directory.
    """
    
    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.sessions_dir = ensure_dir(Path.home() / ".getall" / "sessions")
        self._cache: dict[str, Session] = {}
    
    def _get_session_path(self, key: str) -> Path:
        """Get the file path for a session."""
        safe_key = safe_filename(key.replace(":", "_"))
        return self.sessions_dir / f"{safe_key}.jsonl"
    
    def get_or_create(self, key: str) -> Session:
        """
        Get an existing session or create a new one.
        
        Args:
            key: Session key (usually channel:chat_id).
        
        Returns:
            The session.
        """
        # Check cache
        if key in self._cache:
            return self._cache[key]
        
        # Try to load from disk
        session = self._load(key)
        if session is None:
            session = Session(key=key)
        
        self._cache[key] = session
        return session
    
    def _load(self, key: str) -> Session | None:
        """Load a session from disk."""
        path = self._get_session_path(key)
        
        if not path.exists():
            return None
        
        try:
            messages = []
            metadata = {}
            tool_executions: list[dict[str, Any]] = []
            created_at = None

            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    if data.get("_type") == "metadata":
                        metadata = data.get("metadata", {})
                        created_at = datetime.fromisoformat(data["created_at"]) if data.get("created_at") else None
                        tool_executions = data.get("tool_executions", [])
                    else:
                        messages.append(data)

            return Session(
                key=key,
                messages=messages,
                created_at=created_at or datetime.now(),
                metadata=metadata,
                tool_executions=tool_executions,
            )
        except Exception as e:
            logger.warning(f"Failed to load session {key}: {e}")
            return None
    
    def save(self, session: Session) -> None:
        """Save a session to disk."""
        path = self._get_session_path(session.key)
        
        with open(path, "w") as f:
            # Write metadata first (includes tool_executions)
            metadata_line = {
                "_type": "metadata",
                "created_at": session.created_at.isoformat(),
                "updated_at": session.updated_at.isoformat(),
                "metadata": session.metadata,
                "tool_executions": session.tool_executions,
            }
            f.write(json.dumps(metadata_line) + "\n")

            for msg in session.messages:
                f.write(json.dumps(msg) + "\n")
        
        self._cache[session.key] = session
    
    def delete(self, key: str) -> bool:
        """
        Delete a session.
        
        Args:
            key: Session key.
        
        Returns:
            True if deleted, False if not found.
        """
        # Remove from cache
        self._cache.pop(key, None)
        
        # Remove file
        path = self._get_session_path(key)
        if path.exists():
            path.unlink()
            return True
        return False
    
    def list_sessions(self) -> list[dict[str, Any]]:
        """
        List all sessions.
        
        Returns:
            List of session info dicts.
        """
        sessions = []
        
        for path in self.sessions_dir.glob("*.jsonl"):
            try:
                # Read just the metadata line
                with open(path) as f:
                    first_line = f.readline().strip()
                    if first_line:
                        data = json.loads(first_line)
                        if data.get("_type") == "metadata":
                            sessions.append({
                                "key": path.stem.replace("_", ":"),
                                "created_at": data.get("created_at"),
                                "updated_at": data.get("updated_at"),
                                "path": str(path)
                            })
            except Exception:
                continue
        
        return sorted(sessions, key=lambda x: x.get("updated_at", ""), reverse=True)
