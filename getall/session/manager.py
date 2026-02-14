"""Session management for conversation history."""

import json
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from loguru import logger

from getall.utils.helpers import ensure_dir, safe_filename


# ── Cross-chat handoff persistence ─────────────────────────────────
#
# When the agent proactively DMs a user from a group chat, the sent
# message must appear in the *target* private session so the user's
# reply has context.  We persist a lightweight handoff file keyed by
# channel + target chat_id.  The next incoming message on that chat
# consumes the handoff and injects it into the session.

_HANDOFF_DIR = "handoffs"
_HANDOFF_TTL = timedelta(hours=24)


def _handoff_dir() -> Path:
    d = Path.home() / ".getall" / "data" / _HANDOFF_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def _handoff_path(channel: str, chat_id: str) -> Path:
    safe = safe_filename(f"{channel}_{chat_id}")
    return _handoff_dir() / f"{safe}.json"


def save_handoff(
    channel: str,
    target_chat_id: str,
    content: str,
    source_channel: str,
    source_chat_id: str,
) -> None:
    """Persist a cross-chat handoff message for later injection.

    Called by MessageTool when sending to a chat_id that differs from
    the current conversation context (e.g. a proactive DM from group).
    """
    path = _handoff_path(channel, target_chat_id)
    data: dict[str, Any] = {}
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass

    messages: list[dict[str, Any]] = data.get("messages", [])
    messages.append({
        "role": "assistant",
        "content": content,
        "timestamp": datetime.now().isoformat(),
    })
    data = {
        "messages": messages,
        "source_channel": source_channel,
        "source_chat_id": source_chat_id,
        "updated_at": datetime.now().isoformat(),
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.debug(
        f"Handoff saved: {channel}:{target_chat_id} "
        f"(from {source_channel}:{source_chat_id}, {len(messages)} msg(s))"
    )


def consume_handoff(channel: str, chat_id: str) -> list[dict[str, Any]] | None:
    """Consume a pending handoff for the given chat.

    Returns the list of assistant messages to inject, or None if no
    handoff exists (or it has expired).
    """
    path = _handoff_path(channel, chat_id)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        path.unlink(missing_ok=True)

        # TTL check — ignore stale handoffs
        updated = data.get("updated_at", "")
        if updated:
            age = datetime.now() - datetime.fromisoformat(updated)
            if age > _HANDOFF_TTL:
                logger.debug(f"Discarding stale handoff for {channel}:{chat_id} (age={age})")
                return None

        messages = data.get("messages", [])
        if messages:
            logger.info(
                f"Consuming handoff for {channel}:{chat_id} "
                f"({len(messages)} msg(s) from {data.get('source_channel')}:{data.get('source_chat_id')})"
            )
        return messages or None
    except Exception as exc:
        logger.warning(f"Failed to consume handoff for {channel}:{chat_id}: {exc}")
        path.unlink(missing_ok=True)
        return None


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

    @staticmethod
    def _format_history_message(message: dict[str, Any]) -> dict[str, str]:
        """Format a stored message for LLM history consumption."""
        role = str(message.get("role", "user"))
        content = str(message.get("content", ""))
        if role != "user" or str(message.get("chat_type", "")) != "group":
            return {"role": role, "content": content}
        sender_name = str(
            message.get("sender_name") or message.get("sender_id") or ""
        ).strip()
        if not sender_name:
            return {"role": role, "content": content}
        return {"role": role, "content": f"[{sender_name}] {content}"}

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
        history = [self._format_history_message(m) for m in recent]

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
