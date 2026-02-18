"""Event engine — load YAML configs, evaluate triggers, track activations.

Supports trigger types:
- date:      Active for the entire schedule window.
- recurring: Active on specific weekdays / hours (e.g. every Friday night).
- random:    Probability-based per message, with cooldown.
- keyword:   Activated when user message contains specific keywords.
- game:      Agent proposes a mini-game; persona shifts on win.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class EventTrigger:
    """Trigger configuration for an event."""

    type: str = "date"  # date | recurring | random | keyword | game

    # -- random --
    probability: float = 0.0
    cooldown_minutes: int = 60

    # -- keyword --
    keywords: list[str] = field(default_factory=list)

    # -- keyword & random: how long the overlay stays active --
    duration_minutes: int = 0

    # -- recurring --
    weekdays: list[int] = field(default_factory=list)  # 0=Mon … 6=Sun
    start_hour: int = 0
    end_hour: int = 24

    # -- game --
    game_prompt: str = ""


@dataclass
class EventConfig:
    """A single event definition loaded from YAML."""

    id: str
    name: str
    schedule_start: datetime | None = None
    schedule_end: datetime | None = None
    trigger: EventTrigger = field(default_factory=EventTrigger)
    persona_overlay: str = ""
    card_theme: str = ""  # Lark card header color override
    announcement: str = ""  # First-trigger announcement text
    game_prompt: str = ""  # Game rules (injected before activation)
    brief: str = ""  # Short summary injected into system prompt (progressive disclosure)
    source_file: str = ""  # YAML filename for read_file reference


@dataclass
class ActiveEvent:
    """An event that is active for the current message."""

    config: EventConfig
    show_announcement: bool = False
    is_game_phase: bool = False  # True → show game_prompt instead of overlay


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------

_STATE_FILE = ".state.json"


class EventManager:
    """Load events from ``workspace/events/*.yaml``, evaluate triggers, track state."""

    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace
        self.events_dir = workspace / "events"
        self._events: dict[str, EventConfig] = {}
        self._state: dict[str, Any] = {}
        self._events_mtime: float = 0

    # ── YAML loading (hot-reload on mtime change) ────────────────────────

    def _reload_events(self) -> None:
        if not self.events_dir.is_dir():
            return
        try:
            dir_mtime = max(
                (f.stat().st_mtime for f in self.events_dir.glob("*.yaml")),
                default=0,
            )
        except OSError:
            return
        if dir_mtime <= self._events_mtime and self._events:
            return
        self._events.clear()
        for path in sorted(self.events_dir.glob("*.yaml")):
            try:
                raw = yaml.safe_load(path.read_text(encoding="utf-8"))
                if not isinstance(raw, dict):
                    continue
                ev = self._parse_event(raw, source_file=path.name)
                if ev:
                    self._events[ev.id] = ev
            except Exception as exc:
                logger.warning(f"Failed to load event {path.name}: {exc}")
        self._events_mtime = dir_mtime
        logger.debug(f"Loaded {len(self._events)} event(s) from {self.events_dir}")

    @staticmethod
    def _parse_event(
        raw: dict[str, Any],
        source_file: str = "",
    ) -> EventConfig | None:
        event_id = raw.get("id")
        if not event_id:
            return None

        # ── timezone ──
        sched = raw.get("schedule") or {}
        tz_name = sched.get("timezone", "Asia/Shanghai")
        try:
            from zoneinfo import ZoneInfo

            tz = ZoneInfo(tz_name)
        except Exception:
            tz = timezone(timedelta(hours=8))

        def _parse_dt(val: str | None) -> datetime | None:
            if not val:
                return None
            return datetime.strptime(val, "%Y-%m-%d %H:%M").replace(tzinfo=tz)

        # ── trigger ──
        traw = raw.get("trigger") or {}
        trigger = EventTrigger(
            type=traw.get("type", "date"),
            probability=float(traw.get("probability", 0)),
            cooldown_minutes=int(traw.get("cooldown_minutes", 60)),
            keywords=[k.lower() for k in (traw.get("keywords") or [])],
            duration_minutes=int(traw.get("duration_minutes", 0)),
            weekdays=traw.get("weekdays") or [],
            start_hour=int(traw.get("start_hour", 0)),
            end_hour=int(traw.get("end_hour", 24)),
            game_prompt=traw.get("game_prompt", ""),
        )

        return EventConfig(
            id=event_id,
            name=raw.get("name", event_id),
            schedule_start=_parse_dt(sched.get("start")),
            schedule_end=_parse_dt(sched.get("end")),
            trigger=trigger,
            persona_overlay=raw.get("persona_overlay", ""),
            card_theme=raw.get("card_theme", ""),
            announcement=raw.get("announcement", ""),
            game_prompt=raw.get("game_prompt", trigger.game_prompt),
            brief=raw.get("brief", ""),
            source_file=source_file,
        )

    # ── state persistence ────────────────────────────────────────────────

    def _load_state(self) -> None:
        fp = self.events_dir / _STATE_FILE
        if fp.is_file():
            try:
                self._state = json.loads(fp.read_text(encoding="utf-8"))
            except Exception:
                self._state = {}
        else:
            self._state = {}

    def _save_state(self) -> None:
        self.events_dir.mkdir(parents=True, exist_ok=True)
        fp = self.events_dir / _STATE_FILE
        fp.write_text(
            json.dumps(self._state, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

    @staticmethod
    def _user_key(user_id: str, chat_id: str) -> str:
        return f"{user_id}:{chat_id}"

    # ── trigger evaluation ───────────────────────────────────────────────

    def _in_schedule(self, ev: EventConfig, now: datetime) -> bool:
        if ev.schedule_start and now < ev.schedule_start:
            return False
        if ev.schedule_end and now > ev.schedule_end:
            return False
        return True

    def _check_trigger(
        self,
        ev: EventConfig,
        now: datetime,
        message: str,
        user_key: str,
    ) -> bool:
        t = ev.trigger

        if t.type == "date":
            return True

        if t.type == "recurring":
            if t.weekdays and now.weekday() not in t.weekdays:
                return False
            return t.start_hour <= now.hour < t.end_hour

        if t.type == "random":
            ud = self._state.get("activations", {}).get(ev.id, {}).get(user_key, {})
            # still in active window?
            if _still_active(ud, now):
                return True
            # in cooldown?
            if _in_cooldown(ud, now):
                return False
            if random.random() < t.probability:
                self._record_activation(ev.id, user_key, now, t.duration_minutes, t.cooldown_minutes)
                return True
            return False

        if t.type == "keyword":
            msg_lower = message.lower()
            if any(kw in msg_lower for kw in t.keywords):
                self._record_activation(ev.id, user_key, now, t.duration_minutes, 0)
                return True
            ud = self._state.get("activations", {}).get(ev.id, {}).get(user_key, {})
            return _still_active(ud, now)

        if t.type == "game":
            return True  # always show game prompt during schedule

        return False

    def _record_activation(
        self,
        event_id: str,
        user_key: str,
        now: datetime,
        duration_minutes: int,
        cooldown_minutes: int,
    ) -> None:
        acts = self._state.setdefault("activations", {})
        ev_data = acts.setdefault(event_id, {})
        ud = ev_data.setdefault(user_key, {})
        ud["activated_at"] = now.isoformat()
        if duration_minutes > 0:
            ud["expires_at"] = (now + timedelta(minutes=duration_minutes)).isoformat()
        if cooldown_minutes > 0:
            ud["cooldown_until"] = (now + timedelta(minutes=cooldown_minutes)).isoformat()
        self._save_state()

    def _is_announced(self, event_id: str, user_key: str) -> bool:
        return bool(
            self._state.get("announced", {}).get(event_id, {}).get(user_key)
        )

    def _mark_announced(self, event_id: str, user_key: str) -> None:
        ann = self._state.setdefault("announced", {})
        ann.setdefault(event_id, {})[user_key] = True
        self._save_state()

    # ── public API ───────────────────────────────────────────────────────

    def get_active_events(
        self,
        message: str = "",
        user_id: str = "",
        chat_id: str = "",
    ) -> list[ActiveEvent]:
        """Return events active for this message context."""
        self._reload_events()
        self._load_state()
        if not self._events:
            return []

        try:
            from zoneinfo import ZoneInfo

            now = datetime.now(ZoneInfo("Asia/Shanghai"))
        except Exception:
            now = datetime.now(timezone(timedelta(hours=8)))

        ukey = self._user_key(user_id, chat_id)
        result: list[ActiveEvent] = []

        for ev in self._events.values():
            if not self._in_schedule(ev, now):
                continue
            if not self._check_trigger(ev, now, message, ukey):
                continue

            show_ann = False
            if ev.announcement and not self._is_announced(ev.id, ukey):
                show_ann = True
                self._mark_announced(ev.id, ukey)

            result.append(
                ActiveEvent(
                    config=ev,
                    show_announcement=show_ann,
                    is_game_phase=(ev.trigger.type == "game"),
                )
            )

        return result

    def build_overlay_text(self, events: list[ActiveEvent]) -> str:
        """Build combined persona overlay text for system prompt injection.

        Uses progressive disclosure: if an event defines a ``brief``, only the
        short summary is injected into the system prompt.  The full creative
        toolkit stays in the YAML file and the agent can ``read_file`` it
        on-demand.  This keeps the context window lean for data-fetching tasks
        while still enabling rich creative behaviour when needed.
        """
        parts: list[str] = []
        for ae in events:
            ev = ae.config
            file_hint = (
                f"\n\n💡 Full creative toolkit: `read_file(\"events/{ev.source_file}\")`"
                if ev.source_file
                else ""
            )

            if ae.is_game_phase:
                if ev.brief:
                    parts.append(
                        f"### 🎮 隐藏事件待解锁：{ev.name}\n\n{ev.brief}{file_hint}"
                    )
                elif ev.game_prompt:
                    # Fallback: no brief → inject full game_prompt (legacy)
                    parts.append(
                        f"### 🎮 隐藏事件待解锁：{ev.name}\n\n{ev.game_prompt}"
                    )
            else:
                if ev.brief:
                    parts.append(
                        f"### 🎭 限时事件已激活：{ev.name}\n\n{ev.brief}{file_hint}"
                    )
                elif ev.persona_overlay:
                    # Fallback: no brief → inject full overlay (legacy)
                    parts.append(
                        f"### 🎭 限时事件已激活：{ev.name}\n\n{ev.persona_overlay}"
                    )
        return "\n\n".join(parts)

    def build_announcement(self, events: list[ActiveEvent]) -> str:
        """Build first-trigger announcement (prepended to agent reply)."""
        parts = [ae.config.announcement for ae in events if ae.show_announcement and ae.config.announcement]
        return "\n\n".join(parts)

    def get_card_theme(self, events: list[ActiveEvent]) -> str:
        """Return the card theme from the highest-priority active event (non-game)."""
        for ae in events:
            if ae.config.card_theme and not ae.is_game_phase:
                return ae.config.card_theme
        return ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _still_active(user_data: dict[str, Any], now: datetime) -> bool:
    exp = user_data.get("expires_at")
    if not exp:
        return False
    try:
        return now < datetime.fromisoformat(exp)
    except (ValueError, TypeError):
        return False


def _in_cooldown(user_data: dict[str, Any], now: datetime) -> bool:
    cd = user_data.get("cooldown_until")
    if not cd:
        return False
    try:
        return now < datetime.fromisoformat(cd)
    except (ValueError, TypeError):
        return False
