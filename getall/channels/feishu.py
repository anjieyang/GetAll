"""Feishu/Lark channel – WebSocket long-connection + HTTP webhook support.

Supports:
- Private chat (p2p): user talks to bot directly
- Group chat: reply only when bot is @mentioned
- Auto @mention sender in group replies
- Resolve @name references in LLM output to real Lark <at> tags
"""

import asyncio
import datetime
import json
import platform
import re
import threading
import time as _time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger

from getall.bus.events import OutboundMessage
from getall.bus.queue import MessageBus
from getall.channels.base import BaseChannel
from getall.channels.feishu_emojis import FEISHU_REACTION_EMOJI_CHOICES
from getall.config.schema import FeishuConfig

try:
    import lark_oapi as lark
    from lark_oapi.api.im.v1 import (
        CreateMessageRequest,
        CreateMessageRequestBody,
        CreateMessageReactionRequest,
        CreateMessageReactionRequestBody,
        CreateFileRequest,
        CreateFileRequestBody,
        GetMessageResourceRequest,
        ListChatRequest,
        Emoji,
        GetChatMembersRequest,
        P2ImMessageReceiveV1,
    )

    FEISHU_AVAILABLE = True
except ImportError:
    FEISHU_AVAILABLE = False
    lark = None  # type: ignore[assignment]
    Emoji = None

# Non-text message type labels
_MSG_TYPE_MAP = {
    "image": "[image]",
    "audio": "[audio]",
    "file": "[file]",
    "sticker": "[sticker]",
}

_MEMBER_CACHE_TTL = 300  # 5 minutes
_SENT_MSG_ID_CACHE_MAX = 2000
_MAX_CARD_TABLES = 5  # Lark card: max native table components per card


@dataclass
class _RenderPlan:
    """Structured presentation decision for one outbound Feishu message."""

    mode: str = "text"  # "text" | "interactive"
    mention_sender: bool = False
    title: str = ""
    body: str = ""
    confidence: float = 0.0
    rationale: str = ""


def _split_into_chunks(text: str) -> list[str]:
    """Split a response into human-sized message chunks.

    Rules:
    - Short responses (≤ 500 chars) always stay as one message.
    - Longer responses are split on paragraph boundaries (double newline).
    - Single-paragraph responses stay as one message.
    """
    stripped = text.strip()
    if not stripped:
        return []

    # Short responses → single message to avoid fragmented bubbles
    if len(stripped) <= 500:
        return [stripped]

    # Split on paragraph boundaries for longer text
    paragraphs = re.split(r"\n{2,}", stripped)
    chunks = [p.strip() for p in paragraphs if p.strip()]

    if len(chunks) <= 1:
        return [stripped]

    return chunks


# ---------------------------------------------------------------------------
# Group member cache – for @mention resolution
# ---------------------------------------------------------------------------

class _GroupMemberCache:
    """In-memory cache: chat_id → {name_lower: open_id} + {open_id: name}."""

    def __init__(self) -> None:
        self._by_name: dict[str, dict[str, str]] = {}
        self._by_id: dict[str, dict[str, str]] = {}
        self._ts: dict[str, float] = {}

    @staticmethod
    def _looks_like_open_id(value: str) -> bool:
        v = value.strip()
        return v.startswith("ou_") and len(v) >= 12

    def is_stale(self, chat_id: str) -> bool:
        return (_time.time() - self._ts.get(chat_id, 0)) > _MEMBER_CACHE_TTL

    def set_members(
        self,
        chat_id: str,
        by_name: dict[str, str],
        by_id: dict[str, str],
    ) -> None:
        self._by_name[chat_id] = by_name
        self._by_id[chat_id] = by_id
        self._ts[chat_id] = _time.time()

    def resolve_name(self, chat_id: str, name: str) -> str | None:
        """name (case-insensitive) → open_id or None."""
        return self._by_name.get(chat_id, {}).get(name.lower())

    def display_name(self, chat_id: str, open_id: str) -> str:
        return self._by_id.get(chat_id, {}).get(open_id, open_id)

    def track_sender(self, chat_id: str, open_id: str, name: str) -> None:
        """Incrementally learn a member from an incoming message."""
        clean_name = (name or "").strip()
        if not clean_name:
            return
        # Ignore fallback IDs (e.g. "ou_xxx") so we don't overwrite real names.
        if clean_name == open_id or self._looks_like_open_id(clean_name):
            return
        self._by_name.setdefault(chat_id, {})[clean_name.lower()] = open_id
        self._by_id.setdefault(chat_id, {})[open_id] = clean_name

    def all_names(self, chat_id: str) -> list[str]:
        """Return all known display names for a chat (excluding empties)."""
        return [
            n
            for n in self._by_id.get(chat_id, {}).values()
            if n and not self._looks_like_open_id(n)
        ]


# ---------------------------------------------------------------------------
# Channel
# ---------------------------------------------------------------------------

class FeishuChannel(BaseChannel):
    """Feishu/Lark channel using lark-oapi SDK."""

    name = "feishu"

    def __init__(self, config: FeishuConfig, bus: MessageBus) -> None:
        super().__init__(config, bus)
        self.config: FeishuConfig = config
        self._client: Any = None
        self._ws_client: Any = None
        self._ws_thread: threading.Thread | None = None
        self._event_handler: Any = None  # EventDispatcherHandler – shared by WS & webhook
        self._dedup: OrderedDict[str, None] = OrderedDict()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._members = _GroupMemberCache()
        self._bot_open_id: str = ""
        self._sent_message_ids: OrderedDict[str, None] = OrderedDict()
        self._reaction_llm: Any = None
        self._reaction_model: str = ""
        self._reaction_reasoning_effort: str = ""
        self._reaction_llm_init_attempted = False
        self._render_llm: Any = None
        self._render_model: str = ""
        self._render_reasoning_effort: str = ""
        self._render_llm_init_attempted = False

    # ── helpers ──

    def _domain(self) -> str:
        if self.config.domain.lower() == "lark":
            return lark.LARK_DOMAIN  # https://open.larksuite.com
        return lark.FEISHU_DOMAIN  # https://open.feishu.cn

    def _domain_label(self) -> str:
        return "Lark" if self.config.domain.lower() == "lark" else "Feishu"

    # ── bot identity ──

    def _fetch_bot_open_id_sync(self) -> None:
        """Fetch the bot's own open_id via ``GET /open-apis/bot/v3/info``.

        Must be called after ``self._client`` is initialized.
        Sets ``self._bot_open_id`` on success; logs a warning on failure
        so the heuristic fallback can still kick in later.
        """
        if not self._client:
            return
        try:
            from lark_oapi.core.http import HttpMethod
            from lark_oapi.core.model.base_request import BaseRequest

            req = BaseRequest()
            req.uri = "/open-apis/bot/v3/info"
            req.http_method = HttpMethod.GET
            req.token_types = {lark.AccessTokenType.TENANT}

            resp = self._client.request(req)
            if resp.code == 0 and resp.raw and resp.raw.content:
                data = json.loads(resp.raw.content)
                bot_info = data.get("bot", {})
                open_id = bot_info.get("open_id", "")
                if open_id:
                    self._bot_open_id = open_id
                    logger.info(f"Bot open_id from API: {open_id}")
                    return
            logger.warning(
                f"Bot info API returned no open_id: code={resp.code}, "
                f"msg={getattr(resp, 'msg', '')}"
            )
        except Exception as exc:
            logger.warning(f"Failed to fetch bot open_id via API: {exc}")

    # ── lifecycle notifications ──

    def _list_bot_group_chat_ids_sync(self) -> list[str]:
        """List all *group* chat_ids the bot has joined (blocking, paginated).

        Filters out p2p chats — only ``oc_`` prefixed IDs are returned.
        """
        if not self._client:
            return []
        chat_ids: list[str] = []
        page_token: str | None = None
        try:
            while True:
                builder = ListChatRequest.builder().page_size(100)
                if page_token:
                    builder = builder.page_token(page_token)
                resp = self._client.im.v1.chat.list(builder.build())
                if not resp.success():
                    logger.warning(
                        f"List chats failed: code={resp.code}, msg={resp.msg}"
                    )
                    break
                data = resp.data
                if data and data.items:
                    for chat in data.items:
                        cid = chat.chat_id or ""
                        # Only group chats (oc_ prefix); skip p2p / unknown
                        if cid.startswith("oc_"):
                            chat_ids.append(cid)
                if not data or not data.has_more:
                    break
                page_token = data.page_token
        except Exception as exc:
            logger.warning(f"Error listing bot chats: {exc}")
        return chat_ids

    def _resolve_notification_targets_sync(self) -> list[str]:
        """Determine which chat_ids should receive lifecycle notifications.

        - If ``notification_chat_ids`` is configured, use that (explicit override).
        - Otherwise, auto-discover all group chats the bot has joined.
        """
        if self.config.notification_chat_ids:
            return list(self.config.notification_chat_ids)
        return self._list_bot_group_chat_ids_sync()

    # CloudsWay MaaS model IDs → human-readable names
    _MODEL_DISPLAY_NAMES: dict[str, str] = {
        "maas_cl_sonnet_4.5": "Claude Sonnet 4.5",
        "maas_cl_sonnet_4": "Claude Sonnet 4",
        "maas_cl_opus_4": "Claude Opus 4",
        "maas_cl_haiku_3.5": "Claude Haiku 3.5",
        "maas_gpt_4o": "GPT-4o",
        "maas_gpt_4.1": "GPT-4.1",
        "maas_o3": "o3",
        "maas_o4_mini": "o4-mini",
    }

    @classmethod
    def _friendly_model_name(cls, model: str) -> str:
        """Convert internal model IDs to human-readable names."""
        return cls._MODEL_DISPLAY_NAMES.get(model.lower(), model)

    def _build_lifecycle_card(self, event: str) -> str:
        """Build an interactive card for a lifecycle event.

        ``event`` is ``"online"`` or ``"offline"``.
        """
        from getall import __version__
        from getall.config.loader import load_config

        is_online = event == "online"
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        emoji = "🟢" if is_online else "🔴"
        title = f"{emoji} {'已上线' if is_online else '已下线'}"
        template = "green" if is_online else "red"
        status_text = (
            "**运行中** — 随时待命"
            if is_online
            else "**停机中** — 维护更新"
        )
        detail = (
            f"**状态：** {status_text}\n"
            f"**版本：** v{__version__}\n"
            f"**时间：** {now}\n"
            f"**主机：** {platform.node()}"
        )
        if is_online:
            try:
                cfg = load_config()
                model = cfg.agents.defaults.model
                if model:
                    detail += f"\n**模型：** {self._friendly_model_name(model)}"
            except Exception:
                pass
        card: dict[str, Any] = {
            "config": {"wide_screen_mode": True},
            "header": {
                "title": {"content": title, "tag": "plain_text"},
                "template": template,
            },
            "elements": [{"tag": "markdown", "content": detail}],
        }
        return json.dumps(card, ensure_ascii=False)

    def _send_lifecycle_notification_sync(self, event: str) -> None:
        """Send lifecycle card to target chat_ids (blocking).

        Automatically discovers all bot groups unless overridden by config.
        """
        if not self._client:
            return
        targets = self._resolve_notification_targets_sync()
        if not targets:
            logger.debug("No notification targets found, skipping lifecycle card")
            return
        card = self._build_lifecycle_card(event)
        for chat_id in targets:
            try:
                req = (
                    CreateMessageRequest.builder()
                    .receive_id_type("chat_id")
                    .request_body(
                        CreateMessageRequestBody.builder()
                        .receive_id(chat_id)
                        .msg_type("interactive")
                        .content(card)
                        .build()
                    )
                    .build()
                )
                resp = self._client.im.v1.message.create(req)
                if resp.success():
                    logger.info(f"Sent {event} notification to {chat_id}")
                else:
                    logger.warning(
                        f"Lifecycle notification failed ({chat_id}): "
                        f"code={resp.code}, msg={resp.msg}"
                    )
            except Exception as exc:
                logger.warning(f"Error sending lifecycle notification to {chat_id}: {exc}")

    async def send_lifecycle_notification(self, event: str) -> None:
        """Async wrapper — runs blocking send in executor."""
        if not self._client:
            return
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None, self._send_lifecycle_notification_sync, event
        )

    # ── lifecycle ──

    def get_event_handler(self) -> Any:
        """Create / return the EventDispatcherHandler (usable before start)."""
        if not FEISHU_AVAILABLE:
            return None
        if self._event_handler is None:
            # No-op handler for events we don't process but must ACK
            def _noop(data: Any) -> None:
                pass

            self._event_handler = (
                lark.EventDispatcherHandler.builder(
                    self.config.encrypt_key or "",
                    self.config.verification_token or "",
                    lark.LogLevel.WARNING,
                )
                # ── core ──
                .register_p2_im_message_receive_v1(self._on_message_sync)
                # ── chat lifecycle ──
                .register_p2_im_chat_access_event_bot_p2p_chat_entered_v1(_noop)
                .register_p2_im_chat_disbanded_v1(_noop)
                .register_p2_im_chat_updated_v1(_noop)
                # ── member changes ──
                .register_p2_im_chat_member_bot_added_v1(_noop)
                .register_p2_im_chat_member_bot_deleted_v1(_noop)
                .register_p2_im_chat_member_user_added_v1(_noop)
                .register_p2_im_chat_member_user_deleted_v1(_noop)
                .register_p2_im_chat_member_user_withdrawn_v1(_noop)
                # ── message events ──
                .register_p2_im_message_message_read_v1(_noop)
                .register_p2_im_message_reaction_created_v1(_noop)
                .register_p2_im_message_reaction_deleted_v1(_noop)
                .register_p2_im_message_recalled_v1(_noop)
                # ── events without dedicated SDK handler ──
                .register_p2_customized_event(
                    "im.message.at_message_read_v1", _noop
                )
                .register_p2_customized_event(
                    "im.message.updated_v1", _noop
                )
                .build()
            )
        return self._event_handler

    async def start(self) -> None:
        if not FEISHU_AVAILABLE:
            logger.error("lark-oapi not installed. Run: pip install lark-oapi")
            return
        if not self.config.app_id or not self.config.app_secret:
            logger.error(f"{self._domain_label()} app_id / app_secret not configured")
            return

        self._running = True
        self._loop = asyncio.get_running_loop()
        domain = self._domain()
        label = self._domain_label()

        # HTTP client (send messages, call APIs)
        self._client = (
            lark.Client.builder()
            .app_id(self.config.app_id)
            .app_secret(self.config.app_secret)
            .domain(domain)
            .log_level(lark.LogLevel.INFO)
            .build()
        )

        # Fetch bot's own open_id via API (reliable, not heuristic)
        self._fetch_bot_open_id_sync()

        # Event dispatcher (reused by both WS and webhook paths)
        self.get_event_handler()

        if self.config.use_webhook:
            logger.info(f"{label} bot started in HTTP webhook mode")
            logger.info(
                "Set Request URL in developer console → Events & Callbacks "
                "to: https://<host>:<port>/lark/event"
            )
        else:
            self._ws_client = lark.ws.Client(
                self.config.app_id,
                self.config.app_secret,
                event_handler=self._event_handler,
                log_level=lark.LogLevel.INFO,
                domain=domain,
            )

            def _run_ws() -> None:
                while self._running:
                    try:
                        self._ws_client.start()
                    except Exception as exc:
                        logger.warning(f"{label} WebSocket error: {exc}")
                    if self._running:
                        _time.sleep(5)

            self._ws_thread = threading.Thread(target=_run_ws, daemon=True)
            self._ws_thread.start()
            logger.info(f"{label} bot started (WebSocket long connection)")

        # Send "online" notification after client is ready
        try:
            await self.send_lifecycle_notification("online")
        except Exception as exc:
            logger.warning(f"Failed to send online notification: {exc}")

        while self._running:
            await asyncio.sleep(1)

    async def stop(self) -> None:
        if not self._running:
            return
        self._running = False

        # Send "offline" notification before teardown
        try:
            await self.send_lifecycle_notification("offline")
        except Exception as exc:
            logger.warning(f"Failed to send offline notification: {exc}")

        if self._ws_client:
            try:
                self._ws_client.stop()
            except Exception as exc:
                logger.warning(f"Error stopping WS client: {exc}")
        logger.info(f"{self._domain_label()} bot stopped")

    # (bot open_id is learned from incoming mentions – no startup API call needed)

    # ── group member cache ──

    def _fetch_members_sync(self, chat_id: str) -> None:
        if not self._client:
            return
        try:
            by_name: dict[str, str] = {}
            by_id: dict[str, str] = {}
            page_token: str | None = None

            while True:
                builder = (
                    GetChatMembersRequest.builder()
                    .chat_id(chat_id)
                    .member_id_type("open_id")
                    .page_size(100)
                )
                if page_token:
                    builder = builder.page_token(page_token)

                resp = self._client.im.v1.chat_members.get(builder.build())
                if not resp.success():
                    logger.warning(
                        f"Fetch members failed ({chat_id}): "
                        f"code={resp.code}, msg={resp.msg}"
                    )
                    break

                body = resp.data
                if body and body.items:
                    for m in body.items:
                        if m.name and m.member_id:
                            by_name[m.name.lower()] = m.member_id
                            by_id[m.member_id] = m.name

                if not body or not body.has_more:
                    break
                page_token = body.page_token

            if by_name:
                self._members.set_members(chat_id, by_name, by_id)
                logger.debug(f"Cached {len(by_name)} members for {chat_id}")
        except Exception as exc:
            logger.warning(f"Error fetching members: {exc}")

    async def _ensure_members(self, chat_id: str) -> None:
        if self._members.is_stale(chat_id):
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._fetch_members_sync, chat_id)

    # ── @mention resolution ──

    _AT_RE = re.compile(r"@([\w\u4e00-\u9fff]+(?:\s[\w\u4e00-\u9fff]+){0,4})")

    _AT_ALL_NAMES = frozenset({"all", "所有人", "everyone", "全体", "全体成员"})

    def _resolve_at_tags(self, chat_id: str, text: str) -> str:
        """Replace @name in LLM output with Lark <at> markdown tags.

        Handles ``@all`` / ``@所有人`` → ``<at id=all>所有人</at>``.
        For person mentions, tries progressively shorter word sequences
        so ``@Monica Wan 后续文字`` correctly resolves "Monica Wan" and
        leaves "后续文字" untouched.
        """

        def _replace(m: re.Match[str]) -> str:
            full = m.group(1)

            # Try progressively shorter word sequences until we find a match
            words = full.split()
            for n in range(len(words), 0, -1):
                candidate = " ".join(words[:n])
                rest = full[len(candidate):]

                # @all / @所有人 / @everyone
                if candidate.lower() in self._AT_ALL_NAMES:
                    return f"<at id=all>所有人</at>{rest}"

                # Named member
                oid = self._members.resolve_name(chat_id, candidate)
                if oid:
                    return f"<at id={oid}>{candidate}</at>{rest}"

            return m.group(0)

        return self._AT_RE.sub(_replace, text)

    def _prepend_sender_at(
        self, chat_id: str, sender_open_id: str, text: str
    ) -> str:
        name = self._members.display_name(chat_id, sender_open_id)
        return f"<at id={sender_open_id}>{name}</at> {text}"

    def _prepend_sender_plain(
        self, chat_id: str, sender_open_id: str, text: str
    ) -> str:
        """Plain-text @prefix fallback when card rendering is unavailable."""
        name = self._members.display_name(chat_id, sender_open_id)
        handle = f"@{name}" if name and name != sender_open_id else f"@{sender_open_id}"
        return f"{handle} {text}".strip()

    # ── reaction ──

    _REACTION_EMOJI_CHOICES: tuple[str, ...] = FEISHU_REACTION_EMOJI_CHOICES
    _REACTION_FALLBACK_EMOJI = "THUMBSUP"
    _REACTION_TOOL_NAME = "select_reaction_emoji"
    _RENDER_TOOL_NAME = "plan_feishu_render"

    @staticmethod
    def _extract_json_object(text: str) -> dict[str, Any] | None:
        """Best-effort parse for a JSON object from model text output."""
        raw = (text or "").strip()
        if not raw:
            return None
        candidates = [raw]
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            candidates.append(match.group(0))
        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
            except Exception:
                continue
            if isinstance(parsed, dict):
                return parsed
        return None

    @staticmethod
    def _fallback_render_mode(
        text: str, has_images: bool, has_actions: bool
    ) -> str:
        """Fallback mode when planner LLM is unavailable.

        Pure-agent policy: no content heuristics here.
        Keep only hard rendering constraints that cannot be inferred without
        model output.
        """
        if has_images or has_actions:
            return "interactive"
        return "text"

    @staticmethod
    def _coerce_bool(value: Any, default: bool = False) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "1", "yes", "y"}:
                return True
            if normalized in {"false", "0", "no", "n", ""}:
                return False
        return default

    @classmethod
    def _coerce_render_plan(
        cls,
        payload: dict[str, Any],
        *,
        original_content: str,
        is_group: bool,
        sender_open_id: str,
        has_images: bool,
        has_actions: bool,
        fallback_mode: str,
    ) -> _RenderPlan:
        """Normalize planner payload and enforce platform hard constraints."""
        mode_raw = str(payload.get("mode") or "").strip().lower()
        mode = mode_raw if mode_raw in {"text", "interactive"} else fallback_mode

        mention_sender = cls._coerce_bool(
            payload.get("mention_sender"),
            default=False,
        )

        body = str(payload.get("body") or "").strip()
        if not body:
            body = (original_content or "").strip()

        title = str(payload.get("title") or "").strip()
        if title:
            title = re.sub(r"^#{1,6}\s*", "", title).strip()
            if len(title) > 80:
                title = title[:80].rstrip()

        confidence = 0.0
        raw_conf = payload.get("confidence")
        if raw_conf is not None:
            try:
                confidence = float(raw_conf)
            except (TypeError, ValueError):
                confidence = 0.0
        confidence = max(0.0, min(1.0, confidence))
        rationale = str(payload.get("rationale") or "").strip()

        if not is_group or not sender_open_id:
            mention_sender = False

        # Hard constraints: media/actions and platform @mention behavior.
        if has_images or has_actions:
            mode = "interactive"
        if mention_sender and sender_open_id:
            mode = "interactive"

        return _RenderPlan(
            mode=mode,
            mention_sender=mention_sender,
            title=title,
            body=body,
            confidence=confidence,
            rationale=rationale,
        )

    def _init_render_llm(self) -> None:
        """Lazy init an LLM client used for render planning."""
        if self._render_llm_init_attempted:
            return
        self._render_llm_init_attempted = True
        try:
            from getall.config.loader import load_config
            from getall.providers.litellm_provider import LiteLLMProvider

            cfg = load_config()
            provider = cfg.get_provider()
            if not (provider and provider.api_key):
                logger.warning(
                    "Render planner LLM disabled: no provider API key configured"
                )
                return

            self._render_model = cfg.agents.defaults.model
            self._render_reasoning_effort = (
                cfg.agents.defaults.reasoning_effort or ""
            )
            self._render_llm = LiteLLMProvider(
                api_key=provider.api_key,
                api_base=cfg.get_api_base(),
                default_model=self._render_model,
                extra_headers=provider.extra_headers,
                provider_name=cfg.get_provider_name(),
            )
        except Exception as exc:
            logger.warning(f"Render planner LLM init failed: {exc}")

    async def _plan_render(
        self,
        *,
        content: str,
        chat_id: str,
        is_group: bool,
        sender_open_id: str,
        has_images: bool,
        has_actions: bool,
        source: str = "",
    ) -> _RenderPlan:
        """Use a dedicated LLM call to decide message presentation."""
        text = (content or "").strip()
        fallback_mode = self._fallback_render_mode(
            text, has_images=has_images, has_actions=has_actions
        )
        fallback_plan = self._coerce_render_plan(
            {},
            original_content=text,
            is_group=is_group,
            sender_open_id=sender_open_id,
            has_images=has_images,
            has_actions=has_actions,
            fallback_mode=fallback_mode,
        )
        if not text:
            return fallback_plan

        self._init_render_llm()
        if not self._render_llm:
            return fallback_plan

        sender_name = (
            self._members.display_name(chat_id, sender_open_id)
            if sender_open_id
            else ""
        )
        planner_input = {
            "chat_type": "group" if is_group else "private",
            "source": source or "normal",
            "sender_mention_available": bool(sender_open_id),
            "sender_name": sender_name,
            "has_images": bool(has_images),
            "has_actions": bool(has_actions),
            "content": text[:8000],
        }
        tools = [
            {
                "type": "function",
                "function": {
                    "name": self._RENDER_TOOL_NAME,
                    "description": (
                        "Plan Feishu render strategy and provide final body "
                        "for display."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "mode": {
                                "type": "string",
                                "enum": ["text", "interactive"],
                            },
                            "mention_sender": {"type": "boolean"},
                            "title": {"type": "string"},
                            "body": {"type": "string"},
                            "confidence": {"type": "number"},
                            "rationale": {"type": "string"},
                        },
                        "required": ["mode", "mention_sender", "body"],
                        "additionalProperties": False,
                    },
                },
            }
        ]
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a presentation planner for a Feishu/Lark AI "
                    "assistant. Your goal is to maximize human-like delivery "
                    "quality.\n"
                    "Call the tool plan_feishu_render exactly once.\n"
                    "Keep facts, numbers, links, and commitments unchanged.\n"
                    "Prefer mode=text for brief conversational or progress "
                    "updates.\n"
                    "Prefer mode=interactive for dense structured reports, "
                    "multi-section briefs, or visual hierarchy.\n"
                    "Use mention_sender=true only when direct person-targeting "
                    "improves clarity.\n"
                    "For mode=text, output natural conversational body "
                    "without markdown heading markers.\n"
                    "For mode=interactive, preserve structure and readability."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(planner_input, ensure_ascii=False),
            },
        ]
        try:
            response = await asyncio.wait_for(
                self._render_llm.chat(
                    messages=messages,
                    tools=tools,
                    model=self._render_model,
                    max_tokens=600,
                    temperature=0.1,
                    reasoning_effort=self._render_reasoning_effort,
                ),
                timeout=12.0,
            )
            payload: dict[str, Any] | None = None
            for call in response.tool_calls:
                if call.name == self._RENDER_TOOL_NAME and isinstance(
                    call.arguments, dict
                ):
                    payload = call.arguments
                    break
            if payload is None and response.content:
                payload = self._extract_json_object(response.content)
            if payload is None:
                logger.warning(
                    "Render planner returned no structured payload; using fallback plan"
                )
                return fallback_plan

            plan = self._coerce_render_plan(
                payload,
                original_content=text,
                is_group=is_group,
                sender_open_id=sender_open_id,
                has_images=has_images,
                has_actions=has_actions,
                fallback_mode=fallback_mode,
            )
            logger.debug(
                "Render plan decided: "
                f"mode={plan.mode}, mention_sender={plan.mention_sender}, "
                f"title_len={len(plan.title)}, confidence={plan.confidence:.2f}, "
                f"source={source or 'normal'}"
            )
            return plan
        except Exception as exc:
            logger.warning(
                f"Render planner failed; using fallback plan: {exc}"
            )
            return fallback_plan

    def _init_reaction_llm(self) -> None:
        """Lazy init a lightweight LLM client for emoji selection."""
        if self._reaction_llm_init_attempted:
            return
        self._reaction_llm_init_attempted = True
        try:
            from getall.config.loader import load_config
            from getall.providers.litellm_provider import LiteLLMProvider

            cfg = load_config()
            provider = cfg.get_provider()
            if not (provider and provider.api_key):
                logger.warning(
                    "Reaction LLM disabled: no provider API key configured"
                )
                return

            self._reaction_model = cfg.agents.defaults.model
            # Reactions are latency-sensitive; keep reasoning off.
            self._reaction_reasoning_effort = ""
            self._reaction_llm = LiteLLMProvider(
                api_key=provider.api_key,
                api_base=cfg.get_api_base(),
                default_model=self._reaction_model,
                extra_headers=provider.extra_headers,
                provider_name=cfg.get_provider_name(),
            )
        except Exception as exc:
            logger.warning(f"Reaction LLM init failed: {exc}")

    async def _reformat_for_card(self, content: str) -> str:
        """Use the LLM to rewrite *content* with ≤3 markdown tables.

        Called as a fallback when a Lark card exceeds the native table
        component limit.  The LLM is asked to preserve all data but merge
        or convert excess tables into bullet lists / bold key-value pairs.
        If the LLM is unavailable or fails, return *content* with tables
        stripped to plain key-value lines (best-effort programmatic
        fallback).
        """
        self._init_reaction_llm()
        if self._reaction_llm:
            try:
                resp = await asyncio.wait_for(
                    self._reaction_llm.chat(
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "You are a formatting assistant. The user "
                                    "will give you a markdown message that "
                                    "contains too many tables for a Lark/Feishu "
                                    "message card (max 5 native tables).\n\n"
                                    "Rewrite the message so it has AT MOST 3 "
                                    "markdown tables. Convert the rest into "
                                    "compact bullet lists or **bold key**: "
                                    "value lines. Keep ALL data — nothing may "
                                    "be dropped. Preserve headings, emojis, "
                                    "and overall structure. Output ONLY the "
                                    "rewritten markdown, no commentary."
                                ),
                            },
                            {"role": "user", "content": content},
                        ],
                        model=self._reaction_model,
                        max_tokens=4096,
                        temperature=0.2,
                        reasoning_effort=self._reaction_reasoning_effort,
                    ),
                    timeout=30.0,
                )
                reformatted = (resp.content or "").strip()
                if reformatted and len(reformatted) > 100:
                    logger.info(
                        "Card content reformatted by LLM "
                        f"({len(content)} → {len(reformatted)} chars)"
                    )
                    return reformatted
            except Exception as exc:
                logger.warning(f"LLM reformat failed, using programmatic "
                               f"fallback: {exc}")

        # ── Programmatic fallback: convert tables to key-value lines ──
        import re as _re
        def _table_to_kv(match: _re.Match[str]) -> str:
            lines = [
                ln.strip()
                for ln in match.group(1).strip().split("\n")
                if ln.strip()
            ]
            if len(lines) < 2:
                return match.group(0)
            headers = [c.strip() for c in lines[0].strip("|").split("|")]
            sep_idx = 1 if _re.match(r"^[ \t]*\|[-:\s|]+\|", lines[1]) else -1
            data_start = sep_idx + 1 if sep_idx >= 0 else 1
            out: list[str] = []
            for row in lines[data_start:]:
                cells = [c.strip() for c in row.strip("|").split("|")]
                pairs = " | ".join(
                    f"**{headers[i]}** {cells[i]}"
                    for i in range(min(len(headers), len(cells)))
                    if cells[i]
                )
                if pairs:
                    out.append(f"- {pairs}")
            return "\n".join(out) if out else match.group(0)

        return self._TABLE_RE.sub(_table_to_kv, content)

    def _fallback_reaction_emoji(
        self,
    ) -> str:
        """Protocol-safe fallback when reaction planner is unavailable.

        Pure-agent policy: no heuristic mapping from text patterns to emoji.
        Keep a single safe token as deterministic fallback, while guaranteeing
        the value is part of Feishu's supported enum.
        """
        if self._REACTION_FALLBACK_EMOJI in self._REACTION_EMOJI_CHOICES:
            return self._REACTION_FALLBACK_EMOJI
        if self._REACTION_EMOJI_CHOICES:
            return self._REACTION_EMOJI_CHOICES[0]
        return "THUMBSUP"

    @classmethod
    def _normalize_reaction_token(cls, token: str) -> str:
        """Return canonical emoji_type token or empty string if invalid."""
        candidate = (token or "").strip()
        if not candidate:
            return ""
        if candidate in cls._REACTION_EMOJI_CHOICES:
            return candidate
        lowered = candidate.lower()
        if not lowered:
            return ""
        hits = [
            name for name in cls._REACTION_EMOJI_CHOICES
            if name.lower() == lowered
        ]
        if len(hits) == 1:
            return hits[0]
        return ""

    async def _choose_reaction_emoji(
        self,
        message_id: str,
        msg_type: str,
        content: str,
        chat_type: str,
        event_hint: str = "",
    ) -> str:
        fallback = self._fallback_reaction_emoji()
        self._init_reaction_llm()
        if not self._reaction_llm:
            return fallback

        content_preview = (content or "").strip()
        if len(content_preview) > 240:
            content_preview = f"{content_preview[:240]}..."

        # Build event-aware system prompt for reaction emoji selection
        event_instruction = ""
        if event_hint:
            event_instruction = (
                f"\nA special event is active: {event_hint}. "
                "Strongly prefer themed emojis that match the event mood "
                "(e.g. LOVE/HEART/KISS/ROSE/FINGERHEART for romance, "
                "REDPACKET/FORTUNE/LUCK/FIRECRACKER for Chinese New Year, "
                "PARTY/BEER/Fire for Friday night, etc.).\n"
            )

        tools = [
            {
                "type": "function",
                "function": {
                    "name": self._REACTION_TOOL_NAME,
                    "description": (
                        "Select one valid Feishu reaction emoji_type for this message."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "emoji_type": {
                                "type": "string",
                                "enum": list(self._REACTION_EMOJI_CHOICES),
                            },
                            "rationale": {"type": "string"},
                        },
                        "required": ["emoji_type"],
                        "additionalProperties": False,
                    },
                },
            }
        ]
        messages = [
            {
                "role": "system",
                "content": (
                    "You pick ONE Lark reaction emoji_type for a user message.\n"
                    "Call tool select_reaction_emoji exactly once.\n"
                    "No free text output.\n"
                    f"Choose the most human-like reaction for the context.{event_instruction}"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"message_id: {message_id}\n"
                    f"chat_type: {chat_type}\n"
                    f"msg_type: {msg_type}\n"
                    f"content: {content_preview or '[empty]'}"
                ),
            },
        ]
        try:
            response = await asyncio.wait_for(
                self._reaction_llm.chat(
                    messages=messages,
                    tools=tools,
                    model=self._reaction_model,
                    max_tokens=64,
                    temperature=0.1,
                    reasoning_effort=self._reaction_reasoning_effort,
                ),
                timeout=4.0,
            )
            # Preferred path: structured tool call.
            for call in response.tool_calls:
                if call.name != self._REACTION_TOOL_NAME:
                    continue
                if not isinstance(call.arguments, dict):
                    continue
                token = self._normalize_reaction_token(
                    str(call.arguments.get("emoji_type", ""))
                )
                if token:
                    return token

            # Secondary path: tolerate provider/model returning JSON text.
            if response.content:
                payload = self._extract_json_object(response.content)
                if isinstance(payload, dict):
                    token = self._normalize_reaction_token(
                        str(payload.get("emoji_type", ""))
                    )
                    if token:
                        return token

            logger.warning(
                "Reaction planner returned no valid structured emoji_type; "
                f"using fallback={fallback}"
            )
        except Exception as exc:
            logger.warning(f"Reaction LLM choose failed: {exc}")
        return fallback

    def _add_reaction_sync(self, message_id: str, emoji_type: str) -> None:
        try:
            req = (
                CreateMessageReactionRequest.builder()
                .message_id(message_id)
                .request_body(
                    CreateMessageReactionRequestBody.builder()
                    .reaction_type(
                        Emoji.builder().emoji_type(emoji_type).build()
                    )
                    .build()
                )
                .build()
            )
            resp = self._client.im.v1.message_reaction.create(req)
            if not resp.success():
                logger.warning(
                    f"Reaction failed: code={resp.code}, msg={resp.msg}"
                )
        except Exception as exc:
            logger.warning(f"Error adding reaction: {exc}")

    async def _add_reaction(
        self, message_id: str, emoji_type: str = "THUMBSUP"
    ) -> None:
        if not self._client or not Emoji:
            return
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None, self._add_reaction_sync, message_id, emoji_type
        )

    # ── outbound (send) ──

    # ── table regexes ──

    # Standard markdown table (with separator) OR pipe table without separator
    _TABLE_RE = re.compile(
        r"("
        # Alt-1: standard markdown table with separator row
        r"(?:^[ \t]*\|.+\|[ \t]*\n)"
        r"(?:^[ \t]*\|[-:\s|]+\|[ \t]*\n)"
        r"(?:^[ \t]*\|.+\|[ \t]*\n?)+"
        r"|"
        # Alt-2: pipe table without separator (3+ consecutive rows)
        r"(?:^[ \t]*\|.+\|[ \t]*\n){3,}"
        r")",
        re.MULTILINE,
    )

    # Box-drawing table (┌─┬─┐ │…│ ├─┼─┤ └─┴─┘)
    _BOX_TABLE_RE = re.compile(
        r"((?:^[ \t]*[┌├└│].*\n?)+)",
        re.MULTILINE,
    )

    # ── heading / header extraction ──

    _HEADING_RE = re.compile(r"^#{1,2}\s+(.+?)$", re.MULTILINE)

    # After the first heading is extracted for the card header,
    # ALL remaining headings (#, ##, ###, …) must be downgraded to bold.
    # Lark card markdown technically lists # / ## as "supported" but in
    # practice rendering is unreliable inside card elements, so we
    # convert every heading to **bold** for guaranteed visibility.
    _ANY_HEADING_RE = re.compile(r"^#{1,6}\s+(.+?)$", re.MULTILINE)

    @classmethod
    def _downgrade_headings(cls, text: str) -> str:
        """Replace any remaining ``# heading`` … ``###### heading`` with ``**heading**``.

        If the heading text already contains ``**bold**`` markers, they are
        stripped first to avoid broken ``****`` nesting.
        """

        def _repl(m: re.Match) -> str:
            inner = m.group(1).replace("**", "").strip()
            return f"**{inner}**"

        return cls._ANY_HEADING_RE.sub(_repl, text)

    # Inline code (single/double backtick) is NOT supported by Lark card
    # markdown — only fenced code blocks (triple backtick) work.
    # Strip the backtick delimiters, keeping the content as plain text.
    # Ref: https://www.feishu.cn/content/7gprunv5 (no inline code listed)
    _INLINE_CODE_RE = re.compile(r"(?<!`)(`{1,2})(?!`)(.*?)(?<!`)\1(?!`)")

    # Stray empty backtick sequences on their own lines (e.g. lone `` or ```).
    _STRAY_BACKTICKS_RE = re.compile(r"^[ \t]*`{1,3}[ \t]*$", re.MULTILINE)

    # Leftover markdown image syntax with file:// or other schemes that
    # _extract_image_paths couldn't fully clean.
    _LEFTOVER_IMG_RE = re.compile(
        r"!\[[^\]]*\]\([^)]*\)", re.IGNORECASE
    )

    # Orphaned labels left behind after image path extraction, e.g.
    # "图表路径：", "Chart path:", "图表：" followed by empty/whitespace.
    _ORPHAN_PATH_LABEL_RE = re.compile(
        r"^[ \t]*(?:图表路径|图表地址|图片路径|路径|[Cc]hart\s*[Pp]ath|[Ff]ile\s*[Pp]ath)"
        r"[：:\s]*$",
        re.MULTILINE,
    )

    @classmethod
    def _sanitize_lark_md(cls, text: str) -> str:
        """Sanitise text for Lark card markdown rendering.

        * Strip inline code backticks (unsupported in Lark card markdown).
        * Remove orphaned empty backtick sequences.
        * Remove leftover ``![alt](file://...)`` image syntax.
        * Remove orphaned path labels (e.g. "图表路径：").
        """
        # Strip inline code backticks (` or ``) but keep the content.
        # Preserve fenced code blocks (```) by only matching 1-2 backticks.
        text = cls._INLINE_CODE_RE.sub(r"\2", text)
        # Remove stray backtick-only lines
        text = cls._STRAY_BACKTICKS_RE.sub("", text)
        # Remove leftover markdown image refs (file:// etc.)
        text = cls._LEFTOVER_IMG_RE.sub("", text)
        # Remove orphaned path labels left after image extraction
        text = cls._ORPHAN_PATH_LABEL_RE.sub("", text)
        # Collapse excessive blank lines
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    _VERDICT_COLOR_MAP: dict[str, str] = {
        "🟢": "green",
        "🟡": "yellow",
        "🔴": "red",
        "✅": "green",
        "❌": "red",
    }

    @classmethod
    def _extract_card_header(
        cls, content: str
    ) -> tuple[str, str, str]:
        """Extract card header title, color template, and remaining body.

        Returns ``(title, template_color, remaining_content)``.
        * ``title`` comes from the first ``#``/``##`` heading found.
        * ``template_color`` is derived from verdict emojis (🟢/🟡/🔴).
        * ``remaining_content`` is the original text minus the heading line.
        """
        title = ""
        remaining = content

        m = cls._HEADING_RE.search(content)
        if m:
            # Card header uses plain_text — strip markdown & <at> tags
            title = m.group(1).replace("**", "").replace("~~", "")
            title = re.sub(r"<at[^>]*>(.*?)</at>", r"\1", title).strip()
            remaining = (content[: m.start()] + content[m.end() :]).strip()
            remaining = re.sub(r"\n{3,}", "\n\n", remaining)

        template = "blue"
        for emoji, color in cls._VERDICT_COLOR_MAP.items():
            if emoji in content:
                template = color
                break

        return title, template, remaining

    # ── table parsers ──

    @staticmethod
    def _parse_md_table(table_text: str) -> dict | None:
        """Parse a markdown pipe table (with or without separator row)."""
        lines = [ln.strip() for ln in table_text.strip().split("\n") if ln.strip()]
        if len(lines) < 2:
            return None
        _split = lambda ln: [c.strip() for c in ln.strip("|").split("|")]
        headers = _split(lines[0])

        # Skip separator row (|---|---|) when present
        has_sep = len(lines) > 2 and bool(
            re.match(r"^[ \t]*\|[-:\s|]+\|[ \t]*$", lines[1])
        )
        data_start = 2 if has_sep else 1
        rows = [_split(ln) for ln in lines[data_start:]]
        if not rows:
            return None

        cols = [
            {
                "tag": "column",
                "name": f"c{i}",
                "display_name": h,
                "width": "auto",
                "data_type": "lark_md",
            }
            for i, h in enumerate(headers)
        ]
        return {
            "tag": "table",
            "page_size": len(rows) + 1,
            "columns": cols,
            "rows": [
                {f"c{i}": r[i] if i < len(r) else "" for i in range(len(headers))}
                for r in rows
            ],
        }

    @staticmethod
    def _parse_box_table(table_text: str) -> dict | None:
        """Parse a box-drawing character table (┌─┬─┐ │…│ └─┴─┘)."""
        lines = table_text.strip().split("\n")
        # Keep only data rows (contain │ AND have non-decoration content)
        data_lines = [
            ln
            for ln in lines
            if "│" in ln
            and any(c not in "│─┌┐├┤└┘┬┴┼ \t\n" for c in ln)
        ]
        if len(data_lines) < 2:
            return None
        _split = lambda ln: [c.strip() for c in ln.split("│") if c.strip()]
        headers = _split(data_lines[0])
        rows = [_split(ln) for ln in data_lines[1:]]
        if not headers or not rows:
            return None
        cols = [
            {"tag": "column", "name": f"c{i}", "display_name": h, "width": "auto"}
            for i, h in enumerate(headers)
        ]
        return {
            "tag": "table",
            "page_size": len(rows) + 1,
            "columns": cols,
            "rows": [
                {f"c{i}": r[i] if i < len(r) else "" for i in range(len(headers))}
                for r in rows
            ],
        }

    # ── card element builder ──

    # ``---`` on its own line — must become a native ``{"tag": "hr"}``
    # element because the card markdown component does NOT render ``---``.
    # Ref: https://open.feishu.cn/document/common-capabilities/message-card/message-cards-content/divider-line-module
    _HR_LINE_RE = re.compile(r"^[ \t]*---[ \t]*$", re.MULTILINE)

    @classmethod
    def _split_hrs(cls, elements: list[dict]) -> list[dict]:
        """Replace ``---`` inside markdown elements with native ``hr`` elements.

        Lark card markdown does NOT render ``---`` as a divider — it must be
        a separate ``{"tag": "hr"}`` element in the card ``elements`` array.
        """
        result: list[dict] = []
        for el in elements:
            if el.get("tag") != "markdown":
                result.append(el)
                continue
            parts = cls._HR_LINE_RE.split(el["content"])
            for i, part in enumerate(parts):
                text = part.strip()
                if text:
                    result.append({"tag": "markdown", "content": text})
                if i < len(parts) - 1:
                    result.append({"tag": "hr"})
        return result

    def _build_card_elements(self, content: str) -> list[dict]:
        """Convert markdown content into a list of Lark card ``elements``.

        * Pipe tables and box-drawing tables → native ``table`` components.
        * ``###`` and deeper headings → downgraded to **bold** (unsupported).
        * ``---`` dividers → native ``{"tag": "hr"}`` elements.
        * Everything else → ``markdown`` elements (Lark supports: bold,
          italic, strikethrough, links, inline code, code blocks,
          ordered/unordered lists).
        """
        # Downgrade ALL remaining headings to bold (first one is already
        # extracted into the card header by _extract_card_header).
        content = self._downgrade_headings(content)
        # Clean up stray backticks, leftover image syntax.
        content = self._sanitize_lark_md(content)

        # Collect table regions: (start, end, parsed_component | None)
        regions: list[tuple[int, int, dict | None]] = []

        for m in self._TABLE_RE.finditer(content):
            regions.append(
                (m.start(), m.end(), self._parse_md_table(m.group(1)))
            )

        for m in self._BOX_TABLE_RE.finditer(content):
            start, end = m.start(), m.end()
            if any(s <= start < e for s, e, _ in regions):
                continue  # overlaps with a pipe-table match
            regions.append((start, end, self._parse_box_table(m.group(1))))

        regions.sort(key=lambda r: r[0])

        elements: list[dict] = []
        last = 0
        native_table_count = 0
        for start, end, tbl in regions:
            if start < last:
                continue
            before = content[last:start].strip()
            if before:
                elements.append({"tag": "markdown", "content": before})
            # Lark cards have a hard limit on native table components.
            # Excess tables fall back to markdown text to avoid send errors.
            if tbl:
                native_table_count += 1
                if native_table_count > _MAX_CARD_TABLES:
                    tbl = None  # keep as markdown
            if tbl:
                elements.append(tbl)
            else:
                elements.append(
                    {"tag": "markdown", "content": content[start:end].strip()}
                )
            last = end

        tail = content[last:].strip()
        if tail:
            elements.append({"tag": "markdown", "content": tail})

        if not elements:
            elements = [{"tag": "markdown", "content": content}]

        # --- in markdown doesn't render; replace with native hr elements
        elements = self._split_hrs(elements)

        return elements

    _ACTION_KEY_TO_LABEL: dict[str, str] = {
        "view_detail": "查看详情",
        "subscribe": "订阅",
        "mute": "静音",
        "continue": "继续",
    }
    _ACTION_LABEL_ORDER: tuple[str, ...] = ("查看详情", "订阅", "静音", "继续")
    _ACTION_LABEL_STYLE: dict[str, str] = {
        "查看详情": "default",
        "订阅": "primary",
        "静音": "danger",
        "继续": "primary",
    }
    _URL_RE = re.compile(r"https?://[^\s<>()\"']+", re.IGNORECASE)

    @classmethod
    def _normalize_action_label(cls, value: str) -> str | None:
        raw = (value or "").strip()
        if not raw:
            return None
        normalized_key = raw.lower().replace(" ", "_")
        if normalized_key in cls._ACTION_KEY_TO_LABEL:
            return cls._ACTION_KEY_TO_LABEL[normalized_key]
        if raw in cls._ACTION_LABEL_ORDER:
            return raw
        return None

    @staticmethod
    def _sanitize_action_url(value: str) -> str | None:
        url = (value or "").strip()
        if url.startswith(("http://", "https://", "lark://")):
            return url
        return None

    def _extract_card_actions(
        self, content: str, metadata: dict[str, Any]
    ) -> list[dict[str, str]]:
        """Build card actions from explicit metadata and real URLs only.

        We never fabricate actions. Subscribe/mute/continue require explicit
        URLs in metadata. For "查看详情", we can safely use a real URL found
        in the message text when present.
        """
        actions_by_label: dict[str, dict[str, str]] = {}

        action_urls = metadata.get("feishu_action_urls")
        if isinstance(action_urls, dict):
            for raw_label, raw_url in action_urls.items():
                label = self._normalize_action_label(str(raw_label))
                url = self._sanitize_action_url(str(raw_url))
                if label and url:
                    actions_by_label[label] = {
                        "label": label,
                        "url": url,
                        "style": self._ACTION_LABEL_STYLE.get(label, "default"),
                    }

        raw_actions = metadata.get("feishu_actions")
        if isinstance(raw_actions, list):
            for item in raw_actions:
                if not isinstance(item, dict):
                    continue
                label = self._normalize_action_label(
                    str(item.get("label") or item.get("key") or "")
                )
                url = self._sanitize_action_url(str(item.get("url") or ""))
                if not (label and url):
                    continue
                style = str(item.get("style") or "").strip().lower()
                if style not in {"default", "primary", "danger"}:
                    style = self._ACTION_LABEL_STYLE.get(label, "default")
                actions_by_label[label] = {
                    "label": label,
                    "url": url,
                    "style": style,
                }

        # Real URL present in content -> safe "查看详情" fallback.
        if "查看详情" not in actions_by_label:
            m = self._URL_RE.search(content)
            if m:
                detail_url = self._sanitize_action_url(m.group(0))
                if detail_url:
                    actions_by_label["查看详情"] = {
                        "label": "查看详情",
                        "url": detail_url,
                        "style": self._ACTION_LABEL_STYLE["查看详情"],
                    }

        actions = list(actions_by_label.values())
        order = {label: idx for idx, label in enumerate(self._ACTION_LABEL_ORDER)}
        actions.sort(key=lambda a: order.get(a["label"], 999))
        return actions[:4]

    @staticmethod
    def _build_action_element(actions: list[dict[str, str]]) -> dict:
        buttons = [
            {
                "tag": "button",
                "text": {"tag": "plain_text", "content": a["label"]},
                "url": a["url"],
                "type": a.get("style", "default"),
            }
            for a in actions
        ]
        return {"tag": "action", "actions": buttons}

    # ── image path extraction ──

    # Match: [GENERATED_IMAGE:/path], ![alt](/path), ![alt](file:///path),
    # bare /path/to/file.png
    _GENERATED_TAG_RE = re.compile(r"\[GENERATED_IMAGE:(.*?)\]")
    # Full markdown image syntax: ![alt](file:///path) or ![alt](/path)
    _MD_IMG_SYNTAX_RE = re.compile(
        r"!\[[^\]]*\]\((?:file://)?(/[^\s\)\"'<>]+\.(?:png|jpg|jpeg|gif|webp|bmp))\)",
        re.IGNORECASE,
    )
    # Bare file path (no markdown syntax)
    _IMG_PATH_RE = re.compile(
        r"(?<![(\[])(/[^\s\)\]\"'<>]+\.(?:png|jpg|jpeg|gif|webp|bmp))",
        re.IGNORECASE,
    )

    # ── post (rich-text) parsing ──

    def _parse_post_content(
        self,
        message: Any,
        sender_id: str,
        sender_name: str,
        chat_id: str,
        chat_type: str,
    ) -> tuple[str, str, list[str]]:
        """Parse a Feishu *post* (rich-text) message into plain text.

        Post content JSON structure::

            {
              "<locale>": {            # e.g. "zh_cn", "en_us", "ja_jp"
                "title": "...",
                "content": [           # list of paragraphs
                  [                    # each paragraph is a list of elements
                    {"tag": "text", "text": "Hello "},
                    {"tag": "at",  "user_id": "ou_xxx", "user_name": "Name"},
                    {"tag": "a",   "text": "link",  "href": "https://..."},
                    {"tag": "img", "image_key": "..."},
                    {"tag": "media", "file_key": "..."},
                    {"tag": "emotion", "emoji_type": "SMILE"},
                  ],
                  ...
                ]
              }
            }

        Returns:
            (content, sender_name, image_keys) — extracted text,
            possibly updated sender name, and image keys for download.
        """
        try:
            parsed = json.loads(message.content)
        except (json.JSONDecodeError, TypeError):
            return "[post]", sender_name, []

        # Locate the first available locale block
        locale_block: dict[str, Any] | None = None
        for locale_key in ("zh_cn", "en_us", "ja_jp"):
            if locale_key in parsed:
                locale_block = parsed[locale_key]
                break
        if locale_block is None:
            # Fallback: pick any first dict value that has "content" key
            for v in parsed.values():
                if isinstance(v, dict) and "content" in v:
                    locale_block = v
                    break
        if locale_block is None:
            return "[post]", sender_name, []

        title = locale_block.get("title", "")
        paragraphs: list[list[dict[str, Any]]] = locale_block.get("content", [])

        lines: list[str] = []
        if title:
            lines.append(title)

        bot_mention_names: list[str] = []
        image_keys: list[str] = []

        for para in paragraphs:
            if not isinstance(para, list):
                continue
            parts: list[str] = []
            for elem in para:
                if not isinstance(elem, dict):
                    continue
                tag = elem.get("tag", "")
                if tag == "text":
                    parts.append(elem.get("text", ""))
                elif tag == "a":
                    href = elem.get("href", "")
                    link_text = elem.get("text", href)
                    parts.append(f"{link_text}({href})" if href else link_text)
                elif tag == "at":
                    user_id = elem.get("user_id", "")
                    user_name = elem.get("user_name", "") or elem.get("user_name", "")
                    # Learn sender name from @mention
                    if user_id == sender_id and user_name:
                        sender_name = user_name
                    # Detect bot mention
                    if self._bot_open_id and user_id == self._bot_open_id:
                        if user_name:
                            bot_mention_names.append(user_name)
                        continue  # Don't include bot @mention in content
                    # Track group member
                    if user_id and user_name and chat_type == "group":
                        if user_id != self._bot_open_id:
                            self._members.track_sender(chat_id, user_id, user_name)
                    parts.append(f"@{user_name}" if user_name else f"@{user_id}")
                elif tag == "img":
                    ik = elem.get("image_key", "")
                    if ik:
                        image_keys.append(ik)
                    parts.append("[image]")
                elif tag == "media":
                    parts.append("[media]")
                elif tag == "emotion":
                    parts.append(f"[{elem.get('emoji_type', 'emoji')}]")
                # Other tags: silently skip
            line = "".join(parts).strip()
            if line:
                lines.append(line)

        content = "\n".join(lines)

        # Strip bot @mentions
        for bn in bot_mention_names:
            content = content.replace(f"@{bn}", "").strip()

        return content or "[post]", sender_name, image_keys

    def _extract_image_paths(self, text: str) -> tuple[str, list[str]]:
        """Extract image file paths from text, return (cleaned_text, paths)."""
        paths: list[str] = []
        cleaned = text
        seen: set[str] = set()

        # 1. [GENERATED_IMAGE:...] tags (highest priority)
        for m in self._GENERATED_TAG_RE.finditer(text):
            fpath = m.group(1).strip()
            if fpath not in seen and Path(fpath).is_file():
                paths.append(fpath)
                seen.add(fpath)
            cleaned = cleaned.replace(m.group(0), "")

        # 2. Full markdown image syntax: ![alt](file:///path) or ![alt](/path)
        for m in self._MD_IMG_SYNTAX_RE.finditer(cleaned):
            fpath = m.group(1)
            if fpath not in seen and Path(fpath).is_file():
                paths.append(fpath)
                seen.add(fpath)
            # Always remove the full syntax to avoid leftover ![...](...) text
            cleaned = cleaned.replace(m.group(0), "")

        # 3. Bare paths (no markdown syntax)
        for m in self._IMG_PATH_RE.finditer(cleaned):
            fpath = m.group(1)
            if fpath not in seen and Path(fpath).is_file():
                paths.append(fpath)
                seen.add(fpath)
                cleaned = cleaned.replace(m.group(0), "")

        # Clean up leftover whitespace
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
        return cleaned, paths

    # ── image upload ──

    def _upload_image_sync(self, file_path: str) -> str | None:
        """Upload a local image file to Lark, return image_key or None."""
        from lark_oapi.api.im.v1 import (
            CreateImageRequest as ImgReq,
            CreateImageRequestBody as ImgBody,
        )
        from pathlib import Path

        p = Path(file_path)
        if not p.is_file():
            logger.warning(f"Image file not found: {file_path}")
            return None

        suffix = p.suffix.lower()
        if suffix not in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}:
            return None

        try:
            with open(p, "rb") as f:
                req = (
                    ImgReq.builder()
                    .request_body(
                        ImgBody.builder()
                        .image_type("message")
                        .image(f)
                        .build()
                    )
                    .build()
                )
                resp = self._client.im.v1.image.create(req)

            if resp.success() and resp.data and resp.data.image_key:
                logger.debug(f"Uploaded image: {p.name} → {resp.data.image_key}")
                return resp.data.image_key
            else:
                logger.warning(
                    f"Image upload failed: code={resp.code}, msg={resp.msg}"
                )
                return None
        except Exception as exc:
            logger.warning(f"Error uploading image {file_path}: {exc}")
            return None

    async def _upload_image(self, file_path: str) -> str | None:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, self._upload_image_sync, file_path
        )

    # ── audio download / upload ──

    def _download_audio_sync(self, message_id: str, file_key: str) -> str | None:
        """Download an audio resource from Lark, return local file path or None."""
        try:
            req = (
                GetMessageResourceRequest.builder()
                .message_id(message_id)
                .file_key(file_key)
                .type("file")
                .build()
            )
            resp = self._client.im.v1.message_resource.get(req)

            if not resp.success():
                logger.warning(
                    f"Audio download failed: code={resp.code}, msg={resp.msg}"
                )
                return None

            # Save to media directory.
            import time as _t
            voice_dir = Path.home() / ".getall" / "media" / "voice"
            voice_dir.mkdir(parents=True, exist_ok=True)
            out_path = voice_dir / f"feishu_{int(_t.time() * 1000)}.ogg"

            if hasattr(resp, "file") and resp.file:
                out_path.write_bytes(resp.file.read())
            else:
                logger.warning("Audio download: resp.file is empty")
                return None

            logger.debug(f"Downloaded audio: {message_id}/{file_key} → {out_path}")
            return str(out_path)

        except Exception as exc:
            logger.warning(f"Error downloading audio {file_key}: {exc}")
            return None

    async def _download_audio(self, message_id: str, file_key: str) -> str | None:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, self._download_audio_sync, message_id, file_key
        )

    def _upload_audio_sync(self, file_path: str) -> str | None:
        """Upload a local opus audio file to Lark, return file_key or None."""
        p = Path(file_path)
        if not p.is_file():
            logger.warning(f"Audio file not found: {file_path}")
            return None

        try:
            with open(p, "rb") as f:
                req = (
                    CreateFileRequest.builder()
                    .request_body(
                        CreateFileRequestBody.builder()
                        .file_type("opus")
                        .file_name(p.name)
                        .file(f)
                        .build()
                    )
                    .build()
                )
                resp = self._client.im.v1.file.create(req)

            if resp.success() and resp.data and resp.data.file_key:
                logger.debug(f"Uploaded audio: {p.name} → {resp.data.file_key}")
                return resp.data.file_key
            else:
                logger.warning(
                    f"Audio upload failed: code={resp.code}, msg={resp.msg}"
                )
                return None
        except Exception as exc:
            logger.warning(f"Error uploading audio {file_path}: {exc}")
            return None

    async def _upload_audio(self, file_path: str) -> str | None:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, self._upload_audio_sync, file_path
        )

    # ── image / file download (inbound multimodal) ──

    _IMAGE_SUFFIXES = frozenset({
        ".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp",
    })
    _TEXT_SUFFIXES = frozenset({
        ".txt", ".csv", ".json", ".md", ".py", ".js", ".ts", ".html", ".css",
        ".xml", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".log", ".sql",
        ".sh", ".bash", ".go", ".rs", ".java", ".kt", ".swift", ".rb",
        ".php", ".c", ".cpp", ".h", ".hpp", ".r", ".lua",
    })
    _MAX_FILE_READ_BYTES = 50 * 1024  # 50 KB

    @staticmethod
    def _detect_image_ext(data: bytes) -> str:
        """Detect image format from magic bytes, return file extension."""
        if data[:3] == b"\xff\xd8\xff":
            return ".jpg"
        if data[:4] == b"\x89PNG":
            return ".png"
        if data[:4] == b"GIF8":
            return ".gif"
        if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
            return ".webp"
        if data[:2] == b"BM":
            return ".bmp"
        return ".png"  # fallback

    def _download_image_sync(
        self, message_id: str, file_key: str,
    ) -> str | None:
        """Download an image resource from Lark, return local file path or None."""
        try:
            req = (
                GetMessageResourceRequest.builder()
                .message_id(message_id)
                .file_key(file_key)
                .type("image")
                .build()
            )
            resp = self._client.im.v1.message_resource.get(req)

            if not resp.success():
                logger.warning(
                    f"Image download failed: code={resp.code}, msg={resp.msg}"
                )
                return None

            if not (hasattr(resp, "file") and resp.file):
                logger.warning("Image download: resp.file is empty")
                return None

            data = resp.file.read()
            if not data:
                return None

            ext = self._detect_image_ext(data)
            image_dir = Path.home() / ".getall" / "media" / "images"
            image_dir.mkdir(parents=True, exist_ok=True)
            out_path = image_dir / f"feishu_{int(_time.time() * 1000)}{ext}"
            out_path.write_bytes(data)

            logger.debug(
                f"Downloaded image: {message_id}/{file_key} → {out_path}"
            )
            return str(out_path)

        except Exception as exc:
            logger.warning(f"Error downloading image {file_key}: {exc}")
            return None

    async def _download_image(
        self, message_id: str, file_key: str,
    ) -> str | None:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, self._download_image_sync, message_id, file_key,
        )

    def _download_file_sync(
        self, message_id: str, file_key: str, file_name: str,
    ) -> str | None:
        """Download a file resource from Lark, return local file path or None."""
        try:
            req = (
                GetMessageResourceRequest.builder()
                .message_id(message_id)
                .file_key(file_key)
                .type("file")
                .build()
            )
            resp = self._client.im.v1.message_resource.get(req)

            if not resp.success():
                logger.warning(
                    f"File download failed: code={resp.code}, msg={resp.msg}"
                )
                return None

            if not (hasattr(resp, "file") and resp.file):
                logger.warning("File download: resp.file is empty")
                return None

            data = resp.file.read()
            if not data:
                return None

            file_dir = Path.home() / ".getall" / "media" / "files"
            file_dir.mkdir(parents=True, exist_ok=True)
            safe_name = re.sub(r"[^\w.\-]", "_", file_name)
            safe_name = re.sub(r"\.{2,}", "_", safe_name)  # collapse ..
            safe_name = safe_name.lstrip("._")  # no hidden files
            safe_name = safe_name or "file"
            out_path = file_dir / f"{int(_time.time() * 1000)}_{safe_name}"
            out_path.write_bytes(data)

            logger.debug(
                f"Downloaded file: {message_id}/{file_key} → {out_path}"
            )
            return str(out_path)

        except Exception as exc:
            logger.warning(f"Error downloading file {file_key}: {exc}")
            return None

    async def _download_file(
        self, message_id: str, file_key: str, file_name: str,
    ) -> str | None:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, self._download_file_sync, message_id, file_key, file_name,
        )

    @classmethod
    def _read_file_as_text(cls, file_path: str, file_name: str) -> str:
        """Read file content for text-based files; return description for others."""
        p = Path(file_path)
        suffix = p.suffix.lower()
        if suffix in cls._TEXT_SUFFIXES:
            try:
                raw = p.read_bytes()
                if len(raw) > cls._MAX_FILE_READ_BYTES:
                    text = raw[: cls._MAX_FILE_READ_BYTES].decode(
                        "utf-8", errors="replace"
                    )
                    text += "\n...(truncated)"
                else:
                    text = raw.decode("utf-8", errors="replace")
                return f"[file: {file_name}]\n```\n{text}\n```"
            except Exception:
                pass
        return f"[file: {file_name}, saved: {file_path}]"

    # ── outbound (send) ──

    async def send(self, msg: OutboundMessage) -> None:
        """Send a message through Lark/Feishu.

        Long responses are split into chunks (short lines or paragraphs)
        and sent one by one with a short delay to feel more human.
        Short plain-text chunks are sent as ``text`` messages (compact bubble);
        rich content (tables, images, markdown) uses interactive cards.
        """
        if not self._client:
            logger.warning("Lark client not initialised")
            return

        try:
            is_group = msg.chat_id.startswith("oc_")
            recv_type = "chat_id" if is_group else "open_id"

            # ── Voice message: upload opus → send as audio ──
            audio_sent = False
            if msg.audio_path and Path(msg.audio_path).is_file():
                file_key = await self._upload_audio(msg.audio_path)
                if file_key:
                    audio_body = json.dumps(
                        {"file_key": file_key}, ensure_ascii=False
                    )
                    await self._send_msg(recv_type, msg.chat_id, "audio", audio_body)
                    audio_sent = True
                else:
                    logger.warning("Audio upload failed, falling back to text reply")

            content = msg.content

            if is_group:
                content = self._resolve_at_tags(msg.chat_id, content)

            # Collect image paths: from msg.media + extracted from content text
            image_paths: list[str] = list(msg.media or [])
            content, extra_paths = self._extract_image_paths(content)
            image_paths.extend(extra_paths)
            # Deduplicate while preserving order
            image_paths = list(dict.fromkeys(image_paths))

            # If audio was sent and there are no images/actions, we're done.
            if audio_sent and not image_paths:
                return

            # If audio was sent but there ARE images, send images as a
            # follow-up card (skip the text chunks — the voice covers that).
            if audio_sent and image_paths:
                elements: list[dict[str, Any]] = []
                for img_path in image_paths:
                    img_key = await self._upload_image(img_path)
                    if img_key:
                        elements.append({
                            "tag": "img",
                            "img_key": img_key,
                            "alt": {
                                "tag": "plain_text",
                                "content": Path(img_path).stem,
                            },
                        })
                if elements:
                    card = json.dumps(
                        {"config": {"wide_screen_mode": True}, "elements": elements},
                        ensure_ascii=False,
                    )
                    await self._send_msg(recv_type, msg.chat_id, "interactive", card)
                return

            action_items = self._extract_card_actions(content, msg.metadata or {})

            # Event system: override card theme when a limited-time event is active.
            event_card_theme = (msg.metadata or {}).get("event_card_theme", "")

            sender_oid = ""
            if is_group:
                sender_oid = (msg.metadata or {}).get("sender_open_id", "")
            source = str((msg.metadata or {}).get("source", "") or "").strip().lower()

            render_plan = await self._plan_render(
                content=content,
                chat_id=msg.chat_id,
                is_group=is_group,
                sender_open_id=sender_oid,
                has_images=bool(image_paths),
                has_actions=bool(action_items),
                source=source,
            )

            if render_plan.mode == "interactive":
                planned_body = (render_plan.body or content).strip()
                planned_title = (render_plan.title or "").strip()

                # Extract first heading only for card-title inference and to avoid
                # duplicate visible titles in body.
                inferred_title, template, body_wo_heading = self._extract_card_header(
                    planned_body
                )
                title = planned_title or inferred_title
                if inferred_title:
                    body = body_wo_heading
                else:
                    body = planned_body

                if render_plan.mention_sender and sender_oid:
                    if f"<at id={sender_oid}>" not in body:
                        body = self._prepend_sender_at(
                            msg.chat_id, sender_oid, body
                        )

                # Keep event theme as the top-priority visual override.
                if event_card_theme:
                    template = event_card_theme

                elements = self._build_card_elements(body)
                for img_path in image_paths:
                    img_key = await self._upload_image(img_path)
                    if img_key:
                        elements.append({
                            "tag": "img",
                            "img_key": img_key,
                            "alt": {
                                "tag": "plain_text",
                                "content": Path(img_path).stem,
                            },
                        })
                if action_items:
                    elements.append(self._build_action_element(action_items))
                card_data: dict[str, Any] = {
                    "config": {"wide_screen_mode": True},
                    "elements": elements,
                }
                if title:
                    card_data["header"] = {
                        "title": {"content": title, "tag": "plain_text"},
                        "template": template,
                    }
                card = json.dumps(card_data, ensure_ascii=False)
                try:
                    await self._send_msg(recv_type, msg.chat_id, "interactive", card)
                except RuntimeError as card_exc:
                    if "table" in str(card_exc).lower() and (
                        "over limit" in str(card_exc).lower()
                        or "11310" in str(card_exc)
                    ):
                        # Card table limit hit — ask LLM to reformat with
                        # fewer tables, then retry as a new card.
                        logger.warning(
                            "Card table limit exceeded, requesting LLM "
                            f"reformat: {card_exc}"
                        )
                        reformatted = await self._reformat_for_card(body)
                        r_title, r_tmpl, r_body = self._extract_card_header(
                            reformatted
                        )
                        if (
                            render_plan.mention_sender
                            and sender_oid
                            and f"<at id={sender_oid}>" not in r_body
                        ):
                            r_body = self._prepend_sender_at(
                                msg.chat_id, sender_oid, r_body
                            )
                        if event_card_theme:
                            r_tmpl = event_card_theme
                        r_elems = self._build_card_elements(r_body)
                        r_card_data: dict[str, Any] = {
                            "config": {"wide_screen_mode": True},
                            "elements": r_elems,
                        }
                        if r_title:
                            r_card_data["header"] = {
                                "title": {
                                    "content": r_title,
                                    "tag": "plain_text",
                                },
                                "template": r_tmpl,
                            }
                        r_card = json.dumps(r_card_data, ensure_ascii=False)
                        try:
                            await self._send_msg(
                                recv_type, msg.chat_id, "interactive",
                                r_card,
                            )
                        except Exception:
                            # Last resort: plain text
                            logger.warning(
                                "Reformatted card also failed, sending "
                                "plain text"
                            )
                            fb = reformatted
                            if render_plan.mention_sender and sender_oid:
                                fb = self._prepend_sender_plain(
                                    msg.chat_id, sender_oid, fb
                                )
                            await self._send_msg(
                                recv_type, msg.chat_id, "text",
                                json.dumps(
                                    {"text": fb}, ensure_ascii=False
                                ),
                            )
                    else:
                        raise
            else:
                text_content = (render_plan.body or content).strip() or content
                if render_plan.mention_sender and sender_oid:
                    text_content = self._prepend_sender_plain(
                        msg.chat_id, sender_oid, text_content
                    )
                text_body = json.dumps(
                    {"text": text_content}, ensure_ascii=False
                )
                await self._send_msg(recv_type, msg.chat_id, "text", text_body)

        except Exception as exc:
            logger.error(f"Error sending Lark message: {exc}")
            raise

    async def _send_msg(
        self, recv_type: str, recv_id: str, msg_type: str, content: str
    ) -> None:
        """Send one message to Lark (text, interactive card, etc.)."""
        req = (
            CreateMessageRequest.builder()
            .receive_id_type(recv_type)
            .request_body(
                CreateMessageRequestBody.builder()
                .receive_id(recv_id)
                .msg_type(msg_type)
                .content(content)
                .build()
            )
            .build()
        )
        loop = asyncio.get_running_loop()
        resp = await loop.run_in_executor(
            None, self._client.im.v1.message.create, req
        )

        if not resp.success():
            error_detail = (
                f"Send failed: code={resp.code}, msg={resp.msg}, "
                f"log_id={resp.get_log_id()}"
            )
            logger.error(error_detail)
            raise RuntimeError(error_detail)
        else:
            sent_mid = str(getattr(getattr(resp, "data", None), "message_id", "") or "")
            if sent_mid:
                self._sent_message_ids[sent_mid] = None
                while len(self._sent_message_ids) > _SENT_MSG_ID_CACHE_MAX:
                    self._sent_message_ids.popitem(last=False)
            logger.debug(f"Sent {msg_type} to {recv_id}")

    # ── inbound (receive) ──

    def _on_message_sync(self, data: "P2ImMessageReceiveV1") -> None:
        """Called from WS thread → schedule on asyncio loop."""
        if self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(self._on_message(data), self._loop)

    async def _on_message(self, data: "P2ImMessageReceiveV1") -> None:
        try:
            event = data.event
            message = event.message
            sender = event.sender

            # ── dedup ──
            mid = message.message_id
            if mid in self._dedup:
                return
            self._dedup[mid] = None
            while len(self._dedup) > 1000:
                self._dedup.popitem(last=False)

            # Skip messages from bots
            if sender.sender_type == "bot":
                return

            sender_id = (
                sender.sender_id.open_id if sender.sender_id else "unknown"
            )
            chat_id = message.chat_id
            chat_type = message.chat_type  # "p2p" | "group"
            msg_type = message.message_type

            # ── parse content ──
            content = ""
            sender_name = sender_id
            audio_path: str | None = None
            media_files: list[str] = []

            if msg_type == "text":
                try:
                    parsed = json.loads(message.content)
                    content = parsed.get("text", "")
                except (json.JSONDecodeError, TypeError):
                    content = message.content or ""

                # Resolve @_user_N placeholders → @display_name
                if message.mentions:
                    bot_mention_names: list[str] = []
                    for mention in message.mentions:
                        if not mention.key:
                            continue
                        oid = mention.id.open_id if mention.id else ""

                        # Learn sender's display name
                        if oid == sender_id and mention.name:
                            sender_name = mention.name

                        # Replace placeholder with display name
                        if mention.name:
                            content = content.replace(
                                mention.key, f"@{mention.name}"
                            )

                        # Track group member
                        if oid and mention.name and chat_type == "group":
                            if oid != self._bot_open_id:
                                self._members.track_sender(
                                    chat_id, oid, mention.name
                                )

                        # Collect bot mention names for stripping
                        if (
                            self._bot_open_id
                            and oid == self._bot_open_id
                            and mention.name
                        ):
                            bot_mention_names.append(mention.name)

                    # Strip bot @mentions from content
                    for bn in bot_mention_names:
                        content = content.replace(f"@{bn}", "").strip()
            elif msg_type == "post":
                # ── Rich-text post: extract text + download embedded images ──
                content, sender_name, post_image_keys = self._parse_post_content(
                    message, sender_id, sender_name, chat_id, chat_type,
                )
                for ik in post_image_keys:
                    img_path = await self._download_image(mid, ik)
                    if img_path:
                        media_files.append(img_path)
            elif msg_type == "audio":
                # ── Voice message: download audio for STT ──
                try:
                    parsed = json.loads(message.content)
                    file_key = parsed.get("file_key", "")
                    if file_key:
                        audio_path = await self._download_audio(mid, file_key)
                except (json.JSONDecodeError, TypeError):
                    pass

                if audio_path:
                    content = ""  # Will be transcribed by agent loop.
                else:
                    content = "[audio]"
                    audio_path = None
            elif msg_type == "image":
                # ── Image: download for vision model ──
                # Keep content="[image]" so session history records the image
                # was sent; the actual pixels go via media → base64 → vision.
                content = "[image]"
                try:
                    parsed = json.loads(message.content)
                    image_key = parsed.get("image_key", "")
                    if image_key:
                        img_path = await self._download_image(mid, image_key)
                        if img_path:
                            media_files.append(img_path)
                except (json.JSONDecodeError, TypeError):
                    pass
            elif msg_type == "sticker":
                # ── Sticker: download as image for vision ──
                content = "[sticker]"
                try:
                    parsed = json.loads(message.content)
                    file_key = parsed.get("file_key", "")
                    if file_key:
                        img_path = await self._download_image(mid, file_key)
                        if img_path:
                            media_files.append(img_path)
                except (json.JSONDecodeError, TypeError):
                    pass
            elif msg_type == "file":
                # ── File: download and read/describe ──
                content = "[file]"
                try:
                    parsed = json.loads(message.content)
                    file_key = parsed.get("file_key", "")
                    file_name = parsed.get("file_name", "unknown_file")
                    if file_key:
                        fpath = await self._download_file(
                            mid, file_key, file_name
                        )
                        if fpath:
                            suffix = Path(fpath).suffix.lower()
                            if suffix in self._IMAGE_SUFFIXES:
                                media_files.append(fpath)
                                content = f"[file: {file_name}]"
                            else:
                                content = self._read_file_as_text(fpath, file_name)
                except (json.JSONDecodeError, TypeError):
                    pass
            else:
                content = f"[{msg_type}]"

            if not content and not media_files and msg_type != "audio":
                return

            # ── group mention signal (used by agent-side reply policy) ──
            is_voice_msg = msg_type == "audio" and bool(audio_path)
            bot_was_mentioned = False
            reply_to_bot_thread = False
            if chat_type == "group":
                parent_id = str(getattr(message, "parent_id", "") or "")
                reply_to_bot_thread = bool(
                    parent_id and parent_id in self._sent_message_ids
                )
                if message.mentions:
                    for mention in message.mentions:
                        oid = mention.id.open_id if mention.id else ""
                        if self._bot_open_id and oid == self._bot_open_id:
                            bot_was_mentioned = True
                            break

                self._members.track_sender(chat_id, sender_id, sender_name)
                await self._ensure_members(chat_id)
                resolved_sender_name = self._members.display_name(chat_id, sender_id)
                if resolved_sender_name and resolved_sender_name != sender_id:
                    sender_name = resolved_sender_name

                # ── Record message for group stats (fire-and-forget) ──
                try:
                    from getall.stats.group_stats import get_stats_service
                    await get_stats_service().record(
                        chat_id=chat_id,
                        sender_id=sender_id,
                        sender_name=sender_name,
                        bot_mentioned=bot_was_mentioned,
                    )
                except Exception:
                    pass  # Stats are best-effort; never block message flow

                # Group policy: process @mentions OR voice replies to bot.
                if not bot_was_mentioned:
                    if not (is_voice_msg and reply_to_bot_thread):
                        has_mentions = bool(message.mentions)
                        if has_mentions:
                            mention_ids = [
                                (m.id.open_id if m.id else "?")
                                for m in (message.mentions or [])
                            ]
                            logger.debug(
                                f"Group message dropped (bot not mentioned). "
                                f"bot_open_id={self._bot_open_id!r}, "
                                f"mention_ids={mention_ids}"
                            )
                        return

            # Seen indicator: keep it low-noise in groups (mention only)
            if chat_type != "group" or bot_was_mentioned or is_voice_msg:
                # Check for active events to bias reaction emoji selection
                event_hint = ""
                try:
                    from getall.events.manager import EventManager
                    from getall.settings import get_settings
                    settings = get_settings()
                    workspace = Path(settings.workspace_dir).expanduser()
                    _em = EventManager(workspace)
                    _active = _em.get_active_events(
                        message=content or "",
                        user_id=sender_id,
                        chat_id=chat_id,
                    )
                    if _active:
                        event_hint = ", ".join(
                            ae.config.name for ae in _active if not ae.is_game_phase
                        )
                except Exception:
                    pass  # Non-critical — fall back to normal reaction
                emoji_type = await self._choose_reaction_emoji(
                    message_id=mid,
                    msg_type=msg_type,
                    content=content or "(voice)",
                    chat_type=chat_type,
                    event_hint=event_hint,
                )
                await self._add_reaction(mid, emoji_type)

            # ── forward to bus ──
            reply_to = chat_id if chat_type == "group" else sender_id
            # Include group member names so the agent knows who's in the chat
            member_names = (
                self._members.all_names(chat_id) if chat_type == "group" else []
            )
            if member_names:
                logger.debug(f"Passing {len(member_names)} member names to agent: {member_names}")
            media: list[str] = list(media_files)
            if audio_path:
                media.append(audio_path)
            await self._handle_message(
                sender_id=sender_id,
                chat_id=reply_to,
                content=content,
                media=media,
                metadata={
                    "message_id": mid,
                    "chat_type": chat_type,
                    "msg_type": msg_type,
                    "sender_open_id": sender_id,
                    "sender_name": sender_name,
                    "group_members": member_names,
                    "bot_was_mentioned": bot_was_mentioned,
                    "reply_to_bot_thread": reply_to_bot_thread,
                    "is_audio": is_voice_msg,
                    "audio_path": audio_path or "",
                },
                sender_name=sender_name,
                chat_type="group" if chat_type == "group" else "private",
                thread_id=str(getattr(message, "parent_id", "") or ""),
            )
        except Exception as exc:
            logger.error(f"Error processing Lark message: {exc}")
