"""Feishu/Lark channel – WebSocket long-connection + HTTP webhook support.

Supports:
- Private chat (p2p): user talks to bot directly
- Group chat: reply only when bot is @mentioned
- Auto @mention sender in group replies
- Resolve @name references in LLM output to real Lark <at> tags
"""

import asyncio
import hashlib
import json
import re
import threading
import time as _time
from collections import OrderedDict
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

    # ── helpers ──

    def _domain(self) -> str:
        if self.config.domain.lower() == "lark":
            return lark.LARK_DOMAIN  # https://open.larksuite.com
        return lark.FEISHU_DOMAIN  # https://open.feishu.cn

    def _domain_label(self) -> str:
        return "Lark" if self.config.domain.lower() == "lark" else "Feishu"

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

        while self._running:
            await asyncio.sleep(1)

    async def stop(self) -> None:
        self._running = False
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

    _AT_RE = re.compile(r"@([\w\u4e00-\u9fff]+(?:\s[\w\u4e00-\u9fff]+){0,2})")

    def _resolve_at_tags(self, chat_id: str, text: str) -> str:
        """Replace @name in LLM output with Lark <at> markdown tags."""

        def _replace(m: re.Match[str]) -> str:
            name = m.group(1)
            oid = self._members.resolve_name(chat_id, name)
            if oid:
                return f"<at id={oid}>{name}</at>"
            return m.group(0)

        return self._AT_RE.sub(_replace, text)

    def _prepend_sender_at(
        self, chat_id: str, sender_open_id: str, text: str
    ) -> str:
        name = self._members.display_name(chat_id, sender_open_id)
        return f"<at id={sender_open_id}>{name}</at> {text}"

    # ── reaction ──

    _REACTION_EMOJI_CHOICES: tuple[str, ...] = FEISHU_REACTION_EMOJI_CHOICES
    _REACTION_FALLBACK_TEXT: tuple[str, ...] = (
        "THUMBSUP",
        "SMILE",
        "JIAYI",
        "DONE",
        "PARTY",
        "FIRE",
        "HEART",
        "WAVE",
        "WOW",
        "THINKING",
    )
    _REACTION_FALLBACK_NON_TEXT: tuple[str, ...] = (
        "THUMBSUP",
        "SMILE",
        "JIAYI",
        "WOW",
    )

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

    @staticmethod
    def _stable_pick(options: tuple[str, ...], seed: str) -> str:
        if not options:
            return "THUMBSUP"
        digest = hashlib.sha1(seed.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], "big") % len(options)
        return options[idx]

    def _fallback_reaction_emoji(
        self,
        message_id: str,
        msg_type: str,
        content: str,
        chat_type: str,
    ) -> str:
        text = (content or "").strip()
        seed = f"{chat_type}|{msg_type}|{message_id}|{text[:120]}"
        if msg_type != "text":
            return self._stable_pick(self._REACTION_FALLBACK_NON_TEXT, seed)
        if text.endswith(("?", "？")):
            return self._stable_pick(("THINKING", "WOW", "SMILE"), seed)
        if len(text) <= 8:
            return self._stable_pick(("SMILE", "WAVE", "THUMBSUP"), seed)
        return self._stable_pick(self._REACTION_FALLBACK_TEXT, seed)

    async def _choose_reaction_emoji(
        self,
        message_id: str,
        msg_type: str,
        content: str,
        chat_type: str,
    ) -> str:
        fallback = self._fallback_reaction_emoji(
            message_id=message_id,
            msg_type=msg_type,
            content=content,
            chat_type=chat_type,
        )
        self._init_reaction_llm()
        if not self._reaction_llm:
            return fallback

        content_preview = (content or "").strip()
        if len(content_preview) > 240:
            content_preview = f"{content_preview[:240]}..."

        allowed = ", ".join(self._REACTION_EMOJI_CHOICES)
        messages = [
            {
                "role": "system",
                "content": (
                    "You pick ONE Lark reaction emoji_type for a user message.\n"
                    "Return exactly one token from the allowed list only.\n"
                    "No explanation, no punctuation, no markdown."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"allowed: {allowed}\n"
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
                    model=self._reaction_model,
                    max_tokens=16,
                    temperature=0.1,
                    reasoning_effort=self._reaction_reasoning_effort,
                ),
                timeout=4.0,
            )
            raw = (response.content or "").strip()
            match = re.search(r"[A-Za-z0-9_]+", raw)
            candidate = match.group(0) if match else ""
            if candidate in self._REACTION_EMOJI_CHOICES:
                return candidate

            # Be tolerant to case mismatches from model output while
            # still returning the documented canonical emoji_type token.
            lowered = candidate.lower()
            if lowered:
                case_insensitive_hits = [
                    name
                    for name in self._REACTION_EMOJI_CHOICES
                    if name.lower() == lowered
                ]
                if len(case_insensitive_hits) == 1:
                    return case_insensitive_hits[0]
            logger.warning(
                f"Reaction LLM returned invalid emoji_type '{raw}', using fallback={fallback}"
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

    _TABLE_RE = re.compile(
        r"((?:^[ \t]*\|.+\|[ \t]*\n)"
        r"(?:^[ \t]*\|[-:\s|]+\|[ \t]*\n)"
        r"(?:^[ \t]*\|.+\|[ \t]*\n?)+)",
        re.MULTILINE,
    )

    @staticmethod
    def _parse_md_table(table_text: str) -> dict | None:
        lines = [ln.strip() for ln in table_text.strip().split("\n") if ln.strip()]
        if len(lines) < 3:
            return None
        _split = lambda ln: [c.strip() for c in ln.strip("|").split("|")]
        headers = _split(lines[0])
        rows = [_split(ln) for ln in lines[2:]]
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

    def _build_card_elements(self, content: str) -> list[dict]:
        elements: list[dict] = []
        last = 0
        for m in self._TABLE_RE.finditer(content):
            before = content[last : m.start()].strip()
            if before:
                elements.append({"tag": "markdown", "content": before})
            tbl = self._parse_md_table(m.group(1))
            elements.append(
                tbl or {"tag": "markdown", "content": m.group(1)}
            )
            last = m.end()
        tail = content[last:].strip()
        if tail:
            elements.append({"tag": "markdown", "content": tail})
        return elements or [{"tag": "markdown", "content": content}]

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

    # Match: [GENERATED_IMAGE:/path], ![alt](/path), bare /path/to/file.png
    _GENERATED_TAG_RE = re.compile(r"\[GENERATED_IMAGE:(.*?)\]")
    _IMG_PATH_RE = re.compile(
        r"(?:!\[.*?\]\()?"
        r"(/[^\s\)\]\"'<>]+\.(?:png|jpg|jpeg|gif|webp|bmp))"
        r"\)?",
        re.IGNORECASE,
    )

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

        # 2. Bare paths / markdown image syntax
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

    # ── outbound (send) ──

    @staticmethod
    def _needs_card(text: str, has_images: bool, has_actions: bool) -> bool:
        """Decide whether *text* is rich enough to warrant an interactive card.

        Plain text messages look compact (no wide card chrome), so we prefer
        them for short / simple chunks.  Use a card when:
        - the chunk contains a markdown table
        - the chunk will carry embedded images
        - the chunk includes actionable buttons
        - the chunk uses markdown features that benefit from card rendering
          (bold, links, code blocks, bullet lists, etc.)
        """
        if has_images:
            return True
        if has_actions:
            return True
        # <at> mention tags only render properly in cards, not plain text
        if "<at " in text:
            return True
        # Markdown table
        if re.search(r"^\s*\|.+\|", text, re.MULTILINE):
            return True
        # Rich markdown: bold, italic, links, code blocks, bullet/numbered lists
        if re.search(
            r"\*\*.+?\*\*|__.*?__|`.+?`|```|\[.+?\]\(.+?\)|^[-*]\s|^\d+\.\s",
            text,
            re.MULTILINE,
        ):
            return True
        return False

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

            # Prepend @sender for group chats
            if is_group:
                sender_oid = (msg.metadata or {}).get("sender_open_id", "")
                if sender_oid:
                    content = self._prepend_sender_at(
                        msg.chat_id, sender_oid, content
                    )

            # Send as ONE message: card if rich content, plain text otherwise.
            if self._needs_card(
                content, bool(image_paths), bool(action_items)
            ):
                elements = self._build_card_elements(content)
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
                card = json.dumps(
                    {"config": {"wide_screen_mode": True}, "elements": elements},
                    ensure_ascii=False,
                )
                await self._send_msg(recv_type, msg.chat_id, "interactive", card)
            else:
                text_body = json.dumps(
                    {"text": content}, ensure_ascii=False
                )
                await self._send_msg(recv_type, msg.chat_id, "text", text_body)

        except Exception as exc:
            logger.error(f"Error sending Lark message: {exc}")

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
            logger.error(
                f"Send failed: code={resp.code}, msg={resp.msg}, "
                f"log_id={resp.get_log_id()}"
            )
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

                        # Detect bot mention (sender_type won't tell us,
                        # so we learn: if oid != sender and oid is unknown
                        # to member cache, treat first such as bot)
                        if (
                            not self._bot_open_id
                            and oid
                            and oid != sender_id
                            and mention.name
                        ):
                            # Heuristic: a non-sender mention that isn't
                            # tracked as a human is likely the bot.
                            if not self._members.resolve_name(
                                chat_id, mention.name
                            ):
                                self._bot_open_id = oid
                                logger.debug(f"Learned bot open_id: {oid}")

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
            else:
                content = _MSG_TYPE_MAP.get(msg_type, f"[{msg_type}]")

            if not content and msg_type != "audio":
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
                        # Before we know bot_open_id, check if mention
                        # is a non-sender, non-member (likely the bot)
                        if (
                            not self._bot_open_id
                            and oid
                            and oid != sender_id
                        ):
                            bot_was_mentioned = True
                            break

                self._members.track_sender(chat_id, sender_id, sender_name)
                await self._ensure_members(chat_id)
                resolved_sender_name = self._members.display_name(chat_id, sender_id)
                if resolved_sender_name and resolved_sender_name != sender_id:
                    sender_name = resolved_sender_name

                # Group policy: process @mentions OR voice replies to bot.
                if not bot_was_mentioned:
                    if not (is_voice_msg and reply_to_bot_thread):
                        return

            # Seen indicator: keep it low-noise in groups (mention only)
            if chat_type != "group" or bot_was_mentioned or is_voice_msg:
                emoji_type = await self._choose_reaction_emoji(
                    message_id=mid,
                    msg_type=msg_type,
                    content=content or "(voice)",
                    chat_type=chat_type,
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
            media: list[str] = []
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
