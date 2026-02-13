"""Feishu/Lark channel – WebSocket long-connection + HTTP webhook support.

Supports:
- Private chat (p2p): user talks to bot directly
- Group chat: bot responds to ALL messages (no @mention required)
- Auto @mention sender in group replies
- Resolve @name references in LLM output to real Lark <at> tags
"""

import asyncio
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
from getall.config.schema import FeishuConfig

try:
    import lark_oapi as lark
    from lark_oapi.api.im.v1 import (
        CreateMessageRequest,
        CreateMessageRequestBody,
        CreateMessageReactionRequest,
        CreateMessageReactionRequestBody,
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


# ---------------------------------------------------------------------------
# Group member cache – for @mention resolution
# ---------------------------------------------------------------------------

class _GroupMemberCache:
    """In-memory cache: chat_id → {name_lower: open_id} + {open_id: name}."""

    def __init__(self) -> None:
        self._by_name: dict[str, dict[str, str]] = {}
        self._by_id: dict[str, dict[str, str]] = {}
        self._ts: dict[str, float] = {}

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
        self._by_name.setdefault(chat_id, {})[name.lower()] = open_id
        self._by_id.setdefault(chat_id, {})[open_id] = name


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
            # No-op handlers for events we don't process but must ACK
            def _noop_bot_entered(data: Any) -> None:
                logger.debug("Lark event: bot_p2p_chat_entered (ignored)")

            def _noop_bot_added(data: Any) -> None:
                logger.debug("Lark event: bot_added_to_chat (ignored)")

            def _noop_bot_deleted(data: Any) -> None:
                logger.debug("Lark event: bot_deleted_from_chat (ignored)")

            self._event_handler = (
                lark.EventDispatcherHandler.builder(
                    self.config.encrypt_key or "",
                    self.config.verification_token or "",
                    lark.LogLevel.DEBUG,
                )
                .register_p2_im_message_receive_v1(self._on_message_sync)
                .register_p2_im_chat_access_event_bot_p2p_chat_entered_v1(_noop_bot_entered)
                .register_p2_im_chat_member_bot_added_v1(_noop_bot_added)
                .register_p2_im_chat_member_bot_deleted_v1(_noop_bot_deleted)
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

    # ── outbound (send) ──

    async def send(self, msg: OutboundMessage) -> None:
        if not self._client:
            logger.warning("Lark client not initialised")
            return

        try:
            is_group = msg.chat_id.startswith("oc_")
            recv_type = "chat_id" if is_group else "open_id"

            content = msg.content

            if is_group:
                # Resolve @name → <at> tags
                content = self._resolve_at_tags(msg.chat_id, content)
                # Prepend @sender
                sender_oid = (msg.metadata or {}).get("sender_open_id", "")
                if sender_oid:
                    content = self._prepend_sender_at(
                        msg.chat_id, sender_oid, content
                    )

            # Build card elements (text + tables)
            elements = self._build_card_elements(content)

            # Upload and embed images from media
            if msg.media:
                for path in msg.media:
                    img_key = await self._upload_image(path)
                    if img_key:
                        elements.append({
                            "tag": "img",
                            "img_key": img_key,
                            "alt": {
                                "tag": "plain_text",
                                "content": Path(path).stem,
                            },
                        })

            card = json.dumps(
                {"config": {"wide_screen_mode": True}, "elements": elements},
                ensure_ascii=False,
            )

            req = (
                CreateMessageRequest.builder()
                .receive_id_type(recv_type)
                .request_body(
                    CreateMessageRequestBody.builder()
                    .receive_id(msg.chat_id)
                    .msg_type("interactive")
                    .content(card)
                    .build()
                )
                .build()
            )
            resp = self._client.im.v1.message.create(req)

            if not resp.success():
                logger.error(
                    f"Send failed: code={resp.code}, msg={resp.msg}, "
                    f"log_id={resp.get_log_id()}"
                )
            else:
                logger.debug(f"Sent to {msg.chat_id}")
        except Exception as exc:
            logger.error(f"Error sending Lark message: {exc}")

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
            else:
                content = _MSG_TYPE_MAP.get(msg_type, f"[{msg_type}]")

            if not content:
                return

            # ── group: only respond when bot is @mentioned ──
            if chat_type == "group":
                bot_was_mentioned = False
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

                if not bot_was_mentioned:
                    # Still track sender for member cache, but don't respond
                    self._members.track_sender(chat_id, sender_id, sender_name)
                    return

                self._members.track_sender(chat_id, sender_id, sender_name)
                asyncio.create_task(self._ensure_members(chat_id))

            # Seen indicator (only for messages we'll respond to)
            await self._add_reaction(mid, "THUMBSUP")

            # ── forward to bus ──
            reply_to = chat_id if chat_type == "group" else sender_id
            await self._handle_message(
                sender_id=sender_id,
                chat_id=reply_to,
                content=content,
                metadata={
                    "message_id": mid,
                    "chat_type": chat_type,
                    "msg_type": msg_type,
                    "sender_open_id": sender_id,
                    "sender_name": sender_name,
                },
                sender_name=sender_name,
                chat_type="group" if chat_type == "group" else "private",
                thread_id=str(getattr(message, "parent_id", "") or ""),
            )
        except Exception as exc:
            logger.error(f"Error processing Lark message: {exc}")
