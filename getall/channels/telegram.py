"""Telegram channel implementation using python-telegram-bot."""

from __future__ import annotations

import asyncio
import re
from typing import TYPE_CHECKING, Any

from loguru import logger
from telegram import BotCommand, Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from telegram.request import HTTPXRequest

from getall.bus.events import OutboundMessage
from getall.bus.queue import MessageBus
from getall.channels.base import BaseChannel
from getall.config.schema import TelegramConfig

if TYPE_CHECKING:
    from getall.session.manager import SessionManager


def _split_into_chunks(text: str) -> list[str]:
    """Split a response into human-sized message chunks.

    Rules:
    - Split on double-newline (paragraph break) first.
    - If every line is very short (≤40 chars) and there are many lines,
      treat each line as its own chunk (covers counting, lists, sequences).
    - Single-paragraph responses stay as one message.
    - Empty chunks are discarded.
    """
    stripped = text.strip()
    if not stripped:
        return []

    lines = stripped.split("\n")

    # Heuristic: if ALL non-empty lines are short, send each as its own message
    # (covers: counting 1-10, numbered lists, bullet sequences)
    non_empty = [ln for ln in lines if ln.strip()]
    if len(non_empty) >= 3 and all(len(ln.strip()) <= 40 for ln in non_empty):
        return [ln.strip() for ln in non_empty]

    # Otherwise split on paragraph boundaries (double newline)
    paragraphs = re.split(r"\n{2,}", stripped)
    chunks = [p.strip() for p in paragraphs if p.strip()]

    # Don't split if there's only 1 paragraph
    if len(chunks) <= 1:
        return [stripped]

    return chunks


def _markdown_to_telegram_html(text: str) -> str:
    """
    Convert markdown to Telegram-safe HTML.
    """
    if not text:
        return ""
    
    # 1. Extract and protect code blocks (preserve content from other processing)
    code_blocks: list[str] = []
    def save_code_block(m: re.Match) -> str:
        code_blocks.append(m.group(1))
        return f"\x00CB{len(code_blocks) - 1}\x00"
    
    text = re.sub(r'```[\w]*\n?([\s\S]*?)```', save_code_block, text)
    
    # 2. Extract and protect inline code
    inline_codes: list[str] = []
    def save_inline_code(m: re.Match) -> str:
        inline_codes.append(m.group(1))
        return f"\x00IC{len(inline_codes) - 1}\x00"
    
    text = re.sub(r'`([^`]+)`', save_inline_code, text)
    
    # 3. Headers # Title -> just the title text
    text = re.sub(r'^#{1,6}\s+(.+)$', r'\1', text, flags=re.MULTILINE)
    
    # 4. Blockquotes > text -> just the text (before HTML escaping)
    text = re.sub(r'^>\s*(.*)$', r'\1', text, flags=re.MULTILINE)
    
    # 5. Escape HTML special characters
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    
    # 6. Links [text](url) - must be before bold/italic to handle nested cases
    text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2">\1</a>', text)
    
    # 7. Bold **text** or __text__
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'__(.+?)__', r'<b>\1</b>', text)
    
    # 8. Italic _text_ (avoid matching inside words like some_var_name)
    text = re.sub(r'(?<![a-zA-Z0-9])_([^_]+)_(?![a-zA-Z0-9])', r'<i>\1</i>', text)
    
    # 9. Strikethrough ~~text~~
    text = re.sub(r'~~(.+?)~~', r'<s>\1</s>', text)
    
    # 10. Bullet lists - item -> • item
    text = re.sub(r'^[-*]\s+', '• ', text, flags=re.MULTILINE)
    
    # 11. Restore inline code with HTML tags
    for i, code in enumerate(inline_codes):
        # Escape HTML in code content
        escaped = code.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        text = text.replace(f"\x00IC{i}\x00", f"<code>{escaped}</code>")
    
    # 12. Restore code blocks with HTML tags
    for i, code in enumerate(code_blocks):
        # Escape HTML in code content
        escaped = code.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        text = text.replace(f"\x00CB{i}\x00", f"<pre><code>{escaped}</code></pre>")
    
    return text


class TelegramChannel(BaseChannel):
    """
    Telegram channel using long polling.
    
    Simple and reliable - no webhook/public IP needed.
    """
    
    name = "telegram"
    
    # Commands registered with Telegram's command menu
    BOT_COMMANDS = [
        BotCommand("start", "Start the bot"),
        BotCommand("reset", "Reset conversation history"),
        BotCommand("help", "Show available commands"),
    ]

    @staticmethod
    def _build_sender_id(user: Any) -> str:
        """Build a stable sender id format used across all handlers."""
        sender_id = str(user.id)
        if getattr(user, "username", None):
            sender_id = f"{sender_id}|{user.username}"
        return sender_id
    
    def __init__(
        self,
        config: TelegramConfig,
        bus: MessageBus,
        groq_api_key: str = "",
        session_manager: SessionManager | None = None,
    ):
        super().__init__(config, bus)
        self.config: TelegramConfig = config
        self.groq_api_key = groq_api_key
        self.session_manager = session_manager
        self._app: Application | None = None
        self._chat_ids: dict[str, int] = {}  # Map sender_id to chat_id for replies
        self._typing_tasks: dict[str, asyncio.Task] = {}  # chat_id -> typing loop task
    
    async def start(self) -> None:
        """Start the Telegram bot with long polling."""
        if not self.config.token:
            logger.error("Telegram bot token not configured")
            return
        
        self._running = True
        
        # Build the application with larger connection pool to avoid pool-timeout on long runs
        req = HTTPXRequest(connection_pool_size=16, pool_timeout=5.0, connect_timeout=30.0, read_timeout=30.0)
        builder = Application.builder().token(self.config.token).request(req).get_updates_request(req)
        if self.config.proxy:
            builder = builder.proxy(self.config.proxy).get_updates_proxy(self.config.proxy)
        self._app = builder.build()
        self._app.add_error_handler(self._on_error)
        
        # Add command handlers
        self._app.add_handler(CommandHandler("start", self._on_start))
        self._app.add_handler(CommandHandler("reset", self._on_reset))
        self._app.add_handler(CommandHandler("help", self._on_help))
        
        # Add message handler for text, photos, voice, documents
        self._app.add_handler(
            MessageHandler(
                (filters.TEXT | filters.PHOTO | filters.VOICE | filters.AUDIO | filters.Document.ALL) 
                & ~filters.COMMAND, 
                self._on_message
            )
        )
        
        logger.info("Starting Telegram bot (polling mode)...")
        
        # Initialize and start polling
        await self._app.initialize()
        await self._app.start()
        
        # Get bot info and register command menu
        bot_info = await self._app.bot.get_me()
        logger.info(f"Telegram bot @{bot_info.username} connected")
        
        try:
            await self._app.bot.set_my_commands(self.BOT_COMMANDS)
            logger.debug("Telegram bot commands registered")
        except Exception as e:
            logger.warning(f"Failed to register bot commands: {e}")
        
        # Start polling (this runs until stopped)
        await self._app.updater.start_polling(
            allowed_updates=["message"],
            drop_pending_updates=True  # Ignore old messages on startup
        )
        
        # Keep running until stopped
        while self._running:
            await asyncio.sleep(1)
    
    async def stop(self) -> None:
        """Stop the Telegram bot."""
        self._running = False
        
        # Cancel all typing indicators
        for chat_id in list(self._typing_tasks):
            self._stop_typing(chat_id)
        
        if self._app:
            logger.info("Stopping Telegram bot...")
            await self._app.updater.stop()
            await self._app.stop()
            await self._app.shutdown()
            self._app = None
    
    async def send(self, msg: OutboundMessage) -> None:
        """Send a message through Telegram.

        Long responses are split into chunks (by double-newline paragraphs)
        and sent one by one with a short typing delay to feel more human.
        """
        if not self._app:
            logger.warning("Telegram bot not running")
            return

        # Stop typing indicator for this chat
        self._stop_typing(msg.chat_id)

        try:
            chat_id = int(msg.chat_id)
        except ValueError:
            logger.error(f"Invalid chat_id: {msg.chat_id}")
            return

        chunks = _split_into_chunks(msg.content)

        for i, chunk in enumerate(chunks):
            # Typing delay between chunks to feel human
            if i > 0:
                delay = min(0.3 + len(chunk) * 0.01, 2.0)  # 0.3–2s based on length
                try:
                    await self._app.bot.send_chat_action(chat_id=chat_id, action="typing")
                except Exception:
                    pass
                await asyncio.sleep(delay)

            await self._send_single(chat_id, chunk)

    async def _send_single(self, chat_id: int, text: str) -> None:
        """Send one text chunk to Telegram, with HTML fallback."""
        try:
            html = _markdown_to_telegram_html(text)
            await self._app.bot.send_message(
                chat_id=chat_id, text=html, parse_mode="HTML",
            )
        except Exception as e:
            logger.warning(f"HTML parse failed, falling back to plain text: {e}")
            try:
                await self._app.bot.send_message(chat_id=chat_id, text=text)
            except Exception as e2:
                logger.error(f"Error sending Telegram message: {e2}")
    
    async def _on_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /start command."""
        if not update.message or not update.effective_user:
            return

        raw = (update.message.text or "").strip()
        payload = re.sub(r"^/start(?:@\w+)?", "", raw, count=1).strip()
        user = update.effective_user
        sender_id = self._build_sender_id(user)
        chat_type = "private" if update.message.chat.type == "private" else "group"

        # If the user typed a command + task in one message, execute the task directly.
        if payload:
            await self._handle_message(
                sender_id=sender_id,
                chat_id=str(update.message.chat_id),
                content=payload,
                sender_name=user.first_name or "",
                chat_type=chat_type,
            )
            return

        # Let the agent generate a natural start message based on onboarded state.
        start_event_prompt = (
            "System event: the user just sent /start. "
            "Respond strictly from your real current state and avoid contradictions. "
            "If onboarded=true, greet in a familiar tone and move directly to actionable next steps. "
            "Do NOT proactively repeat your name, IFT, or adoption flow unless the user explicitly asks. "
            "If onboarded=false, follow the onboarding flow: first ask if they have an existing IFT from another platform, "
            "then proceed step by step (do NOT rush all steps into one message). "
            "Keep this first reply short — just the IFT question and a warm greeting."
        )
        await self._handle_message(
            sender_id=sender_id,
            chat_id=str(update.message.chat_id),
            content=start_event_prompt,
            sender_name=user.first_name or "",
            chat_type=chat_type,
            metadata={"event": "start_command", "synthetic": True},
        )
    
    async def _on_reset(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /reset command — clear ALL matching sessions + reset persona."""
        if not update.message or not update.effective_user:
            return

        chat_id = str(update.message.chat_id)
        user_id = str(update.effective_user.id)
        sender_full = self._build_sender_id(update.effective_user)
        cleared = 0

        # ── 1. Clear file-based sessions (both old and new key formats) ──
        if self.session_manager is not None:
            # Brute-force: scan all session files whose name contains the chat_id
            for path in self.session_manager.sessions_dir.glob("*.jsonl"):
                if chat_id in path.name or user_id in path.name or sender_full in path.name:
                    try:
                        path.unlink()
                        cleared += 1
                    except Exception:
                        pass
            # Also flush the in-memory cache entries that match
            keys_to_drop = [
                k for k in list(self.session_manager._cache)
                if chat_id in k or user_id in k or sender_full in k
            ]
            for k in keys_to_drop:
                self.session_manager._cache.pop(k, None)

        # ── 2. Reset persona in DB so the next message triggers fresh onboarding ──
        try:
            from getall.storage.database import get_session_factory
            from getall.storage.repository import IdentityRepo
            factory = get_session_factory()
            async with factory() as db_session:
                repo = IdentityRepo(db_session)
                p = await repo.get_by_external(self.name, sender_full)
                if p is None:
                    # Backward-compat for older bindings that used plain numeric user id
                    p = await repo.get_by_external(self.name, user_id)
                if p is not None:
                    await repo.update_persona(
                        p.id,
                        pet_name="",
                        persona_text="",
                        trading_style_text="",
                        onboarded=False,
                    )
                    await db_session.commit()
                    logger.info(f"/reset: persona cleared for principal {p.id}")
        except Exception as e:
            logger.warning(f"/reset: failed to reset persona in DB: {e}")

        logger.info(f"/reset for telegram:{chat_id} — cleared {cleared} session files")
        await update.message.reply_text("🔄 Conversation history cleared. Let's start fresh!")
    
    async def _on_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle /help command — show available commands."""
        if not update.message:
            return
        
        help_text = (
            "🐾 <b>GetAll Commands</b>\n\n"
            "/start — Restart the bot\n"
            "/reset — Clear conversation history\n"
            "/help — Show this help\n\n"
            "No need to memorize commands — just talk to me naturally!\n"
            'e.g. "Check BTC", "Long ETH", "Review yesterday"'
        )
        await update.message.reply_text(help_text, parse_mode="HTML")
    
    async def _on_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle incoming messages (text, photos, voice, documents)."""
        if not update.message or not update.effective_user:
            return
        
        message = update.message
        user = update.effective_user
        chat_id = message.chat_id
        
        sender_id = self._build_sender_id(user)
        
        # Store chat_id for replies
        self._chat_ids[sender_id] = chat_id
        
        # Build content from text and/or media
        content_parts = []
        media_paths = []
        
        # Text content
        if message.text:
            content_parts.append(message.text)
        if message.caption:
            content_parts.append(message.caption)
        
        # Handle media files
        media_file = None
        media_type = None
        
        if message.photo:
            media_file = message.photo[-1]  # Largest photo
            media_type = "image"
        elif message.voice:
            media_file = message.voice
            media_type = "voice"
        elif message.audio:
            media_file = message.audio
            media_type = "audio"
        elif message.document:
            media_file = message.document
            media_type = "file"
        
        # Download media if present
        if media_file and self._app:
            try:
                file = await self._app.bot.get_file(media_file.file_id)
                ext = self._get_extension(media_type, getattr(media_file, 'mime_type', None))
                
                # Save to workspace/media/
                from pathlib import Path
                media_dir = Path.home() / ".getall" / "media"
                media_dir.mkdir(parents=True, exist_ok=True)
                
                file_path = media_dir / f"{media_file.file_id[:16]}{ext}"
                await file.download_to_drive(str(file_path))
                
                media_paths.append(str(file_path))
                
                # Handle voice transcription
                if media_type == "voice" or media_type == "audio":
                    from getall.providers.transcription import GroqTranscriptionProvider
                    transcriber = GroqTranscriptionProvider(api_key=self.groq_api_key)
                    transcription = await transcriber.transcribe(file_path)
                    if transcription:
                        logger.info(f"Transcribed {media_type}: {transcription[:50]}...")
                        content_parts.append(f"[transcription: {transcription}]")
                    else:
                        content_parts.append(f"[{media_type}: {file_path}]")
                else:
                    content_parts.append(f"[{media_type}: {file_path}]")
                    
                logger.debug(f"Downloaded {media_type} to {file_path}")
            except Exception as e:
                logger.error(f"Failed to download media: {e}")
                content_parts.append(f"[{media_type}: download failed]")
        
        content = "\n".join(content_parts) if content_parts else "[empty message]"
        
        logger.debug(f"Telegram message from {sender_id}: {content[:50]}...")
        
        str_chat_id = str(chat_id)
        
        # Start typing indicator before processing
        self._start_typing(str_chat_id)
        
        # Forward to the message bus
        await self._handle_message(
            sender_id=sender_id,
            chat_id=str_chat_id,
            content=content,
            media=media_paths,
            metadata={
                "message_id": message.message_id,
                "user_id": user.id,
                "username": user.username,
                "first_name": user.first_name,
                "is_group": message.chat.type != "private"
            },
            sender_name=(user.full_name or user.first_name or user.username or ""),
            chat_type="private" if message.chat.type == "private" else "group",
            thread_id=str(getattr(message, "message_thread_id", "") or ""),
        )
    
    def _start_typing(self, chat_id: str) -> None:
        """Start sending 'typing...' indicator for a chat."""
        # Cancel any existing typing task for this chat
        self._stop_typing(chat_id)
        self._typing_tasks[chat_id] = asyncio.create_task(self._typing_loop(chat_id))
    
    def _stop_typing(self, chat_id: str) -> None:
        """Stop the typing indicator for a chat."""
        task = self._typing_tasks.pop(chat_id, None)
        if task and not task.done():
            task.cancel()
    
    async def _typing_loop(self, chat_id: str) -> None:
        """Repeatedly send 'typing' action until cancelled."""
        try:
            while self._app:
                await self._app.bot.send_chat_action(chat_id=int(chat_id), action="typing")
                await asyncio.sleep(4)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.debug(f"Typing indicator stopped for {chat_id}: {e}")
    
    async def _on_error(self, update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Log polling / handler errors instead of silently swallowing them."""
        logger.error(f"Telegram error: {context.error}")

    def _get_extension(self, media_type: str, mime_type: str | None) -> str:
        """Get file extension based on media type."""
        if mime_type:
            ext_map = {
                "image/jpeg": ".jpg", "image/png": ".png", "image/gif": ".gif",
                "audio/ogg": ".ogg", "audio/mpeg": ".mp3", "audio/mp4": ".m4a",
            }
            if mime_type in ext_map:
                return ext_map[mime_type]
        
        type_map = {"image": ".jpg", "voice": ".ogg", "audio": ".mp3", "file": ""}
        return type_map.get(media_type, "")
