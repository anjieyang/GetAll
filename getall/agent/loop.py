"""Agent loop: the core processing engine."""

import asyncio
import json
import re
from pathlib import Path
from typing import Any

from loguru import logger

from getall.bus.events import InboundMessage, OutboundMessage
from getall.bus.queue import MessageBus
from getall.providers.base import LLMProvider
from getall.providers.voice import (
    OpenAIVoiceProvider,
    strip_markdown_for_tts,
    is_code_heavy,
    parse_voice_directive,
    build_voice_options_hint,
    TTS_MAX_TEXT_LENGTH,
)
from getall.agent.context import ContextBuilder
from getall.agent.tools.registry import ToolRegistry
from getall.agent.tools.filesystem import ReadFileTool, WriteFileTool, EditFileTool, ListDirTool
from getall.agent.tools.shell import ExecTool
from getall.agent.tools.web import WebSearchTool, WebFetchTool
from getall.agent.tools.message import MessageTool
from getall.agent.tools.spawn import SpawnTool
from getall.agent.tools.cron import ReminderTool
from getall.agent.tools.workbench import WorkbenchTool
from getall.agent.tools.bitget import (
    BitgetAccountTool,
    BitgetMarketTool,
    BitgetTradeTool,
    BitgetUtaTool,
)
from getall.agent.tools.coingecko import CoinGeckoTool
from getall.agent.tools.credential import CredentialTool
from getall.agent.tools.defillama import DefiLlamaTool
from getall.agent.tools.fear_greed import FearGreedTool
from getall.agent.tools.finnhub import FinnhubTool
from getall.agent.tools.freecrypto import FreeCryptoTool
from getall.agent.tools.group_stats import GroupStatsTool
from getall.agent.tools.persona import PetPersonaTool
from getall.agent.tools.yfinance_tool import YFinanceTool
from getall.trading.data.hub import DataHub
from getall.agent.memory import MemoryStore
from getall.agent.subagent import SubagentManager
from getall.session.manager import Session, SessionManager, save_handoff, consume_handoff

# ── Voice mode triggers (substring match) ──
# System prompt — voice capability section (always injected).
_VOICE_SECTION_TEMPLATE = """\n
## Voice Capability

You have built-in voice (TTS/STT). Default mode: **text (图文)**.

### How mode works
- **Default is text**: Always reply with normal text unless the user explicitly asks for voice.
- **voice mode ON**: Your text reply is automatically converted to speech audio and sent as a voice message.
- Even if the user sends a voice message, reply with text unless they explicitly say they want a voice reply.

### You control the mode — per-reply only (not sticky)
Add one of these tags in your reply to enable voice **for this reply only**:
- `[[voice:on]]` — reply with voice this time
- `[[voice:coral]]` — reply with voice using a specific voice
- `[[voice:cedar instructions=用温柔的语气]]` — voice + style control

**When to use voice**: ONLY when the user explicitly requests voice reply, e.g. "用语音回复", "语音说", "speak to me", "用语音和我说".
**Otherwise**: Always use text. Do NOT enable voice just because the user sent a voice message.
**Use your intelligence** — understand the user's intent from context, not keyword matching.

{voice_active_section}
{voice_options}
"""

_VOICE_ACTIVE_TIPS = """\
### Voice mode tips (currently ON)
- Keep replies concise and conversational (≤300 chars ideal).
- Avoid code blocks, tables, and complex formatting.
- Speak naturally as if having a face-to-face conversation.
- Use punctuation for natural pauses in speech.
- If the answer requires code or long text, reply normally — the system auto-skips TTS for those.
- Pick a voice that fits the mood by adding `[[voice:name]]` in your reply.
"""


class AgentLoop:
    """
    The agent loop is the core processing engine.
    
    It:
    1. Receives messages from the bus
    2. Builds context with history, memory, skills
    3. Calls the LLM
    4. Executes tool calls
    5. Sends responses back
    """

    _NO_REPLY_RE = re.compile(
        r"^\s*(?:\[\s*NO_REPLY\s*\]|<\s*NO_REPLY\s*>|NO_REPLY)\s*$",
        re.IGNORECASE,
    )
    
    def __init__(
        self,
        bus: MessageBus,
        provider: LLMProvider,
        workspace: Path,
        model: str | None = None,
        max_tokens: int = 65536,
        max_iterations: int = 20,
        memory_window: int = 50,
        brave_api_key: str | None = None,
        web_search_provider: str = "brave",
        web_search_api_key: str | None = None,
        web_search_openai_api_key: str | None = None,
        web_search_openai_model: str = "gpt-4o-mini",
        exec_config: "ExecToolConfig | None" = None,
        cron_service: "CronService | None" = None,
        restrict_to_workspace: bool = False,
        session_manager: SessionManager | None = None,
        trading_config: "TradingConfig | None" = None,
        max_concurrent_workers: int = 4,
        reasoning_effort: str = "",
    ):
        from getall.config.schema import ExecToolConfig, TradingConfig
        from getall.cron.service import CronService
        self.bus = bus
        self.provider = provider
        self.workspace = workspace
        self.model = model or provider.get_default_model()
        self.max_tokens = max_tokens
        self.max_iterations = max_iterations
        self.memory_window = memory_window
        self.brave_api_key = brave_api_key
        self.web_search_provider = web_search_provider
        self.web_search_api_key = (
            web_search_api_key if web_search_api_key is not None else brave_api_key
        )
        self.web_search_openai_api_key = web_search_openai_api_key
        self.web_search_openai_model = web_search_openai_model
        self.exec_config = exec_config or ExecToolConfig()
        self.cron_service = cron_service
        self.restrict_to_workspace = restrict_to_workspace
        self.trading_config = trading_config or TradingConfig()
        self.max_concurrent_workers = max(1, int(max_concurrent_workers))
        self.reasoning_effort = reasoning_effort

        # Voice provider (lazy: reads GETALL_OPENAI_API_KEY from env).
        self.voice = OpenAIVoiceProvider()
        
        self.context = ContextBuilder(workspace)
        self.sessions = session_manager or SessionManager(workspace)
        self.tools = ToolRegistry()
        self.subagents = SubagentManager(
            provider=provider,
            workspace=workspace,
            bus=bus,
            model=self.model,
            max_tokens=max_tokens,
            brave_api_key=brave_api_key,
            web_search_provider=self.web_search_provider,
            web_search_api_key=self.web_search_api_key,
            web_search_openai_api_key=self.web_search_openai_api_key,
            web_search_openai_model=self.web_search_openai_model,
            exec_config=self.exec_config,
            restrict_to_workspace=restrict_to_workspace,
            reasoning_effort=reasoning_effort,
        )
        
        self._running = False
        self._worker_tasks: list[asyncio.Task[None]] = []
        self._session_locks: dict[str, asyncio.Lock] = {}
        self._session_lock_refcounts: dict[str, int] = {}
        self._session_lock_guard = asyncio.Lock()
        self._register_default_tools()
    
    def _register_default_tools(self) -> None:
        """Register the default set of tools."""
        # File tools (restrict to workspace if configured)
        allowed_dir = self.workspace if self.restrict_to_workspace else None
        protected_paths = {self.workspace / "SOUL.md"}
        self.tools.register(ReadFileTool(allowed_dir=allowed_dir))
        self.tools.register(
            WriteFileTool(
                allowed_dir=allowed_dir,
                protected_paths=protected_paths,
            )
        )
        self.tools.register(
            EditFileTool(
                allowed_dir=allowed_dir,
                protected_paths=protected_paths,
            )
        )
        self.tools.register(ListDirTool(allowed_dir=allowed_dir))
        
        # Shell tool
        self.tools.register(ExecTool(
            working_dir=str(self.workspace),
            timeout=self.exec_config.timeout,
            restrict_to_workspace=self.restrict_to_workspace,
        ))
        self.tools.register(WorkbenchTool(
            workspace=self.workspace,
            timeout=max(self.exec_config.timeout, 120),
        ))
        
        # Web tools
        self.tools.register(
            WebSearchTool(
                provider=self.web_search_provider,
                api_key=self.web_search_api_key,
                openai_api_key=self.web_search_openai_api_key,
                openai_model=self.web_search_openai_model,
            )
        )
        self.tools.register(WebFetchTool())
        
        # Message tool (with cross-chat handoff for DM context injection)
        message_tool = MessageTool(
            send_callback=self.bus.publish_outbound,
            handoff_callback=save_handoff,
        )
        self.tools.register(message_tool)
        
        # Spawn tool (for subagents)
        spawn_tool = SpawnTool(manager=self.subagents)
        self.tools.register(spawn_tool)
        
        # Cron tool (for scheduling)
        if self.cron_service:
            self.tools.register(ReminderTool(self.cron_service))

        # Bitget trading tools
        self.tools.register(BitgetMarketTool())
        self._bitget_account_tool = BitgetAccountTool()
        self.tools.register(self._bitget_account_tool)
        self._bitget_trade_tool = BitgetTradeTool()
        self.tools.register(self._bitget_trade_tool)
        self._bitget_uta_tool = BitgetUtaTool()
        self.tools.register(self._bitget_uta_tool)

        # Credential tool (save / check / delete exchange API keys)
        self._credential_tool = CredentialTool()
        self.tools.register(self._credential_tool)

        # Pet persona tool
        self._persona_tool = PetPersonaTool()
        self.tools.register(self._persona_tool)

        # Group chat statistics tool
        self._group_stats_tool = GroupStatsTool()
        self.tools.register(self._group_stats_tool)

        # Trading domain tools (all depend on DataHub, lazy-imported to avoid circular imports)
        try:
            from getall.trading.tools.backtest import BacktestTool
            from getall.trading.tools.market_data import MarketDataTool
            from getall.trading.tools.news_sentiment import NewsSentimentTool
            from getall.trading.tools.technical_analysis import TechnicalAnalysisTool
            from getall.trading.tools.portfolio import PortfolioTool
            from getall.trading.tools.trade import TradeTool

            self._trading_hub = DataHub(self.trading_config, self.workspace)
            self.tools.register(BacktestTool(hub=self._trading_hub, workspace=self.workspace))
            self.tools.register(MarketDataTool(hub=self._trading_hub))
            self.tools.register(NewsSentimentTool(hub=self._trading_hub, workspace_path=self.workspace))
            self.tools.register(TechnicalAnalysisTool(hub=self._trading_hub))
            self.tools.register(PortfolioTool(hub=self._trading_hub, workspace_path=self.workspace))
            self.tools.register(TradeTool(hub=self._trading_hub))
        except Exception as e:
            logger.warning(f"Trading tools disabled: {e}")

        # Multi-source market data tools (no DataHub dependency)
        import os
        self.tools.register(CoinGeckoTool(api_key=os.environ.get("COINGECKO_API_KEY", "")))
        self.tools.register(YFinanceTool())

        # DeFi & sentiment tools (free, no API key)
        self.tools.register(DefiLlamaTool())
        self.tools.register(FearGreedTool())

        # Financial data tools (API key required)
        self.tools.register(FinnhubTool(api_key=os.environ.get("FINNHUB_API_KEY", "")))
        self.tools.register(FreeCryptoTool(api_key=os.environ.get("FREECRYPTO_API_KEY", "")))
    
    @staticmethod
    def _resolve_system_origin(chat_id: str) -> tuple[str, str]:
        """Parse system-message chat_id into origin channel/chat pair."""
        if ":" in chat_id:
            parts = chat_id.split(":", 1)
            return parts[0], parts[1]
        return "cli", chat_id

    def _lock_key_for_message(
        self,
        msg: InboundMessage,
        session_key: str | None = None,
    ) -> str:
        """Compute processing lock key.

        Strategy:
        - Private/system flows: serialize per conversation.
        - Group flows: serialize per sender within a chat, so different
          members can run concurrently while one member stays ordered.
        """
        if session_key:
            return session_key
        if msg.channel == "system":
            origin_channel, origin_chat_id = self._resolve_system_origin(msg.chat_id)
            return f"{origin_channel}:{origin_chat_id}"
        if msg.chat_type == "group":
            sender = (msg.sender_id or "-").strip() or "-"
            return f"{msg.channel}:{msg.chat_id}:sender:{sender}"
        return msg.session_key

    async def _acquire_session_lock(self, lock_key: str) -> asyncio.Lock:
        """Acquire serialized processing lock for a conversation key."""
        async with self._session_lock_guard:
            lock = self._session_locks.get(lock_key)
            if lock is None:
                lock = asyncio.Lock()
                self._session_locks[lock_key] = lock
            self._session_lock_refcounts[lock_key] = (
                self._session_lock_refcounts.get(lock_key, 0) + 1
            )

        try:
            await lock.acquire()
        except BaseException:
            async with self._session_lock_guard:
                refs = self._session_lock_refcounts.get(lock_key, 0) - 1
                if refs <= 0:
                    self._session_lock_refcounts.pop(lock_key, None)
                    current = self._session_locks.get(lock_key)
                    if current is lock and not lock.locked():
                        self._session_locks.pop(lock_key, None)
                else:
                    self._session_lock_refcounts[lock_key] = refs
            raise
        return lock

    async def _release_session_lock(self, lock_key: str, lock: asyncio.Lock) -> None:
        """Release serialized processing lock and cleanup idle lock entries."""
        lock.release()
        async with self._session_lock_guard:
            refs = self._session_lock_refcounts.get(lock_key, 0) - 1
            if refs <= 0:
                self._session_lock_refcounts.pop(lock_key, None)
                current = self._session_locks.get(lock_key)
                if current is lock and not current.locked():
                    self._session_locks.pop(lock_key, None)
            else:
                self._session_lock_refcounts[lock_key] = refs

    async def _process_message_with_lock(
        self,
        msg: InboundMessage,
        *,
        session_key: str | None = None,
    ) -> OutboundMessage | None:
        """Serialize same-session messages while allowing cross-session concurrency."""
        lock_key = self._lock_key_for_message(msg, session_key=session_key)
        lock = await self._acquire_session_lock(lock_key)
        try:
            return await self._process_message(msg, session_key=session_key)
        finally:
            await self._release_session_lock(lock_key, lock)

    async def _process_inbound_message(self, msg: InboundMessage) -> None:
        """Process one inbound message and publish outbound response."""
        try:
            response = await self._process_message_with_lock(msg)
            if response:
                await self.bus.publish_outbound(response)
        except Exception as e:
            logger.error(f"Error processing message: {e}", exc_info=True)
            # Send user-friendly error (raw details stay in logs)
            await self.bus.publish_outbound(OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content="出了点小问题，请稍后再试。如果持续出错请联系管理员。",
                metadata=msg.metadata or {},
            ))

    async def _worker_loop(self, worker_id: int) -> None:
        """Worker loop consuming inbound queue concurrently."""
        while self._running:
            try:
                msg = await asyncio.wait_for(self.bus.consume_inbound(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker {worker_id} failed to consume inbound: {e}", exc_info=True)
                continue
            await self._process_inbound_message(msg)

    async def run(self) -> None:
        """Run the agent loop with concurrent workers."""
        self._running = True
        logger.info(f"Agent loop started with {self.max_concurrent_workers} workers")
        self._worker_tasks = [
            asyncio.create_task(self._worker_loop(i + 1))
            for i in range(self.max_concurrent_workers)
        ]
        try:
            await asyncio.gather(*self._worker_tasks)
        finally:
            for task in self._worker_tasks:
                task.cancel()
            await asyncio.gather(*self._worker_tasks, return_exceptions=True)
            self._worker_tasks.clear()
    
    def stop(self) -> None:
        """Stop the agent loop and cancel worker tasks."""
        self._running = False
        for task in self._worker_tasks:
            task.cancel()
        logger.info("Agent loop stopping")
    
    @staticmethod
    def _resolve_memory_scope(msg: InboundMessage) -> str:
        """Resolve memory namespace. Keep group chats shared, isolate others per principal."""
        tenant = msg.tenant_id or "default"
        if msg.chat_type == "group":
            return f"group/{tenant}/{msg.channel}/{msg.chat_id or '-'}"
        if msg.principal_id:
            return f"principal/{tenant}/{msg.principal_id}"
        return f"legacy/{tenant}/{msg.channel}/{msg.chat_id or '-'}"

    @classmethod
    def _is_no_reply(cls, content: str | None) -> bool:
        """Return True when model indicates group silence."""
        if content is None:
            return True
        stripped = content.strip()
        if not stripped:
            return True
        return bool(cls._NO_REPLY_RE.match(stripped))

    @staticmethod
    def _sender_name_for_group(msg: InboundMessage) -> str:
        """Resolve a stable sender label for group messages."""
        if msg.chat_type != "group":
            return ""
        metadata = msg.metadata or {}
        sender_name = str(
            msg.sender_name or metadata.get("sender_name") or ""
        ).strip()
        if sender_name:
            return sender_name
        return str(msg.sender_id or "").strip()

    @classmethod
    def _format_current_user_content(cls, msg: InboundMessage) -> str:
        """Prefix current user turn with sender name in groups."""
        sender_name = cls._sender_name_for_group(msg)
        if not sender_name:
            return msg.content
        return f"[{sender_name}] {msg.content}"

    @staticmethod
    def _inject_group_reply_policy(
        msg: InboundMessage,
        messages: list[dict[str, Any]],
    ) -> None:
        """Inject per-message group reply policy into system prompt."""
        if msg.chat_type != "group" or not messages:
            return
        if messages[0].get("role") != "system":
            return

        metadata = msg.metadata or {}
        bot_was_mentioned = bool(metadata.get("bot_was_mentioned"))
        ambient_probe = bool(metadata.get("ambient_probe"))
        if bot_was_mentioned:
            messages[0]["content"] += f"""

## Group Reply Decision
Current trigger: direct_mention
- This is a strong trigger, so you MUST reply.
- Never output [NO_REPLY] for direct mentions.
- Keep it concise and useful; if intent is unclear, ask one short clarifying question."""
            return
        if ambient_probe:
            messages[0]["content"] += """

## Group Reply Decision
Current trigger: ambient_probe
- This is a rare non-@ probe. Do NOT always reply.
- Only reply when you can add clear value to the current group context.
- If you reply, keep it within 1-2 short sentences (casual, low-noise).
- If not worth replying, output EXACTLY: [NO_REPLY]."""
            return
        messages[0]["content"] += """

## Group Reply Decision
Current trigger: ambient_listener
- Use your judgment, not rigid keyword matching.
- Reply naturally when the user clearly expects you, when this is a follow-up to your recent reply, or when you can add clear value.
- If replying would create noise (side chat, casual banter, emoji-only, no real ask), output EXACTLY: [NO_REPLY]
- For ambient listening (not mentioned), stay restrained. If uncertain, choose [NO_REPLY]."""
    
    # ── tool timing injection ──

    @staticmethod
    def _inject_tool_timings(
        session: "Session",
        messages: list[dict[str, Any]],
    ) -> None:
        """Inject recent tool execution timings into system prompt.

        This lets the agent estimate how long similar tasks will take and
        decide whether to set task-scoped heartbeat reminders.
        """
        timings: list[dict[str, Any]] = session.metadata.get("tool_timings", [])
        if not timings or not messages or messages[0].get("role") != "system":
            return
        # Only show tools that took >= 2s (meaningful for duration estimation)
        significant = [t for t in timings if t.get("elapsed_seconds", 0) >= 2.0]
        if not significant:
            return
        lines = [
            f"- {t['tool']}({t.get('args_summary', '')[:80]}): {t['elapsed_seconds']}s"
            for t in significant[-10:]
        ]
        messages[0]["content"] += (
            "\n\n## Recent Tool Execution Times (reference for estimating task duration)\n"
            + "\n".join(lines)
        )

    # ── voice mode helpers ──

    async def _transcribe_audio(self, msg: InboundMessage) -> str:
        """Transcribe audio from an inbound message.  Returns text or ''."""
        audio_path = (msg.metadata or {}).get("audio_path", "")
        if not audio_path:
            # Also check media list for audio files.
            for m in msg.media:
                if m.endswith((".ogg", ".opus", ".mp3", ".wav", ".m4a")):
                    audio_path = m
                    break
        if not audio_path:
            return ""
        if not self.voice.available:
            logger.warning("Voice provider not available (no OpenAI API key)")
            return ""

        text = await self.voice.stt(Path(audio_path))
        if text:
            logger.info(f"STT transcription: {text[:80]}")
        return text

    async def _maybe_tts(self, text: str, directive: Any = None) -> str:
        """Convert text to speech if feasible.

        Args:
            text: Already-cleaned text (no [[voice:...]] tags).
            directive: Pre-parsed VoiceDirective or None.

        Returns:
            Audio file path, or '' if TTS was skipped.
        """
        if not self.voice.available:
            return ""
        # Skip TTS for code-heavy or very long responses.
        if is_code_heavy(text):
            logger.debug("Skipping TTS: code-heavy response")
            return ""
        if len(text) > TTS_MAX_TEXT_LENGTH:
            logger.debug(f"Skipping TTS: text too long ({len(text)} chars)")
            return ""

        voice: str | None = None
        instructions: str | None = None
        if directive:
            voice = directive.voice
            instructions = directive.instructions or None
            logger.info(f"Voice directive: voice={voice}, instructions={instructions!r}")

        # Prepare text for speech synthesis.
        clean = strip_markdown_for_tts(text)
        if len(clean) < 2:
            return ""
        try:
            path = await self.voice.tts(
                clean,
                voice=voice,
                instructions=instructions,
            )
            return str(path)
        except Exception as exc:
            logger.error(f"TTS failed: {exc}")
            return ""

    async def _process_message(self, msg: InboundMessage, session_key: str | None = None) -> OutboundMessage | None:
        """
        Process a single inbound message.
        
        Args:
            msg: The inbound message to process.
            session_key: Override session key (used by process_direct).
        
        Returns:
            The response message, or None if no response needed.
        """
        # Handle system messages (subagent announces)
        # The chat_id contains the original "channel:chat_id" to route back to
        if msg.channel == "system":
            return await self._process_system_message(msg)
        
        # ── Voice: STT for incoming audio messages ──
        is_audio_msg = bool((msg.metadata or {}).get("is_audio"))
        if is_audio_msg and not msg.content:
            transcribed = await self._transcribe_audio(msg)
            if transcribed:
                msg.content = transcribed
            else:
                msg.content = "（语音消息，未能识别）"
                logger.warning("Failed to transcribe voice message")

        preview = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
        logger.info(f"Processing message from {msg.channel}:{msg.sender_id}: {preview}")
        
        # Get or create session
        session = self.sessions.get_or_create(session_key or msg.session_key)

        # ── Voice mode: default OFF ──
        # Voice replies only when the agent emits [[voice:on]] (user explicitly asked).

        # ── Cross-chat handoff injection ──
        # If the agent proactively DM'd this user from a group chat earlier,
        # inject the sent messages into this session so the agent has context.
        if msg.channel and msg.chat_id:
            handoff_msgs = consume_handoff(msg.channel, msg.chat_id)
            if handoff_msgs:
                for hm in handoff_msgs:
                    session.add_message(
                        hm.get("role", "assistant"),
                        hm.get("content", ""),
                    )
                self.sessions.save(session)
                logger.info(
                    f"Injected {len(handoff_msgs)} handoff message(s) into "
                    f"session {session.key}"
                )

        # Always reset persona tool context first so any failure path below
        # cannot accidentally reuse stale private/group bindings.
        self._persona_tool.clear_context()
        
        # Update tool contexts
        message_tool = self.tools.get("message")
        if isinstance(message_tool, MessageTool):
            message_tool.set_context(msg.channel, msg.chat_id)
        
        spawn_tool = self.tools.get("spawn")
        if isinstance(spawn_tool, SpawnTool):
            spawn_tool.set_context(msg.channel, msg.chat_id)
        
        reminders_tool = self.tools.get("reminders")
        if isinstance(reminders_tool, ReminderTool):
            reminders_tool.set_context(msg.channel, msg.chat_id)

        # Group stats tool context
        self._group_stats_tool.set_context(msg.chat_id or "", msg.chat_type or "")

        # ── Identity resolution: ensure every message has a principal_id ──
        persona: dict[str, Any] | None = None
        is_group = msg.chat_type == "group"
        try:
            from getall.storage.database import get_session_factory
            from getall.storage.repository import IdentityRepo
            from getall.identity.router_hook import IdentityRouterHook
            factory = get_session_factory()

            # Step 1: resolve or create principal from platform identity
            if not msg.principal_id:
                async with factory() as db_session:
                    hook = IdentityRouterHook(db_session)
                    # Group chats must have an identity independent from any
                    # individual user. We key group identity by chat_id and
                    # disable IFT extraction from group message text.
                    resolver_user_id = msg.sender_id
                    resolver_message_text = msg.content
                    if is_group:
                        resolver_user_id = (
                            f"group:{msg.tenant_id or 'default'}:{msg.chat_id or '-'}"
                        )
                        resolver_message_text = ""
                    resolution = await hook.resolve(
                        tenant_id=msg.tenant_id or "default",
                        channel=msg.channel,
                        chat_id=msg.chat_id,
                        user_id=resolver_user_id,
                        thread_id=msg.thread_id,
                        message_text=resolver_message_text,
                    )
                    await db_session.commit()
                    msg.principal_id = resolution.principal_id
                    msg.agent_identity_id = resolution.agent_identity_id
                    logger.info(f"Identity resolved: {msg.channel}:{msg.sender_id} -> {resolution.ift}")

            # Step 2: load persona.
            # Private chats: full persona for personalized experience.
            # Group chats: lightweight check (onboarded flag only) so the
            # bot can nudge unregistered users to private chat.
            if msg.principal_id:
                async with factory() as db_session:
                    repo = IdentityRepo(db_session)
                    principal = await repo.get_by_id(msg.principal_id)
                    if principal is not None:
                        if is_group:
                            # Group: only pass onboarded status + sender info
                            persona = {
                                "onboarded": principal.onboarded,
                                "sender_open_id": msg.sender_id,
                            }
                        else:
                            # Private: full persona
                            persona = {
                                "pet_name": principal.pet_name or "",
                                "persona_text": principal.persona_text or "",
                                "trading_style_text": principal.trading_style_text or "",
                                "ift": principal.ift or "",
                                "onboarded": principal.onboarded,
                            }
                            # Set persona tool context so LLM can update persona
                            self._persona_tool.set_context(principal.id, factory)

            # Set credential + bitget tool contexts (need principal_id + db factory)
            if msg.principal_id:
                self._credential_tool.set_context(msg.principal_id, factory, msg.chat_type)
                self._bitget_account_tool.set_context(msg.principal_id, factory)
                self._bitget_trade_tool.set_context(msg.principal_id, factory)
                self._bitget_uta_tool.set_context(msg.principal_id, factory)
        except Exception as e:
            logger.warning(f"Identity/persona resolution failed: {e}")

        # Set strict identity context for reminders (list/remove/add are owner-scoped).
        if isinstance(reminders_tool, ReminderTool):
            reminders_tool.set_context(
                msg.channel,
                msg.chat_id,
                tenant_id=msg.tenant_id or "default",
                principal_id=msg.principal_id,
                agent_identity_id=msg.agent_identity_id,
                sender_id=msg.sender_id,
                thread_id=msg.thread_id,
                chat_type=msg.chat_type,
                synthetic=bool((msg.metadata or {}).get("synthetic")),
                source=str((msg.metadata or {}).get("source", "")),
            )

        # Persist per-principal route so cron/WS can deliver to this user later.
        # Private and group routes are stored separately — monitoring always
        # uses the private route so personal data is never leaked to groups.
        if msg.principal_id:
            from getall.routing import save_last_route
            save_last_route(msg.channel, msg.chat_id, msg.principal_id, msg.chat_type)

        memory_scope = self._resolve_memory_scope(msg)
        session.metadata["memory_scope"] = memory_scope

        # ── Group persona: load from file and enable persona tool ──
        if is_group:
            group_persona_path = self.workspace / "memory" / memory_scope / "persona.json"
            group_persona = PetPersonaTool.load_group_persona(group_persona_path)
            if group_persona:
                persona = persona or {}
                persona["pet_name"] = group_persona.get("pet_name", "")
                persona["persona_text"] = group_persona.get("persona_text", "")
                persona["trading_style_text"] = group_persona.get("trading_style_text", "")
            self._persona_tool.set_group_context(group_persona_path)

        # Consolidate memory with correct scope once identity/session metadata are ready.
        if len(session.messages) > self.memory_window:
            await self._consolidate_memory(session)
        
        # Build initial messages (use get_history for LLM-formatted messages)
        # For audio messages, exclude audio file paths from media sent to LLM.
        llm_media = [
            m for m in (msg.media or [])
            if not m.endswith((".ogg", ".opus", ".mp3", ".wav", ".m4a"))
        ] or None
        messages = self.context.build_messages(
            history=session.get_history(),
            current_message=self._format_current_user_content(msg),
            media=llm_media,
            channel=msg.channel,
            chat_id=msg.chat_id,
            chat_type=msg.chat_type,
            persona=persona,
            memory_scope=memory_scope,
        )

        # Inject voice capability section into system prompt (always).
        voice_mode_active = False  # default OFF; agent enables via [[voice:on]] when user explicitly asks
        if self.voice.available and messages and messages[0].get("role") == "system":
            mode_label = "ON" if voice_mode_active else "OFF"
            voice_section = _VOICE_SECTION_TEMPLATE.format(
                mode=mode_label,
                voice_active_section=_VOICE_ACTIVE_TIPS if voice_mode_active else "",
                voice_options=build_voice_options_hint(),
            )
            messages[0]["content"] += voice_section

        # Inject group member list so the agent knows who's in the chat
        group_members = (msg.metadata or {}).get("group_members", [])
        if group_members and messages and messages[0].get("role") == "system":
            names = ", ".join(group_members)
            messages[0]["content"] += (
                f"\n\n## Group Members (from API)\n"
                f"This group has {len(group_members)} members: {names}\n"
                f"This is the complete member list from the Lark API, not just people who have spoken."
            )
            logger.debug(f"Injected {len(group_members)} group members into context: {names}")
        self._inject_group_reply_policy(msg, messages)

        # ── Inject tool execution timing history for duration estimation ──
        self._inject_tool_timings(session, messages)
        
        # ── Task-scoped heartbeat infrastructure ──
        # Background tasks that fire progress messages after a delay.
        # All auto-cancelled when processing completes.
        _heartbeat_tasks: dict[str, asyncio.Task[None]] = {}
        _heartbeat_counter = 0

        async def _fire_heartbeat(delay: float, content: str) -> None:
            """Background coroutine: sleep then send a progress message."""
            try:
                await asyncio.sleep(delay)
                await self.bus.publish_outbound(OutboundMessage(
                    channel=msg.channel,
                    chat_id=msg.chat_id,
                    content=content,
                    metadata=msg.metadata or {},
                ))
                logger.info(f"Heartbeat fired: {content[:60]}")
            except asyncio.CancelledError:
                pass

        def _register_heartbeat(delay_seconds: int, message: str) -> str:
            nonlocal _heartbeat_counter
            _heartbeat_counter += 1
            hb_id = f"hb_{_heartbeat_counter}"
            task = asyncio.create_task(_fire_heartbeat(delay_seconds, message))
            _heartbeat_tasks[hb_id] = task
            return hb_id

        def _cancel_heartbeat(hb_id: str) -> bool:
            task = _heartbeat_tasks.get(hb_id)
            if task and not task.done():
                task.cancel()
                return True
            return False

        # Wire heartbeat callbacks into the reminder tool
        if isinstance(reminders_tool, ReminderTool):
            reminders_tool.set_heartbeat_callbacks(_register_heartbeat, _cancel_heartbeat)

        # Agent loop
        iteration = 0
        final_content = None
        tools_used: list[str] = []
        autofix_hints_sent = 0
        
        while iteration < self.max_iterations:
            iteration += 1
            
            # Call LLM
            response = await self.provider.chat(
                messages=messages,
                tools=self.tools.get_definitions(),
                model=self.model,
                max_tokens=self.max_tokens,
                reasoning_effort=self.reasoning_effort,
            )
            
            # LLM error → stop loop, use error message as final response
            if response.finish_reason == "error":
                final_content = response.content
                logger.warning(f"LLM returned error (iteration {iteration}): {final_content}")
                break

            # Handle tool calls
            if response.has_tool_calls:
                # ── Intermediate text: send LLM's accompanying text to user ──
                # When the LLM says something alongside tool calls (e.g.
                # "让我来查一下…") we deliver it immediately so the user
                # sees natural, human-like progress rather than silence.
                intermediate_text = (response.content or "").strip()
                if intermediate_text and not self._is_no_reply(intermediate_text):
                    await self.bus.publish_outbound(OutboundMessage(
                        channel=msg.channel,
                        chat_id=msg.chat_id,
                        content=intermediate_text,
                        metadata=msg.metadata or {},
                    ))
                    logger.info(f"Sent intermediate text: {intermediate_text[:80]}")

                # Add assistant message with tool calls
                tool_call_dicts = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments)  # Must be JSON string
                        }
                    }
                    for tc in response.tool_calls
                ]
                messages = self.context.add_assistant_message(
                    messages, response.content, tool_call_dicts,
                    reasoning_content=response.reasoning_content,
                )
                
                # Execute tools (with timing for duration estimation)
                for tool_call in response.tool_calls:
                    tools_used.append(tool_call.name)
                    args_str = json.dumps(tool_call.arguments, ensure_ascii=False)
                    logger.info(f"Tool call: {tool_call.name}({args_str[:200]})")
                    t0 = asyncio.get_event_loop().time()
                    result = await self.tools.execute(tool_call.name, tool_call.arguments)
                    elapsed = round(asyncio.get_event_loop().time() - t0, 1)
                    if elapsed >= 1.0:
                        logger.info(f"Tool {tool_call.name} took {elapsed}s")
                    # Record execution time for agent estimation
                    session.add_tool_execution(
                        tool_call.name,
                        tool_call.arguments,
                        result[:200] if result else "",
                    )
                    session.metadata.setdefault("tool_timings", []).append({
                        "tool": tool_call.name,
                        "args_summary": args_str[:120],
                        "elapsed_seconds": elapsed,
                    })
                    # Keep only recent timings
                    timings = session.metadata["tool_timings"]
                    if len(timings) > 50:
                        session.metadata["tool_timings"] = timings[-50:]

                    messages = self.context.add_tool_result(
                        messages, tool_call.id, tool_call.name, result
                    )
                    hint = self._build_autofix_hint(tool_call.name, result)
                    if hint and autofix_hints_sent < 2:
                        messages.append({"role": "system", "content": hint})
                        autofix_hints_sent += 1
            else:
                # No tool calls, we're done
                final_content = response.content
                break
        
        if final_content is None:
            # Exhausted max_iterations with every round being tool calls.
            # Make one final LLM call WITHOUT tools to force a text summary.
            logger.warning(
                f"Agent loop exhausted {self.max_iterations} iterations without "
                "a final text response; forcing summary call"
            )
            messages.append({
                "role": "system",
                "content": (
                    "You have used all available tool-call rounds. "
                    "Now give the user a concise final answer based on "
                    "everything you've gathered so far. Do NOT call any tools."
                ),
            })
            try:
                summary_resp = await self.provider.chat(
                    messages=messages,
                    tools=None,
                    model=self.model,
                    max_tokens=self.max_tokens,
                    reasoning_effort=self.reasoning_effort,
                )
                final_content = (summary_resp.content or "").strip()
            except Exception as exc:
                logger.error(f"Summary call failed after loop exhaustion: {exc}")
            if not final_content:
                final_content = "处理完成，但未能生成最终回复，请再试一次。"

        # ── Cancel all pending heartbeats (task completed) ──
        for hb_id, task in _heartbeat_tasks.items():
            if not task.done():
                task.cancel()
                logger.debug(f"Heartbeat {hb_id} auto-cancelled (task complete)")
        _heartbeat_tasks.clear()
        if isinstance(reminders_tool, ReminderTool):
            reminders_tool.clear_heartbeat_callbacks()

        metadata = msg.metadata or {}
        strong_trigger = bool(metadata.get("bot_was_mentioned"))
        if is_group and strong_trigger and self._is_no_reply(final_content):
            logger.warning(
                "Model returned NO_REPLY on strong group trigger; applying fallback reply"
            )
            final_content = "我在，收到。你继续说，我马上处理。"

        suppress_group_reply = (
            is_group and (not strong_trigger) and self._is_no_reply(final_content)
        )
        if suppress_group_reply:
            logger.info(
                "Suppressing group outbound by model decision: "
                f"{msg.channel}:{msg.chat_id}"
            )

        # Parse [[voice:...]] directive — strip from text; apply to this reply only.
        _voice_directive = None
        if final_content:
            final_content, _voice_directive = parse_voice_directive(final_content)
            if _voice_directive and _voice_directive.mode_switch:
                new_mode = _voice_directive.mode_switch == "on"
                if new_mode != voice_mode_active:
                    voice_mode_active = new_mode
                    logger.info(f"Voice mode {'ON' if new_mode else 'OFF'} (agent directive, this reply only)")

        # Log response preview
        preview = final_content[:120] + "..." if len(final_content) > 120 else final_content
        logger.info(f"Response to {msg.channel}:{msg.sender_id}: {preview}")
        
        # Save to session (include tool names so consolidation sees what happened)
        is_synthetic_event = bool((msg.metadata or {}).get("synthetic"))
        if not is_synthetic_event:
            user_message: dict[str, Any] = {"chat_type": msg.chat_type}
            if is_audio_msg:
                user_message["voice"] = True
            if is_group:
                user_message["sender_id"] = msg.sender_id
                sender_name = self._sender_name_for_group(msg)
                if sender_name:
                    user_message["sender_name"] = sender_name
            session.add_message("user", msg.content, **user_message)
        if not suppress_group_reply:
            session.add_message(
                "assistant",
                final_content,
                tools_used=tools_used if tools_used else None,
            )
        self.sessions.save(session)

        if suppress_group_reply:
            return None

        # If the agent already sent messages via the message tool TO THE SAME CHAT,
        # suppress the final outbound to avoid duplicate output.
        # Messages sent to OTHER chats (e.g. a DM from a group context) don't suppress.
        if "message" in tools_used:
            msg_tool = self.tools.get("message")
            sent_to_same = any(
                cid == msg.chat_id
                for cid in getattr(msg_tool, "sent_to_chat_ids", [])
            )
            if sent_to_same:
                logger.debug("Suppressing final outbound — message tool already sent to this chat")
                return None
        
        # Extract generated image paths from tool results in the conversation
        media = self._extract_generated_images(messages)

        # ── Voice: TTS for outbound when voice mode is active ──
        audio_path = ""
        if voice_mode_active and final_content:
            audio_path = await self._maybe_tts(final_content, _voice_directive)
            if audio_path:
                logger.info(f"TTS generated: {audio_path}")

        return OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content=final_content,
            media=media,
            metadata=msg.metadata or {},  # Pass through for channel-specific needs (e.g. Slack thread_ts)
            audio_path=audio_path,
        )
    
    _GENERATED_IMAGE_RE = re.compile(r"\[GENERATED_IMAGE:(.*?)\]")
    _AUTOFIX_ERROR_PATTERNS = (
        "command not found",
        "not configured",
        "no module named",
        "module not found",
        "tool '",
        "missing required",
        "permission denied",
        "timed out",
        "blocked by safety guard",
        "outside allowed directory",
        "error: script not found",
    )

    @classmethod
    def _extract_generated_images(cls, messages: list[dict[str, Any]]) -> list[str]:
        """Extract [GENERATED_IMAGE:/path] markers from tool results."""
        paths: list[str] = []
        for m in messages:
            if m.get("role") != "tool":
                continue
            content = m.get("content", "")
            for match in cls._GENERATED_IMAGE_RE.finditer(content):
                fpath = match.group(1).strip()
                if Path(fpath).is_file():
                    paths.append(fpath)
        return paths

    @classmethod
    def _build_autofix_hint(cls, tool_name: str, result: str) -> str | None:
        text = (result or "").lower()
        if not text:
            return None
        if not (text.startswith("error") or "stderr:" in text):
            return None
        if not any(p in text for p in cls._AUTOFIX_ERROR_PATTERNS):
            return None
        return (
            "Tool execution failed due to missing capability/setup. "
            "Autonomously self-heal: (1) use web_search (prefer DuckDuckGo and reputable skill/tool registries "
            "like clawhub.ai, smithery.ai, MCP registry, GitHub), "
            "(2) install or create a reusable shared capability with workbench, "
            "(3) retry the original task immediately, "
            "(4) report to user in one concise natural-language line what was installed. "
            "For privileged actions, ask for one explicit confirmation first."
        )

    async def _process_system_message(self, msg: InboundMessage) -> OutboundMessage | None:
        """
        Process a system message (e.g., subagent announce).
        
        The chat_id field contains "original_channel:original_chat_id" to route
        the response back to the correct destination.
        """
        logger.info(f"Processing system message from {msg.sender_id}")
        
        # Parse origin from chat_id (format: "channel:chat_id")
        origin_channel, origin_chat_id = self._resolve_system_origin(msg.chat_id)
        
        # Use the origin session for context
        session_key = f"{origin_channel}:{origin_chat_id}"
        session = self.sessions.get_or_create(session_key)
        
        # Update tool contexts
        message_tool = self.tools.get("message")
        if isinstance(message_tool, MessageTool):
            message_tool.set_context(origin_channel, origin_chat_id)
        
        spawn_tool = self.tools.get("spawn")
        if isinstance(spawn_tool, SpawnTool):
            spawn_tool.set_context(origin_channel, origin_chat_id)
        
        reminders_tool = self.tools.get("reminders")
        if isinstance(reminders_tool, ReminderTool):
            reminders_tool.set_context(origin_channel, origin_chat_id)
        
        # Build messages with the announce content
        memory_scope = str(
            session.metadata.get("memory_scope")
            or f"legacy/default/{origin_channel}/{origin_chat_id}"
        )
        messages = self.context.build_messages(
            history=session.get_history(),
            current_message=msg.content,
            channel=origin_channel,
            chat_id=origin_chat_id,
            memory_scope=memory_scope,
        )
        
        # Agent loop (limited for announce handling)
        iteration = 0
        final_content = None
        
        while iteration < self.max_iterations:
            iteration += 1
            
            response = await self.provider.chat(
                messages=messages,
                tools=self.tools.get_definitions(),
                model=self.model,
                max_tokens=self.max_tokens,
                reasoning_effort=self.reasoning_effort,
            )

            if response.finish_reason == "error":
                final_content = response.content
                logger.warning(f"LLM error in system handler (iter {iteration}): {final_content}")
                break
            
            if response.has_tool_calls:
                tool_call_dicts = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments)
                        }
                    }
                    for tc in response.tool_calls
                ]
                messages = self.context.add_assistant_message(
                    messages, response.content, tool_call_dicts,
                    reasoning_content=response.reasoning_content,
                )
                
                for tool_call in response.tool_calls:
                    args_str = json.dumps(tool_call.arguments, ensure_ascii=False)
                    logger.info(f"Tool call: {tool_call.name}({args_str[:200]})")
                    result = await self.tools.execute(tool_call.name, tool_call.arguments)
                    messages = self.context.add_tool_result(
                        messages, tool_call.id, tool_call.name, result
                    )
            else:
                final_content = response.content
                break
        
        if final_content is None:
            final_content = "Background task completed."
        
        # Save to session (mark as system message in history)
        session.add_message("user", f"[System: {msg.sender_id}] {msg.content}")
        session.add_message("assistant", final_content)
        self.sessions.save(session)
        
        return OutboundMessage(
            channel=origin_channel,
            chat_id=origin_chat_id,
            content=final_content
        )
    
    @staticmethod
    def _extract_consolidation_json(raw: str) -> dict[str, str]:
        """Best-effort extraction of {history_entry, memory_update} from LLM text.

        Strategies (in order):
        1. Direct json.loads after stripping markdown fences.
        2. Find the first '{' … last '}' substring and parse.
        3. Regex extraction of each field individually.
        """
        import json as _json

        def _try_parse(text: str) -> dict[str, str] | None:
            try:
                obj = _json.loads(text)
                if isinstance(obj, dict):
                    return obj
            except (ValueError, TypeError):
                return None

        # Strategy 1: strip markdown fences
        text = raw.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
        result = _try_parse(text)
        if result is not None:
            return result

        # Strategy 2: find outermost braces
        first = text.find("{")
        last = text.rfind("}")
        if first != -1 and last > first:
            result = _try_parse(text[first : last + 1])
            if result is not None:
                return result

        # Strategy 3: regex extraction per field
        extracted: dict[str, str] = {}
        for key in ("history_entry", "memory_update"):
            m = re.search(
                rf'"{key}"\s*:\s*"((?:[^"\\]|\\.)*)"',
                text,
                re.DOTALL,
            )
            if m:
                extracted[key] = m.group(1).replace('\\"', '"').replace("\\n", "\n")
        if extracted:
            logger.debug(f"Memory consolidation: used regex fallback, extracted keys: {list(extracted)}")
            return extracted

        raise ValueError(f"Could not extract JSON from LLM response ({len(raw)} chars)")

    async def _consolidate_memory(self, session) -> None:
        """Consolidate old messages into MEMORY.md + HISTORY.md, then trim session."""
        memory_scope = str(session.metadata.get("memory_scope") or "global")
        memory = MemoryStore(self.workspace, memory_scope)
        keep_count = min(10, max(2, self.memory_window // 2))
        old_messages = session.messages[:-keep_count]  # Everything except recent ones
        if not old_messages:
            return
        logger.info(
            "Memory consolidation started: "
            f"{len(session.messages)} messages, archiving {len(old_messages)}, "
            f"keeping {keep_count}, scope={memory_scope}"
        )

        # Format messages for LLM (include tool names when available)
        lines = []
        for m in old_messages:
            if not m.get("content"):
                continue
            speaker = ""
            if m.get("role") == "user" and str(m.get("chat_type", "")) == "group":
                sender_name = str(
                    m.get("sender_name") or m.get("sender_id") or ""
                ).strip()
                if sender_name:
                    speaker = f" [{sender_name}]"
            tools = f" [tools: {', '.join(m['tools_used'])}]" if m.get("tools_used") else ""
            lines.append(
                f"[{m.get('timestamp', '?')[:16]}] {m['role'].upper()}"
                f"{speaker}{tools}: {m['content']}"
            )
        conversation = "\n".join(lines)
        current_memory = memory.read_long_term()

        prompt = f"""You are a memory consolidation agent. Process this conversation and return a JSON object with exactly two keys:

1. "history_entry": A paragraph (2-5 sentences) summarizing the key events/decisions/topics. Start with a timestamp like [YYYY-MM-DD HH:MM]. Include enough detail to be useful when found by grep search later.

2. "memory_update": The updated long-term memory content. Add any new facts: user location, preferences, personal info, habits, project context, technical decisions, tools/services used. If nothing new, return the existing content unchanged.

## Current Long-term Memory
{current_memory or "(empty)"}

## Conversation to Process
{conversation}

Respond with ONLY valid JSON, no markdown fences."""

        try:
            response = await self.provider.chat(
                messages=[
                    {"role": "system", "content": "You are a memory consolidation agent. Respond only with valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                model=self.model,
            )
            import json as _json
            text = (response.content or "").strip()
            result = self._extract_consolidation_json(text)

            if entry := result.get("history_entry"):
                memory.append_history(entry)
            if update := result.get("memory_update"):
                if update != current_memory:
                    memory.write_long_term(update)

            logger.info(f"Memory consolidation done, session trimmed to {len(session.messages)} → {keep_count}")
        except Exception as e:
            logger.error(f"Memory consolidation failed: {e}")
        finally:
            # Always trim session to prevent unbounded growth,
            # even when consolidation fails.
            session.messages = session.messages[-keep_count:]
            self.sessions.save(session)

    async def process_direct(
        self,
        content: str,
        session_key: str = "cli:direct",
        channel: str = "cli",
        chat_id: str = "direct",
        *,
        sender_id: str = "user",
        tenant_id: str = "default",
        principal_id: str = "",
        agent_identity_id: str = "",
        thread_id: str = "",
        chat_type: str = "private",
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Process a message directly (for CLI or reminders usage).
        
        Args:
            content: The message content.
            session_key: Session identifier (overrides channel:chat_id for session lookup).
            channel: Source channel (for tool context routing).
            chat_id: Source chat ID (for tool context routing).
            sender_id: Sender identifier used for identity routing.
            tenant_id: Multi-tenant namespace.
            principal_id: Optional pre-resolved principal_id.
            agent_identity_id: Optional pre-resolved agent identity.
            thread_id: Optional thread identifier.
            chat_type: "private" or "group".
        
        Returns:
            The agent's response.
        """
        msg = InboundMessage(
            channel=channel,
            sender_id=sender_id,
            chat_id=chat_id,
            content=content,
            tenant_id=tenant_id,
            principal_id=principal_id,
            agent_identity_id=agent_identity_id,
            thread_id=thread_id,
            chat_type=chat_type,
            metadata=metadata or {},
        )
        
        response = await self._process_message_with_lock(msg, session_key=session_key)
        return response.content if response else ""
