"""Agent loop: the core processing engine."""

import asyncio
import json
import mimetypes
import re
from datetime import datetime, timedelta
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
from getall.agent.tools.admin import AdminTool
from getall.agent.tools.feedback import FeedbackTool
from getall.agent.tools.browser_use import BrowserUseTool
from getall.agent.tools.rss_news import RssNewsTool
from getall.agent.tools.yfinance_tool import YFinanceTool
from getall.agent.tools.akshare_tool import AKShareTool
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

    # ── Context-window budget constants ──
    # Known context limits per model prefix.  Add new entries as needed.
    _MODEL_CONTEXT_LIMITS: dict[str, int] = {
        "claude": 200_000,
        "maas_cl": 200_000,       # CloudsWay Claude proxies
        "gpt-4o": 128_000,
        "gpt-4": 128_000,
        "gpt-codex": 128_000,
        "deepseek": 128_000,
        "qwen": 131_072,
        "gemini": 1_000_000,
        "minimax": 1_000_000,     # MiniMax M2.5 series
    }
    _DEFAULT_CONTEXT_LIMIT = 128_000
    _CONTEXT_SAFETY_MARGIN = 8_000  # reserve for tool defs + overhead

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Fast token estimate for mixed CJK/Latin text.

        Heuristic (no external dep):
        - CJK ideographs ≈ 1 token each
        - Everything else ≈ 1 token per 3.5 chars (English avg)
        """
        cjk = 0
        other = 0
        for ch in text:
            cp = ord(ch)
            # CJK Unified Ideographs + common CJK ranges
            if (
                0x4E00 <= cp <= 0x9FFF
                or 0x3400 <= cp <= 0x4DBF
                or 0xF900 <= cp <= 0xFAFF
                or 0x20000 <= cp <= 0x2A6DF
            ):
                cjk += 1
            else:
                other += 1
        return cjk + int(other / 3.5) + 1

    @classmethod
    def _estimate_messages_tokens(cls, messages: list[dict[str, Any]]) -> int:
        """Estimate total tokens across all messages."""
        total = 0
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                total += cls._estimate_tokens(content)
            elif isinstance(content, list):
                # Multimodal: text parts only
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        total += cls._estimate_tokens(part.get("text", ""))
            # Tool calls in assistant messages
            for tc in msg.get("tool_calls", []):
                fn = tc.get("function", {})
                total += cls._estimate_tokens(fn.get("arguments", ""))
                total += cls._estimate_tokens(fn.get("name", ""))
            # Per-message overhead (role, formatting)
            total += 4
        return total

    def _get_context_limit(self, model: str) -> int:
        """Resolve context window size for the active model."""
        model_lower = model.lower()
        for prefix, limit in self._MODEL_CONTEXT_LIMITS.items():
            if prefix in model_lower:
                return limit
        return self._DEFAULT_CONTEXT_LIMIT

    def _fit_messages_to_budget(
        self,
        messages: list[dict[str, Any]],
        model: str,
    ) -> list[dict[str, Any]]:
        """Trim conversation history if total tokens would exceed context window.

        Preserves: system prompt (idx 0), current user message (last),
        and trims oldest history messages from the middle.

        Returns a new list (original is not mutated).
        """
        context_limit = self._get_context_limit(model)
        # Budget = context_limit - max_output_tokens - safety_margin
        budget = context_limit - self.max_tokens - self._CONTEXT_SAFETY_MARGIN
        if budget < 10_000:
            budget = 10_000  # absolute minimum for a useful conversation

        est = self._estimate_messages_tokens(messages)
        if est <= budget:
            return messages

        # Need to trim.  Keep system (first) + user message (last).
        # Remove oldest history messages until we fit.
        logger.warning(
            f"Context budget exceeded: ~{est} tokens > {budget} budget "
            f"(model={model}, context_limit={context_limit}, max_tokens={self.max_tokens}). "
            f"Trimming oldest history messages."
        )

        if len(messages) <= 2:
            return messages

        system_msg = messages[0]
        current_user_msg = messages[-1]
        history = list(messages[1:-1])

        # Tokens consumed by fixed parts
        fixed_tokens = (
            self._estimate_messages_tokens([system_msg])
            + self._estimate_messages_tokens([current_user_msg])
        )
        remaining = budget - fixed_tokens
        if remaining < 1000:
            # System prompt alone is too large; nothing we can do but send
            # system + user and hope for the best.
            logger.error(
                f"System prompt alone uses ~{fixed_tokens} tokens, "
                f"budget is {budget}. Sending with minimal history."
            )
            return [system_msg, current_user_msg]

        # Keep history from the END (newest first), drop oldest
        kept: list[dict[str, Any]] = []
        used = 0
        for msg in reversed(history):
            msg_tokens = self._estimate_messages_tokens([msg])
            if used + msg_tokens > remaining:
                break
            kept.append(msg)
            used += msg_tokens
        kept.reverse()

        trimmed = len(history) - len(kept)
        logger.info(
            f"Trimmed {trimmed} oldest history messages. "
            f"Keeping {len(kept)} ({used} tokens) + system + user."
        )
        return [system_msg, *kept, current_user_msg]
    
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

        # Event system (time-limited personas / hidden events)
        from getall.events.manager import EventManager
        self.event_manager = EventManager(workspace)
        
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
        # Buffer-based concurrency: buffer key presence = session active.
        # New messages for an active session are appended to the buffer
        # and drained by the agent loop at natural checkpoints.
        self._pending_buffers: dict[str, list[InboundMessage]] = {}
        self._pending_guard = asyncio.Lock()

        # ── Message coalescing ──
        # Media-only messages in private chats are held briefly so a
        # follow-up text message can be merged before LLM processing.
        self._coalescing_buffer: dict[str, list[InboundMessage]] = {}
        self._coalescing_tasks: dict[str, asyncio.Task[None]] = {}
        self._coalescing_lock = asyncio.Lock()

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

        # Admin tool (context set per-message based on principal role)
        self._admin_tool = AdminTool()
        self.tools.register(self._admin_tool)

        # Feedback tool (available to all users; admin gets elevated permissions)
        self._feedback_tool = FeedbackTool()
        self.tools.register(self._feedback_tool)

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
        self.tools.register(AKShareTool())

        # DeFi & sentiment tools (free, no API key)
        self.tools.register(DefiLlamaTool())
        self.tools.register(FearGreedTool())

        # RSS news aggregator (free, no API key — 7 English crypto outlets)
        self.tools.register(RssNewsTool())

        # Financial data tools (API key required)
        self.tools.register(FinnhubTool(api_key=os.environ.get("FINNHUB_API_KEY", "")))
        self.tools.register(FreeCryptoTool(api_key=os.environ.get("FREECRYPTO_API_KEY", "")))

        # Browser automation tool (browser-use) — uses a dedicated OpenAI model
        # that natively supports structured output + vision for reliable scraping.
        _browser_api_key = os.environ.get("GETALL_OPENAI_API_KEY", "")
        if _browser_api_key:
            self.tools.register(BrowserUseTool(
                model="gpt-5.2",
                api_key=_browser_api_key,
                timeout=180,
            ))
        else:
            logger.warning("GETALL_OPENAI_API_KEY not set — browser_use tool disabled")

    async def _get_active_model(self, chat_type: str | None = None) -> str:
        """Resolve active model from system_config, falling back to self.model.

        Checks ``model:private`` or ``model:group`` in the DB, falls back to
        the default model configured at startup.
        """
        try:
            from getall.storage.database import get_session_factory
            from getall.storage.repository import SystemConfigRepo

            key = f"model:{chat_type}" if chat_type in ("private", "group") else "model:private"
            factory = get_session_factory()
            async with factory() as session:
                repo = SystemConfigRepo(session)
                value = await repo.get(key)
            if value:
                return value
        except Exception as exc:
            logger.debug(f"Failed to load active model from DB: {exc}")
        return self.model

    async def _record_llm_usage(
        self,
        response: "LLMResponse",
        *,
        tenant_id: str = "default",
        principal_id: str = "",
        session_key: str = "",
        call_type: str = "chat",
    ) -> None:
        """Persist a single LLM call's token usage and cost to the database.

        Runs fire-and-forget inside a try/except so it never blocks the
        main agent loop or surfaces errors to the user.
        """
        if not response.usage:
            return
        try:
            from getall.storage.database import get_session_factory
            from getall.storage.repository import LLMUsageRepo
            from getall.storage.models import LLMUsageRecord

            rec = LLMUsageRecord(
                tenant_id=tenant_id,
                principal_id=principal_id,
                session_key=session_key,
                model=response.model,
                provider=self._detect_provider_name(response.model),
                prompt_tokens=response.usage.get("prompt_tokens", 0),
                completion_tokens=response.usage.get("completion_tokens", 0),
                total_tokens=response.usage.get("total_tokens", 0),
                cost_usd=f"{response.cost_usd:.8f}",
                call_type=call_type,
            )
            factory = get_session_factory()
            async with factory() as session:
                repo = LLMUsageRepo(session)
                await repo.record(rec)
                await session.commit()
        except Exception as exc:
            logger.debug(f"Failed to record LLM usage: {exc}")

    @staticmethod
    def _detect_provider_name(model: str) -> str:
        """Best-effort provider name from the resolved model string."""
        if "/" in model:
            return model.split("/")[0]  # e.g. "openrouter/..." → "openrouter"
        return ""

    @staticmethod
    def _resolve_system_origin(chat_id: str) -> tuple[str, str]:
        """Parse system-message chat_id into origin channel/chat pair."""
        if ":" in chat_id:
            parts = chat_id.split(":", 1)
            return parts[0], parts[1]
        return "cli", chat_id

    def _buffer_key_for_message(
        self,
        msg: InboundMessage,
        session_key: str | None = None,
    ) -> str:
        """Compute buffer key for mid-loop message injection.

        Strategy:
        - Private chats use ``channel:chat_id`` so that system messages
          (reminders) share the same buffer key and can be injected into
          an active user session.
        - Group chats serialize per sender within a chat so different
          members can run concurrently while one member stays ordered.
        - System messages resolve to ``origin_channel:origin_chat_id``
          which matches the private-chat key of the target conversation.
        """
        if session_key:
            return session_key
        if msg.channel == "system":
            origin_channel, origin_chat_id = self._resolve_system_origin(msg.chat_id)
            return f"{origin_channel}:{origin_chat_id}"
        if msg.chat_type == "group":
            sender = (msg.sender_id or "-").strip() or "-"
            return f"{msg.channel}:{msg.chat_id}:sender:{sender}"
        return f"{msg.channel}:{msg.chat_id}"

    # ── Buffer helpers ────────────────────────────────────────────────

    async def _drain_pending(self, buffer_key: str) -> list[InboundMessage]:
        """Atomically drain all buffered messages for *buffer_key*.

        Returns the list of messages (may be empty).  The buffer entry
        itself is kept (only cleared) so new arrivals continue to be
        buffered while the session is active.
        """
        async with self._pending_guard:
            msgs = self._pending_buffers.get(buffer_key, [])
            if msgs:
                self._pending_buffers[buffer_key] = []
            return msgs

    _AUDIO_SUFFIXES = (".ogg", ".opus", ".mp3", ".wav", ".m4a")
    _MAX_SESSION_MEDIA_BUFFER = 10

    # Content patterns that indicate a turn involved media (protocol markers
    # inserted by the channel layer, NOT user-facing text classification).
    _MEDIA_CONTENT_MARKERS = ("[image]", "[sticker]", "[file:")

    async def _inject_buffered_messages(
        self,
        pending: list[InboundMessage],
        messages: list[dict[str, Any]],
        session: Any,
        llm_media: list[str] | None,
    ) -> list[str] | None:
        """Inject buffered messages into the live conversation context.

        For each pending ``InboundMessage``:
        - Audio messages are transcribed via STT.
        - User messages are formatted (group sender prefix) and media is
          base64-encoded for vision models.
        - System messages (reminders) are injected as ``role: system``.
        - Session history and ``_session_media`` are updated.

        Returns the (possibly updated) *llm_media* list so the caller
        can keep its reference current.
        """
        if not pending:
            return llm_media

        # ── Framing hint: tell the LLM these are live follow-ups ──
        user_msgs = [m for m in pending if m.channel != "system"]
        sys_msgs = [m for m in pending if m.channel == "system"]
        if user_msgs:
            messages.append({
                "role": "system",
                "content": (
                    f"[The user sent {len(user_msgs)} additional message(s) "
                    "while you were working. These are follow-ups or "
                    "supplements to the current task — treat them as extra "
                    "context / corrections / instructions. Act on them "
                    "proactively; do NOT ask for clarification unless the "
                    "intent is genuinely ambiguous.]"
                ),
            })

        for p_msg in pending:
            # ── Audio STT ──
            is_audio = bool((p_msg.metadata or {}).get("is_audio"))
            if is_audio and not p_msg.content:
                transcribed = await self._transcribe_audio(p_msg)
                p_msg.content = transcribed or "（语音消息，未能识别）"

            # ── Media handling ──
            p_media = [
                m for m in (p_msg.media or [])
                if not m.endswith(self._AUDIO_SUFFIXES)
            ] or None

            if p_media:
                # Persist to session media store (for future reference).
                existing: list[str] = session.metadata.get("_session_media", [])
                merged = list(dict.fromkeys(existing + p_media))
                merged = [p for p in merged if Path(p).is_file()]
                session.metadata["_session_media"] = merged[
                    -self._MAX_SESSION_MEDIA_BUFFER :
                ]
                # Track current-turn media only (no historical mixing).
                if llm_media is None:
                    llm_media = list(p_media)
                else:
                    llm_media = list(dict.fromkeys(llm_media + p_media))

            # ── Build content and inject ──
            if p_msg.channel == "system":
                messages.append({
                    "role": "system",
                    "content": (
                        "[A scheduled reminder fired while you were working. "
                        "Decide whether to act on it now or after finishing "
                        "the current task.]\n"
                        f"{p_msg.content}"
                    ),
                })
            else:
                text = self._format_current_user_content(p_msg)
                # Buffer-injected media is always "current" — no historical
                # mixing; the annotation label comes from the media param.
                user_content = self.context._build_user_content(
                    text, media=p_media,
                )
                messages.append({"role": "user", "content": user_content})

            # ── Persist to session history ──
            hist_kwargs: dict[str, Any] = {"chat_type": p_msg.chat_type}
            if is_audio:
                hist_kwargs["voice"] = True
            if p_msg.chat_type == "group":
                hist_kwargs["sender_id"] = p_msg.sender_id
                sender_name = self._sender_name_for_group(p_msg)
                if sender_name:
                    hist_kwargs["sender_name"] = sender_name
            session.add_message(
                "user" if p_msg.channel != "system" else "system",
                p_msg.content,
                **hist_kwargs,
            )

        self.sessions.save(session)
        n = len(pending)
        logger.info(
            f"Injected {n} buffered message(s) into active loop"
        )

        return llm_media

    # ── Message coalescing helpers ──────────────────────────────────

    # Content placeholders that indicate a media-only message.
    _MEDIA_PLACEHOLDER_RE = re.compile(
        r"^\s*(?:\[image\]|\[sticker\]|\[file:\s*[^\]]*\])\s*$",
        re.IGNORECASE,
    )
    _COALESCE_DELAY_SECONDS = 3.0

    def _should_coalesce(self, msg: InboundMessage) -> bool:
        """Return True when the message should wait for a follow-up.

        Conditions: private chat, has media, and no meaningful user text
        (content is just a placeholder like ``[image]``).
        """
        if msg.chat_type == "group":
            return False
        if not msg.media:
            return False
        return not self._has_meaningful_text(msg)

    @staticmethod
    def _has_meaningful_text(msg: InboundMessage) -> bool:
        """True when the message carries real user-typed text."""
        content = (msg.content or "").strip()
        if not content:
            return False
        return not AgentLoop._MEDIA_PLACEHOLDER_RE.match(content)

    @staticmethod
    def _merge_coalesced(msgs: list[InboundMessage]) -> InboundMessage:
        """Merge several buffered messages into a single InboundMessage.

        Media paths are deduplicated; text parts are concatenated.
        Metadata and routing fields come from the *first* message.
        """
        base = msgs[0]
        all_media: list[str] = []
        text_parts: list[str] = []
        for m in msgs:
            if m.media:
                all_media.extend(m.media)
            if AgentLoop._has_meaningful_text(m):
                text_parts.append(m.content.strip())

        merged_content = "\n".join(text_parts) if text_parts else base.content
        merged_media = list(dict.fromkeys(all_media))  # dedupe, preserve order

        return InboundMessage(
            channel=base.channel,
            sender_id=base.sender_id,
            chat_id=base.chat_id,
            content=merged_content,
            timestamp=base.timestamp,
            media=merged_media,
            metadata=base.metadata,
            tenant_id=base.tenant_id,
            principal_id=base.principal_id,
            agent_identity_id=base.agent_identity_id,
            thread_id=base.thread_id,
            sender_name=base.sender_name,
            chat_type=base.chat_type,
        )

    async def _fire_coalesced(self, buffer_key: str) -> None:
        """Timer callback: coalescing window expired, process collected messages."""
        await asyncio.sleep(self._COALESCE_DELAY_SECONDS)
        async with self._coalescing_lock:
            msgs = self._coalescing_buffer.pop(buffer_key, None)
            self._coalescing_tasks.pop(buffer_key, None)
        if not msgs:
            return
        merged = self._merge_coalesced(msgs)
        logger.info(
            f"Coalescing window expired for {buffer_key}: "
            f"{len(msgs)} msg(s) merged, media={len(merged.media)}"
        )
        await self._dispatch_inbound(merged)

    async def _dispatch_inbound(self, msg: InboundMessage) -> None:
        """Common path that buffers or processes a (possibly merged) message."""
        buffer_key = self._buffer_key_for_message(msg)

        async with self._pending_guard:
            if buffer_key in self._pending_buffers:
                self._pending_buffers[buffer_key].append(msg)
                preview = (msg.content or "")[:60]
                logger.info(
                    f"Buffered message for active session {buffer_key}: "
                    f"{preview!r}"
                )
                return
            self._pending_buffers[buffer_key] = []

        try:
            response = await self._process_message(
                msg, buffer_key=buffer_key,
            )
            if response:
                await self.bus.publish_outbound(response)
        except Exception as e:
            logger.error(f"Error processing message: {e}", exc_info=True)
            await self.bus.publish_outbound(OutboundMessage(
                channel=msg.channel,
                chat_id=msg.chat_id,
                content="出了点小问题，请稍后再试。如果持续出错请联系管理员。",
                metadata=msg.metadata or {},
            ))
        finally:
            async with self._pending_guard:
                remaining = self._pending_buffers.pop(buffer_key, [])
            for queued in remaining:
                await self.bus.publish_inbound(queued)

    async def _process_inbound_message(self, msg: InboundMessage) -> None:
        """Process one inbound message; coalesce or buffer if needed."""
        buffer_key = self._buffer_key_for_message(msg)

        # ── Coalescing: merge media-only → text follow-ups ──
        async with self._coalescing_lock:
            # 1. An existing coalescing window is open for this session.
            if buffer_key in self._coalescing_buffer:
                self._coalescing_buffer[buffer_key].append(msg)
                if self._has_meaningful_text(msg):
                    # Text arrived — cancel timer, process immediately.
                    task = self._coalescing_tasks.pop(buffer_key, None)
                    if task:
                        task.cancel()
                    msgs = self._coalescing_buffer.pop(buffer_key, [])
                    merged = self._merge_coalesced(msgs)
                    logger.info(
                        f"Coalescing early trigger for {buffer_key}: "
                        f"{len(msgs)} msg(s) merged, media={len(merged.media)}"
                    )
                else:
                    # Another media-only msg; stay in window.
                    return
                # Fall through to dispatch the merged message.
                msg = merged

            # 2. New message that qualifies for coalescing.
            elif self._should_coalesce(msg):
                self._coalescing_buffer[buffer_key] = [msg]
                task = asyncio.create_task(self._fire_coalesced(buffer_key))
                self._coalescing_tasks[buffer_key] = task
                logger.debug(
                    f"Coalescing window opened for {buffer_key} "
                    f"({self._COALESCE_DELAY_SECONDS}s)"
                )
                return

        # ── Normal dispatch (buffer-or-process) ──
        await self._dispatch_inbound(msg)

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
    
    # ── deferred memory consolidation ──

    async def _deferred_consolidation(self, session: "Session") -> None:
        """Run memory consolidation in the background after response delivery."""
        try:
            await self._consolidate_memory(session)
            self.sessions.save(session)
        except Exception as e:
            logger.error(f"Deferred memory consolidation failed: {e}", exc_info=True)

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

    async def _process_message(
        self,
        msg: InboundMessage,
        session_key: str | None = None,
        buffer_key: str | None = None,
    ) -> OutboundMessage | None:
        """
        Process a single inbound message.
        
        Args:
            msg: The inbound message to process.
            session_key: Override session key (used by process_direct).
            buffer_key: Buffer key for mid-loop message injection.
                When set, the agent loop drains buffered messages at
                each step and injects them into the conversation.
        
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

            # Set admin + feedback tool contexts
            if msg.principal_id and principal is not None:
                user_role = getattr(principal, "role", "user") or "user"
                self._admin_tool.set_context(
                    principal_id=msg.principal_id,
                    role=user_role,
                    session_factory=factory,
                    send_callback=self.bus.publish_outbound,
                    chat_type=msg.chat_type or "",
                    sender_id=msg.sender_id or "",
                    channel=msg.channel or "",
                )
                self._feedback_tool.set_context(
                    principal_id=msg.principal_id,
                    role=user_role,
                    session_key=msg.session_key,
                    session_factory=factory,
                    send_callback=self.bus.publish_outbound,
                )
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

        # Flag for deferred consolidation — run AFTER response so the LLM
        # can start immediately (consolidation used to block 30-60s).
        # Group chats get a much larger window so individual user context
        # isn't evicted within hours in a busy room.
        consolidation_threshold = self.memory_window * 3 if is_group else self.memory_window
        _needs_consolidation = len(session.messages) > consolidation_threshold

        # ── Events: check for active time-limited events / hidden personas ──
        event_overlay: str | None = None
        event_announcement: str | None = None
        active_events = self.event_manager.get_active_events(
            message=msg.content or "",
            user_id=msg.principal_id or msg.sender_id,
            chat_id=msg.chat_id or "",
        )
        if active_events:
            event_overlay = self.event_manager.build_overlay_text(active_events)
            event_announcement = self.event_manager.build_announcement(active_events)
            card_theme = self.event_manager.get_card_theme(active_events)
            if card_theme:
                msg.metadata = msg.metadata or {}
                msg.metadata["event_card_theme"] = card_theme
            ev_names = [ae.config.name for ae in active_events]
            logger.info(f"Active events: {ev_names}")
        
        # ── Resolve admin flag and active model for this message ──
        is_admin = False
        if msg.principal_id:
            try:
                from getall.storage.database import get_session_factory as _gsf2
                async with _gsf2()() as _db:
                    from getall.storage.repository import IdentityRepo as _IR2
                    _pr = await _IR2(_db).get_by_id(msg.principal_id)
                    if _pr and getattr(_pr, "role", "user") == "admin":
                        is_admin = True
            except Exception:
                pass

        active_model = await self._get_active_model(msg.chat_type)

        # ── Load system configs for prompt injection ──
        system_configs: dict[str, str] = {}
        try:
            from getall.storage.database import get_session_factory as _gsf3
            from getall.storage.repository import SystemConfigRepo as _SCR
            async with _gsf3()() as _db3:
                _all_cfg = await _SCR(_db3).list_all()
                system_configs = {c.key: c.value for c in _all_cfg}
        except Exception:
            pass

        # Build initial messages (use get_history for LLM-formatted messages)
        # Group chats get a wider history window so the agent sees more
        # per-user context across many concurrent speakers.
        history_window = self.memory_window * 2 if is_group else self.memory_window

        # For audio messages, exclude audio file paths from media sent to LLM.
        llm_media = [
            m for m in (msg.media or [])
            if not m.endswith((".ogg", ".opus", ".mp3", ".wav", ".m4a"))
        ] or None

        # ── Smart media context: separate current vs historical ──────
        _MAX_SESSION_MEDIA = 10
        _MAX_HISTORICAL_INJECT = _MAX_SESSION_MEDIA
        # Migrate legacy key
        if "_recent_images" in session.metadata:
            old = session.metadata.pop("_recent_images")
            if old and "_session_media" not in session.metadata:
                session.metadata["_session_media"] = old
            session.metadata.pop("_recent_images_ts", None)

        # ``llm_media`` = current turn's media (from msg.media).
        # ``historical_media`` = recent session images always injected so
        #   the agent can decide relevance itself (no keyword gating).
        historical_media: list[str] | None = None

        if llm_media:
            # User sent new media → persist but do NOT mix in old images.
            existing: list[str] = session.metadata.get("_session_media", [])
            merged = list(dict.fromkeys(existing + llm_media))
            merged = [p for p in merged if Path(p).is_file()]
            session.metadata["_session_media"] = merged[-_MAX_SESSION_MEDIA:]
            # historical_media stays None — focus on the new content.
        elif session.metadata.get("_session_media"):
            # No new media — always inject recent history so the agent
            # can judge relevance from conversation context.
            valid = [p for p in session.metadata["_session_media"] if Path(p).is_file()]
            if valid:
                historical_media = valid[-_MAX_HISTORICAL_INJECT:]
                logger.debug(
                    f"Injecting {len(historical_media)} historical media "
                    f"for follow-up turn (of {len(valid)} stored)"
                )
            else:
                session.metadata.pop("_session_media", None)

        # Combine for vision-fallback check (both lists contribute).
        _all_media = (llm_media or []) + (historical_media or [])

        # ── Vision fallback: swap to a capable model when images are present ──
        _vision_fallback_active = False
        if _all_media and active_model:
            from getall.config.schema import VISION_CAPABLE_MODELS, VISION_FALLBACK_MODEL
            _has_images = any(
                mimetypes.guess_type(p)[0] and mimetypes.guess_type(p)[0].startswith("image/")  # type: ignore[union-attr]
                for p in _all_media if Path(p).is_file()
            )
            if _has_images and active_model not in VISION_CAPABLE_MODELS:
                logger.info(
                    f"Vision fallback: {active_model} → {VISION_FALLBACK_MODEL} "
                    f"(model lacks vision, {len(_all_media)} media file(s))"
                )
                active_model = VISION_FALLBACK_MODEL
                _vision_fallback_active = True

        messages = self.context.build_messages(
            history=session.get_history(max_messages=history_window),
            current_message=self._format_current_user_content(msg),
            media=llm_media,
            historical_media=historical_media,
            channel=msg.channel,
            chat_id=msg.chat_id,
            chat_type=msg.chat_type,
            persona=persona,
            memory_scope=memory_scope,
            event_overlay=event_overlay,
            is_admin=is_admin,
            active_model=active_model,
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

        # ── Inject system configs (welcome_message_dm, etc.) ──
        if system_configs and messages and messages[0].get("role") == "system":
            cfg_section_parts: list[str] = []
            welcome_dm = system_configs.get("welcome_message_dm")
            if welcome_dm and not is_group:
                cfg_section_parts.append(
                    f"## Welcome Message (Private Chat)\n\n"
                    f"When a new/unregistered user first messages you in private chat, "
                    f"send this welcome message BEFORE anything else:\n\n"
                    f"```\n{welcome_dm}\n```\n\n"
                    f"Only send it once per user (check if they are already onboarded)."
                )
            # Generic system configs (exclude model:* and welcome_message_dm)
            other_cfgs = {
                k: v for k, v in system_configs.items()
                if not k.startswith("model:") and k != "welcome_message_dm"
            }
            if other_cfgs:
                lines = [f"- {k}: {v[:200]}" for k, v in other_cfgs.items()]
                cfg_section_parts.append(
                    f"## System Configs (admin-managed)\n\n" + "\n".join(lines)
                )
            if cfg_section_parts:
                messages[0]["content"] += "\n\n" + "\n\n".join(cfg_section_parts)

        # ── Inject tool execution timing history for duration estimation ──
        self._inject_tool_timings(session, messages)

        # ── Context-window safety: trim history if it would exceed budget ──
        messages = self._fit_messages_to_budget(messages, active_model)

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
        _draft_response: str | None = None  # First text-only response held internally
        
        while iteration < self.max_iterations:
            iteration += 1

            # ── Point A: drain buffered messages before LLM call ──
            # Catches messages that arrived during previous iteration's
            # tool execution (or between loop entry and first LLM call).
            if buffer_key:
                _pending = await self._drain_pending(buffer_key)
                if _pending:
                    llm_media = await self._inject_buffered_messages(
                        _pending, messages, session, llm_media,
                    )

            # Call LLM (use dynamic model resolved from system_config)
            response = await self.provider.chat(
                messages=messages,
                tools=self.tools.get_definitions(),
                model=active_model,
                max_tokens=self.max_tokens,
                reasoning_effort=self.reasoning_effort,
            )
            # Record token usage & cost (fire-and-forget)
            asyncio.ensure_future(self._record_llm_usage(
                response,
                tenant_id=msg.tenant_id or "default",
                principal_id=msg.principal_id or "",
                session_key=session.key,
                call_type="chat",
            ))
            
            # LLM error → stop loop, use error message as final response
            if response.finish_reason == "error":
                final_content = response.content
                logger.warning(f"LLM returned error (iteration {iteration}): {final_content}")
                break

            # Handle tool calls
            if response.has_tool_calls:
                # NOTE: LLM text alongside tool calls is intentionally NOT
                # auto-sent to the user.  It's usually a fragment ("让我调整
                # 一下参数：") that reads awkwardly as a standalone message.
                # The agent controls all user-facing communication itself via
                # the message tool or task-scoped heartbeats.

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
                # No tool calls — check if this is genuinely final.
                content = (response.content or "").strip()

                # ── Point B: drain buffer before any break/nudge ──
                # Catches messages that arrived while the LLM was
                # thinking.  If new messages exist, add the LLM's
                # current response to context, inject the new user
                # messages, and let the LLM respond again.
                if buffer_key:
                    _pending = await self._drain_pending(buffer_key)
                    if _pending:
                        messages = self.context.add_assistant_message(
                            messages, content,
                        )
                        llm_media = await self._inject_buffered_messages(
                            _pending, messages, session, llm_media,
                        )
                        # Reset draft state — the conversation has new
                        # context so any previous draft is stale.
                        _draft_response = None
                        continue

                # ── First text-only response with zero tools used ──
                # The agent may have described plans instead of executing.
                # Save the draft internally (NOT sent to user) and nudge
                # the LLM to follow through.  All user communication goes
                # through `message` tool or `reminders` heartbeat — the
                # loop never auto-sends progress.
                # Skip nudge when the user sent media (image/file/sticker):
                # the model's first response is typically a direct analysis
                # that IS the final answer — nudging just confuses it.
                # Also skip nudge when the response is substantial (≥500
                # chars): long, detailed responses are almost always the
                # complete final answer — nudging causes the LLM to
                # generate a short follow-up that replaces the real content.
                has_media = bool(llm_media)
                _is_substantial = bool(content and len(content) >= 500)
                if (
                    not tools_used
                    and not _draft_response
                    and not has_media
                    and not _is_substantial
                    and iteration < self.max_iterations
                    and content
                    and not self._is_no_reply(content)
                ):
                    _draft_response = content
                    logger.info(
                        f"Draft response saved (not sent), nudging agent: "
                        f"{content[:60]}"
                    )
                    messages = self.context.add_assistant_message(
                        messages, content,
                    )
                    messages.append({
                        "role": "system",
                        "content": (
                            "Your response above has NOT been sent to the user yet. "
                            "Review it carefully:\n"
                            "1. ACTION CHECK: Does your response claim any action "
                            "was COMPLETED (model switched, data fetched, role "
                            "changed, config set, message broadcast, etc.)? "
                            "If YES but you did NOT call the corresponding tool, "
                            "the action did NOT actually happen — you MUST call "
                            "the tool NOW to execute it for real.\n"
                            "2. If you need to fetch data or execute actions, use "
                            "your tools NOW. To tell the user you're working on it, "
                            "call `message(content=\"...\")`. For long tasks, also "
                            "set `reminders(task_scoped=true, delay_seconds=N, "
                            "message=\"...\")` as a heartbeat.\n"
                            "3. If your response is already the complete final "
                            "answer with no unexecuted action claims, respond "
                            "with EXACTLY: [NO_REPLY] and it will be delivered as-is."
                        ),
                    })
                    continue

                # If the LLM returned [NO_REPLY] after a draft, deliver
                # the draft as the final answer.
                if _draft_response and self._is_no_reply(content):
                    final_content = _draft_response
                else:
                    final_content = content
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

        # ── Prepend event announcement (first-trigger only) ──
        if event_announcement and final_content and not self._is_no_reply(final_content):
            final_content = f"{event_announcement}\n\n---\n\n{final_content}"

        metadata = msg.metadata or {}
        strong_trigger = bool(metadata.get("bot_was_mentioned"))

        if is_group and strong_trigger and self._is_no_reply(final_content):
            logger.warning(
                "Model returned NO_REPLY on strong group trigger; forcing retry"
            )
            messages.append({
                "role": "system",
                "content": (
                    "You output [NO_REPLY], but the user explicitly @-mentioned you. "
                    "You MUST reply. Give a brief, natural response to their message."
                ),
            })
            try:
                retry_resp = await self.provider.chat(
                    messages=messages,
                    tools=None,
                    model=active_model,
                    max_tokens=self.max_tokens,
                    reasoning_effort=self.reasoning_effort,
                )
                retry_content = (retry_resp.content or "").strip()
                if retry_content and not self._is_no_reply(retry_content):
                    final_content = retry_content
            except Exception as exc:
                logger.error(f"Strong-trigger retry call failed: {exc}")

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

        # ── Deferred memory consolidation (runs after response, non-blocking) ──
        if _needs_consolidation:
            asyncio.create_task(self._deferred_consolidation(session))

        if suppress_group_reply:
            return None

        # Suppress final outbound ONLY when the agent explicitly marked a
        # same-chat message as final=true (i.e. the full answer was already
        # delivered via the message tool).  Progress messages (final=false,
        # the default) never trigger suppression.
        if "message" in tools_used:
            msg_tool = self.tools.get("message")
            if isinstance(msg_tool, MessageTool) and msg_tool.final_delivered:
                logger.debug(
                    "Suppressing final outbound — agent marked message as final delivery"
                )
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

    @staticmethod
    def _build_autofix_hint(tool_name: str, result: str) -> str | None:
        """Nudge the agent to recover when a tool call errors.

        No keyword matching — the agent reads the actual error and decides
        how to fix it.  We only detect *that* an error occurred, never
        *what kind*.
        """
        text = (result or "").strip()
        if not text:
            return None
        low = text.lower()
        if not (low.startswith("error") or "stderr:" in low):
            return None
        return (
            "The tool call above returned an error. "
            "Read the error carefully, diagnose the root cause, "
            "and recover autonomously — try a different approach, "
            "a different data source, or a different tool. "
            "Do not give up after one failure."
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
        
        # Context-window safety
        messages = self._fit_messages_to_budget(messages, active_model)

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
            asyncio.ensure_future(self._record_llm_usage(
                response,
                tenant_id=msg.tenant_id or "default",
                principal_id=msg.principal_id or "",
                session_key=session_key,
                call_type="chat",
            ))

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
        """Consolidate old messages into MEMORY.md + HISTORY.md, then trim session.

        Group chats get a larger keep_count and per-sender preservation
        so individual user context isn't lost in high-traffic rooms.
        """
        memory_scope = str(session.metadata.get("memory_scope") or "global")
        memory = MemoryStore(self.workspace, memory_scope)
        is_group = memory_scope.startswith("group")

        # ── Compute keep_count (group-aware) ──
        if is_group:
            # In a busy group, 10 messages evaporate in minutes.
            # Keep more context so the agent can still see each user's recent activity.
            keep_count = min(30, max(10, self.memory_window))
        else:
            keep_count = min(10, max(2, self.memory_window // 2))

        # ── Per-sender preservation for group chats ──
        # Guarantee at least the last 2 messages per active sender survive,
        # even if they spoke earlier than the global keep_count window.
        if is_group:
            cutoff_idx = max(0, len(session.messages) - keep_count)
            # Collect indices of each sender's most recent messages
            sender_last: dict[str, list[int]] = {}
            for idx, m in enumerate(session.messages):
                if m.get("role") == "user":
                    sender = str(
                        m.get("sender_name") or m.get("sender_id") or ""
                    ).strip()
                    if sender:
                        sender_last.setdefault(sender, []).append(idx)

            preserve_indices: set[int] = set()
            for indices in sender_last.values():
                # Keep last 2 messages per sender
                for i in indices[-2:]:
                    if i < cutoff_idx:
                        preserve_indices.add(i)

            if preserve_indices:
                # Merge preserved messages into the kept window
                kept_msgs = [
                    session.messages[i]
                    for i in sorted(preserve_indices)
                ] + session.messages[cutoff_idx:]
                old_messages = [
                    m for idx, m in enumerate(session.messages)
                    if idx < cutoff_idx and idx not in preserve_indices
                ]
            else:
                old_messages = session.messages[:-keep_count]
                kept_msgs = session.messages[-keep_count:]
        else:
            old_messages = session.messages[:-keep_count]
            kept_msgs = session.messages[-keep_count:]

        if not old_messages:
            return
        logger.info(
            "Memory consolidation started: "
            f"{len(session.messages)} messages, archiving {len(old_messages)}, "
            f"keeping {len(kept_msgs)}, scope={memory_scope}"
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

        group_instruction = ""
        if is_group:
            group_instruction = (
                "\n\nIMPORTANT — This is a GROUP chat consolidation. "
                "Attribute actions/requests to specific users by name. "
                "For each user who made a request or received a task, "
                "record WHO asked, WHAT they asked, and the OUTCOME. "
                "Example: 'Bill Lang asked for BTC/ETH pattern analysis; "
                "agent delivered a 3-scenario prediction chart.' "
                "Include any PENDING/UNFINISHED tasks with the user's name."
            )

        prompt = f"""You are a memory consolidation agent. Process this conversation and return a JSON object with exactly two keys:

1. "history_entry": A paragraph (2-5 sentences) summarizing the key events/decisions/topics. Start with a timestamp like [YYYY-MM-DD HH:MM]. Include enough detail to be useful when found by grep search later. Attribute actions to specific users by name when applicable.{group_instruction}

2. "memory_update": The updated long-term memory content. Add any new facts: user location, preferences, personal info, habits, project context, technical decisions, tools/services used, pending tasks per user. If nothing new, return the existing content unchanged.

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
            asyncio.ensure_future(self._record_llm_usage(
                response,
                session_key=session.key,
                call_type="consolidation",
            ))
            import json as _json
            text = (response.content or "").strip()
            result = self._extract_consolidation_json(text)

            if entry := result.get("history_entry"):
                memory.append_history(entry)
            if update := result.get("memory_update"):
                if update != current_memory:
                    memory.write_long_term(update)

            logger.info(
                f"Memory consolidation done, session trimmed "
                f"{len(session.messages)} → {len(kept_msgs)}"
            )
        except Exception as e:
            logger.error(f"Memory consolidation failed: {e}")
        finally:
            # Always trim session to prevent unbounded growth,
            # even when consolidation fails.
            session.messages = kept_msgs
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
        
        buffer_key = self._buffer_key_for_message(msg, session_key=session_key)
        response = await self._process_message(
            msg, session_key=session_key, buffer_key=buffer_key,
        )
        return response.content if response else ""
