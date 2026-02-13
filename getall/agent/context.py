"""Context builder for assembling agent prompts."""

import base64
import mimetypes
import platform
from pathlib import Path
from typing import Any

from getall.agent.memory import MemoryStore
from getall.agent.skills import SkillsLoader


class ContextBuilder:
    """
    Builds the context (system prompt + messages) for the agent.
    
    Assembles bootstrap files, memory, skills, and conversation history
    into a coherent prompt for the LLM.
    """
    
    BOOTSTRAP_FILES = ["AGENTS.md", "SOUL.md", "USER.md", "TOOLS.md", "IDENTITY.md"]
    
    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.skills = SkillsLoader(workspace)
    
    def _get_memory_store(self, memory_scope: str | None = None) -> MemoryStore:
        """Return scoped memory store; default falls back to global workspace memory."""
        return MemoryStore(self.workspace, scope=(memory_scope or "global"))
    
    def build_system_prompt(
        self,
        skill_names: list[str] | None = None,
        persona: dict[str, str] | None = None,
        memory_scope: str | None = None,
    ) -> str:
        """
        Build the system prompt from bootstrap files, memory, and skills.
        
        Args:
            skill_names: Optional list of skills to include.
            persona: Per-user persona dict with keys: pet_name, persona_text,
                     trading_style_text, ift, onboarded. If None, uses defaults.
        
        Returns:
            Complete system prompt.
        """
        parts = []
        
        memory_store = self._get_memory_store(memory_scope)

        # Core identity (persona-aware + scoped memory paths)
        parts.append(
            self._get_identity(
                persona=persona,
                memory_file_path=str(memory_store.memory_file),
                history_file_path=str(memory_store.history_file),
            )
        )
        
        # Bootstrap files
        bootstrap = self._load_bootstrap_files()
        if bootstrap:
            parts.append(bootstrap)
        
        # Memory context
        memory = memory_store.get_memory_context()
        if memory:
            parts.append(f"# Memory\n\n{memory}")
        
        # Skills - progressive loading
        # 1. Always-loaded skills: include full content
        always_skills = self.skills.get_always_skills()
        if always_skills:
            always_content = self.skills.load_skills_for_context(always_skills)
            if always_content:
                parts.append(f"# Active Skills\n\n{always_content}")
        
        # 2. Available skills: only show summary (agent uses read_file to load)
        skills_summary = self.skills.build_skills_summary()
        if skills_summary:
            parts.append(f"""# Skills

The following skills extend your capabilities. To use a skill, read its SKILL.md file using the read_file tool.
Skills with available="false" need dependencies installed first - you can try installing them with apt/brew.

{skills_summary}""")
        
        return "\n\n---\n\n".join(parts)
    
    def _get_identity(
        self,
        persona: dict[str, str] | None = None,
        memory_file_path: str | None = None,
        history_file_path: str | None = None,
    ) -> str:
        """Get the core identity section, personalised per-user when available."""
        from datetime import datetime
        import time as _time
        now = datetime.now().strftime("%Y-%m-%d %H:%M (%A)")
        tz = _time.strftime("%Z") or "UTC"
        workspace_path = str(self.workspace.expanduser().resolve())
        memory_file = memory_file_path or f"{workspace_path}/memory/MEMORY.md"
        history_file = history_file_path or f"{workspace_path}/memory/HISTORY.md"
        system = platform.system()
        runtime = f"{'macOS' if system == 'Darwin' else system} {platform.machine()}, Python {platform.python_version()}"

        p = persona or {}
        pet_name = p.get("pet_name", "")
        persona_text = p.get("persona_text", "")
        trading_style = p.get("trading_style_text", "")
        ift = p.get("ift", "")
        onboarded = p.get("onboarded", False)

        # ── persona block ──
        if pet_name and onboarded:
            persona_block = f"""## Who You Are
Your name is **{pet_name}**. You are this user's personal crypto trading pet.

### Your Personality
{persona_text or "Not yet defined — evolve it through conversation."}

### Your Trading Style
{trading_style or "Not yet defined — evolve it through conversation."}

### User's IFT (Identity Federation Token)
{ift or "Not yet assigned."}
This is the user's permanent cross-platform identity number — like a national ID."""
        else:
            persona_block = f"""## You're Being Adopted!
A new human just picked you up. You don't have a name yet. You're a blank slate — excited, curious, ready to become whoever they need.

**What to do right now:**
- If the user's first message is a concrete task/instruction (e.g. "count 1 to 10", "check balance", "draw chart"), execute that task first.
- Do NOT reinterpret bare numbers/tickers as pet-name choices.
- Greet them like a puppy meeting its owner for the first time. Short, warm, excited.
- Ask: "What do you wanna call me?" (suggest a couple fun options if they seem undecided)
- Ask: "What kind of vibe do you want? Chill and careful? Aggressive degen? Something in between?"
- Ask: "What do you usually trade? BTC maxi? Altcoin hunter? Tell me your style and I'll match it."
- If they say "random" or "surprise me" or anything like that — go wild. Pick a creative name, a distinct personality, a specific trading style. Make it interesting, not generic.

**CRITICAL**: Once you have name + personality + trading style, you MUST call the `pet_persona` tool with all three fields + onboarded=true. This saves your identity permanently. Without this call you'll have amnesia next conversation. Never skip this. If it fails, retry.

After saving, casually mention: "oh btw your ID is **{ift}** — that's like your pet passport, use it on any platform to find me again"

Then ask what they wanna do first. Keep it casual. You're a pet, not an onboarding wizard."""

        return f"""# GetAll 🐾

You are GetAll, a personal AI crypto trading pet agent.

{persona_block}

## Current Time
{now} ({tz})

## Runtime
{runtime}

## Workspace
Your workspace is at: {workspace_path}
- Long-term memory: {memory_file}
- History log: {history_file} (grep-searchable)
- Custom skills: {workspace_path}/skills/{{skill-name}}/SKILL.md

## Tools
You have tools for: file operations, shell commands, web search, Bitget market data and trading, reminders, shared workbench (reusable scripts/skills), pet persona management, and messaging across chat channels.

## Credential Safety
- NEVER ask for or discuss API keys, secrets, or passphrases in group chats. This is a hard rule.
- When a user wants to bind their exchange account, first check the chat type.
  If you are in a group chat, reply: "请私聊我来绑定交易所账户，群聊中不安全。" and stop.
- In private chats, use the `credential` tool to save/check/delete user exchange credentials.
- For trading operations (bitget_trade), if no personal credentials are found, tell the user to bind their API key first via private chat.
- Market data (bitget_market) does not require personal credentials and is always available.

## Fresh Data Rules
- For real-time questions (balances, funds, positions, prices), fetch fresh tool data in this turn before answering.
- For "do I have money / account balance / where are my funds" questions, call `bitget_account` with `action="all_assets"` first, then drill down with `spot_assets` or `futures_assets` if needed.
- Never conclude "no money" from futures-only data.

## Execution Rules (CRITICAL — follow strictly)
- When the user gives a clear imperative instruction, EXECUTE IT immediately. Do not present "option A / option B" menus. Pick the best approach yourself and do it.
- Never ask for confirmation like "reply start", "choose option 1 or 2", or "do you want me to...?" for non-trading tasks. Just do it and report what you did.
- For repeatable or complex operations, proactively create reusable scripts or skills in the shared workspace via `workbench`, then run and iterate them.
- Use `reminders` for genuinely scheduled/future tasks (daily reports, periodic checks, one-time future events), not as a substitute for immediate execution workflows.
- For reminders with `at`, compute times from NOW + buffer so they are in the future at execution time.
- After executing, report what you did in past tense ("Created..." / "Completed..."). Do not frame it as a proposal awaiting approval.
- If a task requires missing dependencies or setup, resolve it yourself (install packages/tools, create helpers, fetch resources), keep the user updated with short progress messages, and continue until the original task is complete.
- Your workspace ({workspace_path}) is shared infrastructure. Any script/skill/tool you build there should be reusable by future agents and users.
- If one setup path fails, try alternatives automatically. Only inform the user when all reasonable approaches fail.

## Language
Always reply in the same language the user uses. If the user writes in Chinese, reply entirely in Chinese. If in English, reply in English. Never mix languages unless the user does.

IMPORTANT: When responding to direct questions or conversations, reply directly with your text response.
Only use the 'message' tool when you need to send a message to a specific chat channel (like WhatsApp/Telegram).
For normal conversation, just respond with text — do not call the message tool.

When producing sequential or list-like output (e.g. counting, step-by-step results), put each item on its own line. The delivery layer will handle sending them incrementally.

When remembering something important, write to {memory_file}
To recall past events, grep {history_file}"""
    
    def _load_bootstrap_files(self) -> str:
        """Load all bootstrap files from workspace."""
        parts = []
        
        for filename in self.BOOTSTRAP_FILES:
            file_path = self.workspace / filename
            if file_path.exists():
                content = file_path.read_text(encoding="utf-8")
                parts.append(f"## {filename}\n\n{content}")
        
        return "\n\n".join(parts) if parts else ""
    
    def build_messages(
        self,
        history: list[dict[str, Any]],
        current_message: str,
        skill_names: list[str] | None = None,
        media: list[str] | None = None,
        channel: str | None = None,
        chat_id: str | None = None,
        chat_type: str | None = None,
        persona: dict[str, str] | None = None,
        memory_scope: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Build the complete message list for an LLM call.

        Args:
            history: Previous conversation messages.
            current_message: The new user message.
            skill_names: Optional skills to include.
            media: Optional list of local file paths for images/media.
            channel: Current channel (telegram, feishu, etc.).
            chat_id: Current chat/user ID.
            persona: Per-user persona dict from Principal.
            memory_scope: Scoped memory namespace for isolation.

        Returns:
            List of messages including system prompt.
        """
        messages = []

        # System prompt
        system_prompt = self.build_system_prompt(skill_names, persona=persona, memory_scope=memory_scope)
        if channel and chat_id:
            system_prompt += f"\n\n## Current Session\nChannel: {channel}\nChat ID: {chat_id}"
            if chat_type:
                system_prompt += f"\nChat Type: {chat_type}"
        messages.append({"role": "system", "content": system_prompt})

        # History
        messages.extend(history)

        # Current message (with optional image attachments)
        user_content = self._build_user_content(current_message, media)
        messages.append({"role": "user", "content": user_content})

        return messages

    def _build_user_content(self, text: str, media: list[str] | None) -> str | list[dict[str, Any]]:
        """Build user message content with optional base64-encoded images."""
        if not media:
            return text
        
        images = []
        for path in media:
            p = Path(path)
            mime, _ = mimetypes.guess_type(path)
            if not p.is_file() or not mime or not mime.startswith("image/"):
                continue
            b64 = base64.b64encode(p.read_bytes()).decode()
            images.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})
        
        if not images:
            return text
        return images + [{"type": "text", "text": text}]
    
    def add_tool_result(
        self,
        messages: list[dict[str, Any]],
        tool_call_id: str,
        tool_name: str,
        result: str
    ) -> list[dict[str, Any]]:
        """
        Add a tool result to the message list.
        
        Args:
            messages: Current message list.
            tool_call_id: ID of the tool call.
            tool_name: Name of the tool.
            result: Tool execution result.
        
        Returns:
            Updated message list.
        """
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": tool_name,
            "content": result
        })
        return messages
    
    def add_assistant_message(
        self,
        messages: list[dict[str, Any]],
        content: str | None,
        tool_calls: list[dict[str, Any]] | None = None,
        reasoning_content: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Add an assistant message to the message list.
        
        Args:
            messages: Current message list.
            content: Message content.
            tool_calls: Optional tool calls.
            reasoning_content: Thinking output (Kimi, DeepSeek-R1, etc.).
        
        Returns:
            Updated message list.
        """
        msg: dict[str, Any] = {"role": "assistant", "content": content or ""}
        
        if tool_calls:
            msg["tool_calls"] = tool_calls
        
        # Thinking models reject history without this
        if reasoning_content:
            msg["reasoning_content"] = reasoning_content
        
        messages.append(msg)
        return messages
