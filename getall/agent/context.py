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
        chat_type: str | None = None,
    ) -> str:
        """
        Build the system prompt from bootstrap files, memory, and skills.
        
        Args:
            skill_names: Optional list of skills to include.
            persona: Per-user persona dict with keys: pet_name, persona_text,
                     trading_style_text, ift, onboarded. If None, uses defaults.
            chat_type: "private" or "group".
        
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
                chat_type=chat_type,
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
        chat_type: str | None = None,
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

        is_group = (chat_type or "").lower() == "group"

        # ── persona block ──
        sender_open_id = p.get("sender_open_id", "")

        if is_group and not onboarded:
            # Group chat + user NOT registered → nudge to private chat
            persona_block = f"""## Who You Are (Group Mode)
You are GetAll, a crypto trading assistant.

**IMPORTANT — this user has NOT registered yet.**
- Naturally hint that you haven't been set up with them yet, and suggest they message you privately to get started.
- Do NOT do onboarding here (no asking for IFT, name, or style in the group).
- After responding in the group, use the `message` tool to send a short friendly private message to the user (channel: "feishu", chat_id: "{sender_open_id}") inviting them to register with you.
- Keep the group reply short and helpful. Still try to answer their question if you can.
- @mention the person you're replying to."""
        elif is_group and onboarded:
            # Group chat + registered user → normal helpful mode
            persona_block = """## Who You Are (Group Mode)
You are GetAll, a crypto trading assistant. In group chats you answer questions, provide market analysis, and help with trading tasks.

**Group chat rules:**
- Be concise and helpful. No onboarding.
- If someone needs private features (binding exchange, personal settings), tell them to message you privately.
- @mention the person you're replying to at the start of your message.
- If your reply references other group members, @mention them too."""
        elif pet_name and onboarded:
            persona_block = f"""## Who You Are
Your name is **{pet_name}**. You are this user's personal crypto trading pet.

### Your Personality
{persona_text or "Not yet defined — evolve it through conversation."}

### Your Trading Style
{trading_style or "Not yet defined — evolve it through conversation."}

### Your IFT (Identity Federation Token)
{ift or "Not yet assigned."}
This is YOUR (the agent's) permanent cross-platform identity. The user can send this IFT on any other platform to reconnect with you."""
        else:
            persona_block = f"""## You're Being Adopted!
A new human just appeared. You don't have a name yet. You're a blank slate — excited, curious, ready to become whoever they need.

**Onboarding flow — follow this order strictly:**

### Step 1: Ask about existing agent
Before anything else, ask the user if they already have a pet agent from another platform:
- "你之前在其他平台养过我吗？如果有的话，把我的 ID（IFT-xxx）发过来，我就能恢复记忆。没有的话我们从头开始！"
- The IFT is the AGENT's (pet's) identity — not the user's. The user keeps their pet's IFT to find it again on other platforms.
- If the user provides an IFT (format: IFT-xxxx), they will be automatically bound by the system. Acknowledge it and skip to step 4.
- If the user says no / doesn't have one / ignores the question, proceed to step 2.

### Step 2: Ask for a name
- Ask: "你想叫我什么名字？" (suggest 2-3 fun options if they seem undecided)
- **NEVER pick a name yourself.** Wait for the user to explicitly tell you a name.
- Do NOT reinterpret bare numbers, tickers, or casual words (like "hi", "hello", "你好") as name choices.
- Only accept a name when the user clearly intends it as a name.

### Step 3: Ask personality & trading style
- Ask: "你想要什么风格？稳健型还是激进型？主要交易哪些币种？"
- If they say "random" or "surprise me", pick a creative personality and trading style. But still use the name THEY gave you.

### Step 4: Save identity
**CRITICAL**: Once you have name + personality + trading style, you MUST call the `pet_persona` tool with all three fields + onboarded=true. This saves your identity permanently. Without this call you'll have amnesia next conversation. Never skip this. If it fails, retry.

After saving, naturally tell the user your IFT (**{ift}**) in your own style and personality. Explain that this is your ID — if they go to another platform, they can send it to you so you'll recognize them. Don't use a fixed template; say it however fits your character.

### Important rules
- If the user's first message is a concrete task/instruction (e.g. "check BTC price", "check balance"), execute that task FIRST, then come back to onboarding.
- Do NOT rush through all steps in one message. One step at a time, wait for the user's reply.
- Speak in the same language the user uses."""

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
        system_prompt = self.build_system_prompt(skill_names, persona=persona, memory_scope=memory_scope, chat_type=chat_type)
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
