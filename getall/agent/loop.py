"""Agent loop: the core processing engine."""

import asyncio
import json
from pathlib import Path
from typing import Any

from loguru import logger

from getall.bus.events import InboundMessage, OutboundMessage
from getall.bus.queue import MessageBus
from getall.providers.base import LLMProvider
from getall.agent.context import ContextBuilder
from getall.agent.tools.registry import ToolRegistry
from getall.agent.tools.filesystem import ReadFileTool, WriteFileTool, EditFileTool, ListDirTool
from getall.agent.tools.shell import ExecTool
from getall.agent.tools.web import WebSearchTool, WebFetchTool
from getall.agent.tools.message import MessageTool
from getall.agent.tools.spawn import SpawnTool
from getall.agent.tools.cron import ReminderTool
from getall.agent.tools.workbench import WorkbenchTool
from getall.agent.tools.bitget import BitgetMarketTool, BitgetAccountTool, BitgetTradeTool
from getall.agent.tools.credential import CredentialTool
from getall.agent.tools.persona import PetPersonaTool
from getall.agent.memory import MemoryStore
from getall.agent.subagent import SubagentManager
from getall.session.manager import SessionManager


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
    
    def __init__(
        self,
        bus: MessageBus,
        provider: LLMProvider,
        workspace: Path,
        model: str | None = None,
        max_iterations: int = 20,
        memory_window: int = 50,
        brave_api_key: str | None = None,
        exec_config: "ExecToolConfig | None" = None,
        cron_service: "CronService | None" = None,
        restrict_to_workspace: bool = False,
        session_manager: SessionManager | None = None,
    ):
        from getall.config.schema import ExecToolConfig
        from getall.cron.service import CronService
        self.bus = bus
        self.provider = provider
        self.workspace = workspace
        self.model = model or provider.get_default_model()
        self.max_iterations = max_iterations
        self.memory_window = memory_window
        self.brave_api_key = brave_api_key
        self.exec_config = exec_config or ExecToolConfig()
        self.cron_service = cron_service
        self.restrict_to_workspace = restrict_to_workspace
        
        self.context = ContextBuilder(workspace)
        self.sessions = session_manager or SessionManager(workspace)
        self.tools = ToolRegistry()
        self.subagents = SubagentManager(
            provider=provider,
            workspace=workspace,
            bus=bus,
            model=self.model,
            brave_api_key=brave_api_key,
            exec_config=self.exec_config,
            restrict_to_workspace=restrict_to_workspace,
        )
        
        self._running = False
        self._register_default_tools()
    
    def _register_default_tools(self) -> None:
        """Register the default set of tools."""
        # File tools (restrict to workspace if configured)
        allowed_dir = self.workspace if self.restrict_to_workspace else None
        self.tools.register(ReadFileTool(allowed_dir=allowed_dir))
        self.tools.register(WriteFileTool(allowed_dir=allowed_dir))
        self.tools.register(EditFileTool(allowed_dir=allowed_dir))
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
        self.tools.register(WebSearchTool(api_key=self.brave_api_key))
        self.tools.register(WebFetchTool())
        
        # Message tool
        message_tool = MessageTool(send_callback=self.bus.publish_outbound)
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

        # Credential tool (save / check / delete exchange API keys)
        self._credential_tool = CredentialTool()
        self.tools.register(self._credential_tool)

        # Pet persona tool
        self._persona_tool = PetPersonaTool()
        self.tools.register(self._persona_tool)
    
    async def run(self) -> None:
        """Run the agent loop, processing messages from the bus."""
        self._running = True
        logger.info("Agent loop started")
        
        while self._running:
            try:
                # Wait for next message
                msg = await asyncio.wait_for(
                    self.bus.consume_inbound(),
                    timeout=1.0
                )
                
                # Process it
                try:
                    response = await self._process_message(msg)
                    if response:
                        await self.bus.publish_outbound(response)
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    # Send error response
                    await self.bus.publish_outbound(OutboundMessage(
                        channel=msg.channel,
                        chat_id=msg.chat_id,
                        content=f"Sorry, I encountered an error: {str(e)}"
                    ))
            except asyncio.TimeoutError:
                continue
    
    def stop(self) -> None:
        """Stop the agent loop."""
        self._running = False
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
        
        preview = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
        logger.info(f"Processing message from {msg.channel}:{msg.sender_id}: {preview}")
        
        # Get or create session
        session = self.sessions.get_or_create(session_key or msg.session_key)
        
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
                    resolution = await hook.resolve(
                        tenant_id=msg.tenant_id or "default",
                        channel=msg.channel,
                        chat_id=msg.chat_id,
                        user_id=msg.sender_id,
                        thread_id=msg.thread_id,
                        message_text=msg.content,
                    )
                    await db_session.commit()
                    msg.principal_id = resolution.principal_id
                    msg.agent_identity_id = resolution.agent_identity_id
                    logger.info(f"Identity resolved: {msg.channel}:{msg.sender_id} -> {resolution.ift}")

            # Step 2: load persona — only for private chats.
            # In group chats the bot keeps a stable default personality
            # (from SOUL.md) rather than switching per-sender.
            if msg.principal_id and not is_group:
                async with factory() as db_session:
                    repo = IdentityRepo(db_session)
                    principal = await repo.get_by_id(msg.principal_id)
                    if principal is not None:
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
            )

        memory_scope = self._resolve_memory_scope(msg)
        session.metadata["memory_scope"] = memory_scope

        # Consolidate memory with correct scope once identity/session metadata are ready.
        if len(session.messages) > self.memory_window:
            await self._consolidate_memory(session)
        
        # Build initial messages (use get_history for LLM-formatted messages)
        messages = self.context.build_messages(
            history=session.get_history(),
            current_message=msg.content,
            media=msg.media if msg.media else None,
            channel=msg.channel,
            chat_id=msg.chat_id,
            chat_type=msg.chat_type,
            persona=persona,
            memory_scope=memory_scope,
        )
        
        # Agent loop
        iteration = 0
        final_content = None
        tools_used: list[str] = []
        
        while iteration < self.max_iterations:
            iteration += 1
            
            # Call LLM
            response = await self.provider.chat(
                messages=messages,
                tools=self.tools.get_definitions(),
                model=self.model
            )
            
            # Handle tool calls
            if response.has_tool_calls:
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
                
                # Execute tools
                for tool_call in response.tool_calls:
                    tools_used.append(tool_call.name)
                    args_str = json.dumps(tool_call.arguments, ensure_ascii=False)
                    logger.info(f"Tool call: {tool_call.name}({args_str[:200]})")
                    result = await self.tools.execute(tool_call.name, tool_call.arguments)
                    messages = self.context.add_tool_result(
                        messages, tool_call.id, tool_call.name, result
                    )
            else:
                # No tool calls, we're done
                final_content = response.content
                break
        
        if final_content is None:
            final_content = "I've completed processing but have no response to give."
        
        # Log response preview
        preview = final_content[:120] + "..." if len(final_content) > 120 else final_content
        logger.info(f"Response to {msg.channel}:{msg.sender_id}: {preview}")
        
        # Save to session (include tool names so consolidation sees what happened)
        is_synthetic_event = bool((msg.metadata or {}).get("synthetic"))
        if not is_synthetic_event:
            session.add_message("user", msg.content)
        session.add_message("assistant", final_content,
                            tools_used=tools_used if tools_used else None)
        self.sessions.save(session)

        # If the agent already sent messages via the message tool,
        # suppress the final outbound to avoid duplicate output.
        if "message" in tools_used:
            logger.debug("Suppressing final outbound — messages already sent via message tool")
            return None
        
        return OutboundMessage(
            channel=msg.channel,
            chat_id=msg.chat_id,
            content=final_content,
            metadata=msg.metadata or {},  # Pass through for channel-specific needs (e.g. Slack thread_ts)
        )
    
    async def _process_system_message(self, msg: InboundMessage) -> OutboundMessage | None:
        """
        Process a system message (e.g., subagent announce).
        
        The chat_id field contains "original_channel:original_chat_id" to route
        the response back to the correct destination.
        """
        logger.info(f"Processing system message from {msg.sender_id}")
        
        # Parse origin from chat_id (format: "channel:chat_id")
        if ":" in msg.chat_id:
            parts = msg.chat_id.split(":", 1)
            origin_channel = parts[0]
            origin_chat_id = parts[1]
        else:
            # Fallback
            origin_channel = "cli"
            origin_chat_id = msg.chat_id
        
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
                model=self.model
            )
            
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
            tools = f" [tools: {', '.join(m['tools_used'])}]" if m.get("tools_used") else ""
            lines.append(f"[{m.get('timestamp', '?')[:16]}] {m['role'].upper()}{tools}: {m['content']}")
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
            # Strip markdown fences that LLMs often add despite instructions
            if text.startswith("```"):
                text = text.split("\n", 1)[-1].rsplit("```", 1)[0].strip()
            result = _json.loads(text)

            if entry := result.get("history_entry"):
                memory.append_history(entry)
            if update := result.get("memory_update"):
                if update != current_memory:
                    memory.write_long_term(update)

            # Trim session to recent messages
            session.messages = session.messages[-keep_count:]
            self.sessions.save(session)
            logger.info(f"Memory consolidation done, session trimmed to {len(session.messages)} messages")
        except Exception as e:
            logger.error(f"Memory consolidation failed: {e}")

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
        )
        
        response = await self._process_message(msg, session_key=session_key)
        return response.content if response else ""
