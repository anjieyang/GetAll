"""CLI commands for GetAll."""

import asyncio
import os
import signal
from pathlib import Path
import select
import sys

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table
from rich.text import Text

from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.patch_stdout import patch_stdout

from getall import __version__, __logo__

app = typer.Typer(
    name="getall",
    help=f"{__logo__} GetAll - AI-Native Crypto Pet Agent",
    no_args_is_help=True,
)

console = Console()
EXIT_COMMANDS = {"exit", "quit", "/exit", "/quit", ":q"}

# ---------------------------------------------------------------------------
# CLI input: prompt_toolkit for editing, paste, history, and display
# ---------------------------------------------------------------------------

_PROMPT_SESSION: PromptSession | None = None
_SAVED_TERM_ATTRS = None  # original termios settings, restored on exit


def _flush_pending_tty_input() -> None:
    """Drop unread keypresses typed while the model was generating output."""
    try:
        fd = sys.stdin.fileno()
        if not os.isatty(fd):
            return
    except Exception:
        return

    try:
        import termios
        termios.tcflush(fd, termios.TCIFLUSH)
        return
    except Exception:
        pass

    try:
        while True:
            ready, _, _ = select.select([fd], [], [], 0)
            if not ready:
                break
            if not os.read(fd, 4096):
                break
    except Exception:
        return


def _restore_terminal() -> None:
    """Restore terminal to its original state (echo, line buffering, etc.)."""
    if _SAVED_TERM_ATTRS is None:
        return
    try:
        import termios
        termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, _SAVED_TERM_ATTRS)
    except Exception:
        pass


def _init_prompt_session() -> None:
    """Create the prompt_toolkit session with persistent file history."""
    global _PROMPT_SESSION, _SAVED_TERM_ATTRS

    # Save terminal state so we can restore it on exit
    try:
        import termios
        _SAVED_TERM_ATTRS = termios.tcgetattr(sys.stdin.fileno())
    except Exception:
        pass

    history_file = Path.home() / ".getall" / "history" / "cli_history"
    history_file.parent.mkdir(parents=True, exist_ok=True)

    _PROMPT_SESSION = PromptSession(
        history=FileHistory(str(history_file)),
        enable_open_in_editor=False,
        multiline=False,   # Enter submits (single line mode)
    )


def _print_agent_response(response: str, render_markdown: bool) -> None:
    """Render assistant response with consistent terminal styling."""
    content = response or ""
    body = Markdown(content) if render_markdown else Text(content)
    console.print()
    console.print(f"[cyan]{__logo__} GetAll[/cyan]")
    console.print(body)
    console.print()


def _is_exit_command(command: str) -> bool:
    """Return True when input should end interactive chat."""
    return command.lower() in EXIT_COMMANDS


async def _read_interactive_input_async() -> str:
    """Read user input using prompt_toolkit (handles paste, history, display).

    prompt_toolkit natively handles:
    - Multiline paste (bracketed paste mode)
    - History navigation (up/down arrows)
    - Clean display (no ghost characters or artifacts)
    """
    if _PROMPT_SESSION is None:
        raise RuntimeError("Call _init_prompt_session() first")
    try:
        with patch_stdout():
            return await _PROMPT_SESSION.prompt_async(
                HTML("<b fg='ansiblue'>You:</b> "),
            )
    except EOFError as exc:
        raise KeyboardInterrupt from exc



def version_callback(value: bool):
    if value:
        console.print(f"{__logo__} GetAll v{__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        None, "--version", "-v", callback=version_callback, is_eager=True
    ),
):
    """GetAll - AI-Native Crypto Pet Agent."""
    pass


# ============================================================================
# Onboard / Setup
# ============================================================================


@app.command()
def onboard():
    """Initialize GetAll configuration and workspace."""
    from getall.config.loader import get_config_path, save_config
    from getall.config.schema import Config
    from getall.utils.helpers import get_workspace_path
    
    config_path = get_config_path()
    
    if config_path.exists():
        console.print(f"[yellow]Config already exists at {config_path}[/yellow]")
        if not typer.confirm("Overwrite?"):
            raise typer.Exit()
    
    # Create default config
    config = Config()
    save_config(config)
    console.print(f"[green]✓[/green] Created config at {config_path}")
    
    # Create workspace
    workspace = get_workspace_path()
    console.print(f"[green]✓[/green] Created workspace at {workspace}")
    
    # Create default bootstrap files
    _create_workspace_templates(workspace)
    
    console.print(f"\n{__logo__} GetAll is ready!")
    console.print("\nNext steps:")
    console.print("  1. Add your API key to [cyan]~/.getall/config.json[/cyan]")
    console.print("     Get one at: https://openrouter.ai/keys")
    console.print("  2. Chat: [cyan]getall agent -m \"Hello!\"[/cyan]")
    console.print("\n[dim]Want Telegram/WhatsApp? Configure channels in config.json[/dim]")




def _find_bundled_workspace() -> Path | None:
    """Locate the bundled workspace/ directory shipped with the package.

    Checks (in order):
    1. Relative to source tree (development mode).
    2. /app/workspace (Docker image).
    """
    candidates = [
        Path(__file__).parent.parent.parent / "workspace",  # dev: repo root
        Path("/app/workspace"),  # Docker
    ]
    for p in candidates:
        if p.is_dir() and (p / "SOUL.md").exists():
            return p
    return None


def _create_workspace_templates(workspace: Path):
    """Create default workspace bootstrap files.

    Files are only created if they don't already exist, so user
    customisations are never overwritten.
    """
    _pkg_ws = _find_bundled_workspace()

    templates: dict[str, str] = {}
    if _pkg_ws:
        for name in ("AGENTS.md", "SOUL.md", "HEARTBEAT.md"):
            src = _pkg_ws / name
            if src.exists():
                templates[name] = src.read_text(encoding="utf-8")

    # USER.md is a lightweight starter (not bundled)
    templates.setdefault("USER.md", """# User

Information about the user goes here.

## Preferences

- Communication style: (casual/formal)
- Timezone: (your timezone)
- Language: (your preferred language)
""")
    
    for filename, content in templates.items():
        file_path = workspace / filename
        if not file_path.exists():
            file_path.write_text(content)
            console.print(f"  [dim]Created {filename}[/dim]")
    
    # Sync events directory (copy missing YAML files, never overwrite existing)
    if _pkg_ws:
        _sync_events(workspace, _pkg_ws)

    # Create memory directories
    memory_dir = workspace / "memory"
    memory_dir.mkdir(exist_ok=True)
    (memory_dir / "trading").mkdir(exist_ok=True)
    (memory_dir / "profile").mkdir(exist_ok=True)

    memory_file = memory_dir / "MEMORY.md"
    if not memory_file.exists():
        memory_file.write_text("# Long-term Memory\n\nImportant information that persists across sessions.\n")
        console.print("  [dim]Created memory/MEMORY.md[/dim]")

    history_file = memory_dir / "HISTORY.md"
    if not history_file.exists():
        history_file.write_text("")
        console.print("  [dim]Created memory/HISTORY.md[/dim]")

    # Create skills directory for custom user skills
    skills_dir = workspace / "skills"
    skills_dir.mkdir(exist_ok=True)


def _sync_events(workspace: Path, bundled_ws: Path) -> None:
    """Copy bundled event YAML files to workspace if they don't exist."""
    src_events = bundled_ws / "events"
    if not src_events.is_dir():
        return
    dst_events = workspace / "events"
    dst_events.mkdir(exist_ok=True)
    for src_file in src_events.glob("*.yaml"):
        dst_file = dst_events / src_file.name
        if not dst_file.exists():
            dst_file.write_text(src_file.read_text(encoding="utf-8"), encoding="utf-8")
            console.print(f"  [dim]Created events/{src_file.name}[/dim]")


def _make_provider(config):
    """Create LiteLLMProvider from config. Exits if no API key found."""
    from getall.providers.litellm_provider import LiteLLMProvider
    p = config.get_provider()
    model = config.agents.defaults.model
    if not (p and p.api_key) and not model.startswith("bedrock/"):
        console.print("[red]Error: No API key configured.[/red]")
        console.print("Set one in ~/.getall/config.json under providers section")
        raise typer.Exit(1)
    return LiteLLMProvider(
        api_key=p.api_key if p else None,
        api_base=config.get_api_base(),
        default_model=model,
        extra_headers=p.extra_headers if p else None,
        provider_name=config.get_provider_name(),
    )


# ============================================================================
# Gateway / Server
# ============================================================================


@app.command()
def gateway(
    port: int = typer.Option(18790, "--port", "-p", help="Gateway port"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """Start the GetAll gateway."""
    from loguru import logger
    from getall.config.loader import load_config, get_data_dir
    from getall.bus.queue import MessageBus
    from getall.agent.loop import AgentLoop
    from getall.channels.manager import ChannelManager
    from getall.session.manager import SessionManager
    from getall.cron.service import CronService
    from getall.cron.types import CronJob
    from getall.heartbeat.service import HeartbeatService
    
    if verbose:
        import logging
        logging.basicConfig(level=logging.DEBUG)
    
    console.print(f"{__logo__} Starting GetAll gateway on port {port}...")
    
    config = load_config()

    # Ensure default workspace files exist (SOUL.md, AGENTS.md, events/, etc.)
    _create_workspace_templates(config.workspace_path)
    bus = MessageBus()
    provider = _make_provider(config)

    # Register per-model credentials so runtime model switching works.
    # Each allowed model resolves to its own provider's api_key + api_base.
    from getall.config.schema import ALLOWED_MODELS
    for model_name in ALLOWED_MODELS:
        p_cfg = config.get_provider(model_name)
        p_base = config.get_api_base(model_name)
        if p_cfg and p_cfg.api_key:
            provider.register_model_credentials(model_name, p_cfg.api_key, p_base)
            logger.info(f"Model credentials registered: {model_name}")

    session_manager = SessionManager(config.workspace_path)
    
    # Create cron service first (callback set after agent creation)
    cron_store_path = get_data_dir() / "cron" / "jobs.json"
    cron = CronService(cron_store_path)
    
    # Create agent with cron service
    agent = AgentLoop(
        bus=bus,
        provider=provider,
        workspace=config.workspace_path,
        model=config.agents.defaults.model,
        max_tokens=config.agents.defaults.max_tokens,
        max_iterations=config.agents.defaults.max_tool_iterations,
        memory_window=config.agents.defaults.memory_window,
        brave_api_key=config.tools.web.search.api_key or None,
        web_search_provider=config.tools.web.search.provider,
        web_search_api_key=config.tools.web.search.api_key or None,
        web_search_openai_api_key=(
            config.tools.web.search.openai_api_key
            or config.providers.openai.api_key
            or None
        ),
        web_search_openai_model=config.tools.web.search.openai_model,
        exec_config=config.tools.exec,
        cron_service=cron,
        restrict_to_workspace=config.tools.restrict_to_workspace,
        session_manager=session_manager,
        trading_config=config.trading,
        max_concurrent_workers=config.agents.defaults.max_concurrent_workers,
        reasoning_effort=config.agents.defaults.reasoning_effort,
    )

    def _telegram_group_mention(sender_id: str) -> str:
        """Build a Telegram mention token from stored sender_id."""
        raw = (sender_id or "").strip()
        if not raw:
            return ""
        if "|" in raw:
            uid, username = raw.split("|", 1)
            username = username.strip().lstrip("@")
            if username:
                return f"@{username}"
            raw = uid.strip()
        if raw.isdigit():
            # Markdown link; Telegram channel adapter converts markdown -> HTML.
            return f"[you](tg://user?id={raw})"
        return ""

    def _feishu_sender_open_id(sender_id: str) -> str:
        """Extract Feishu/Lark open_id from stored sender_id."""
        raw = (sender_id or "").strip()
        if not raw:
            return ""
        if "|" in raw:
            raw = raw.split("|", 1)[0].strip()
        return raw

    def _render_group_targeted_content(job: CronJob, content: str) -> str:
        """Prefix reminder output with owner mention in Telegram group chats."""
        channel = (job.payload.channel or "").strip().lower()
        if channel != "telegram":
            return content
        if (job.payload.chat_type or "").strip().lower() != "group":
            return content
        mention = _telegram_group_mention(job.payload.sender_id)
        if not mention:
            return content
        stripped = (content or "").lstrip()
        if stripped.startswith(mention):
            return content
        return f"{mention} {content}"

    def _build_group_targeted_metadata(job: CronJob) -> dict[str, str]:
        """Build channel-native group targeting metadata for outbound delivery.

        Feishu/Lark mentions are handled in channel layer to avoid breaking
        markdown heading detection (e.g. '##' must stay at column 0).
        """
        channel = (job.payload.channel or "").strip().lower()
        if channel not in {"feishu", "lark"}:
            return {}
        if (job.payload.chat_type or "").strip().lower() != "group":
            return {}
        sender_open_id = _feishu_sender_open_id(job.payload.sender_id)
        if not sender_open_id:
            return {}
        return {
            "sender_open_id": sender_open_id,
            "source": "cron",
            "cron_job_id": job.id,
        }

    def _build_cron_agent_prompt(job: CronJob, task_prompt: str) -> str:
        """Wrap cron-triggered task with anti-recursion guardrails."""
        return (
            "You are executing an existing scheduled task trigger.\n"
            f"Cron Job ID: {job.id}\n\n"
            "CRITICAL RULES:\n"
            "1) The schedule already exists. Do NOT create/update/list/delete reminders now.\n"
            "2) Execute only the task content below.\n"
            "3) Use data/trading/tools as needed and return this trigger's result.\n"
            "4) Do NOT send any acknowledgment or confirmation message (e.g. '收到', '好的', '马上处理') "
            "before executing the task. No one sent you this message — it is a scheduled trigger. "
            "Go straight to work and only send the final result.\n\n"
            f"Task:\n{task_prompt}"
        )
    
    # Set cron callback (needs agent)
    async def on_cron_job(job: CronJob) -> str | None:
        """Execute a cron job through the agent.

        Jobs with a ``principal_id`` are user-scoped and route to that user's
        own channel/chat. Jobs without a ``principal_id`` but with explicit
        ``channel`` + ``to`` are treated as legacy single-recipient jobs.
        Truly global jobs (no principal and no explicit target, e.g. cron seed
        trading tasks) are executed once per registered principal route so each
        user receives their own notification.
        """
        from getall.routing import load_all_routes

        # Direct notification mode: send payload as-is, do not invoke LLM.
        if job.payload.kind == "direct_message":
            content = job.payload.message
            if (
                job.max_runs is not None
                and job.state.run_count >= job.max_runs
                and job.payload.final_message
            ):
                content = f"{content}\n{job.payload.final_message}"
            content = _render_group_targeted_content(job, content)
            if job.payload.deliver and job.payload.to:
                from getall.bus.events import OutboundMessage
                outbound_metadata = _build_group_targeted_metadata(job)
                await bus.publish_outbound(OutboundMessage(
                    channel=job.payload.channel or "cli",
                    chat_id=job.payload.to,
                    content=content,
                    metadata=outbound_metadata,
                ))
            return content

        reminder_prompt = job.payload.message
        if (
            job.max_runs is not None
            and job.state.run_count >= job.max_runs
            and job.payload.final_message
        ):
            reminder_prompt = (
                f"{job.payload.message}\n\n"
                f"This is the final run for this reminder. "
                f"After completing the main task, also send exactly: {job.payload.final_message}"
            )

        # ── Legacy targeted job (no principal_id, but explicit destination) ──
        # Older reminder records may miss identity fields. Never fan them out
        # globally when they already carry an explicit delivery target.
        if not job.payload.principal_id and job.payload.channel and job.payload.to:
            # Use the target chat's session so cron output appears in the
            # conversation context when the user replies in the same chat.
            target_session_key = f"{job.payload.channel}:{job.payload.to}"
            response = await agent.process_direct(
                reminder_prompt,
                session_key=target_session_key,
                channel=job.payload.channel,
                chat_id=job.payload.to,
                sender_id=job.payload.sender_id or "user",
                tenant_id=job.payload.tenant_id or "default",
                thread_id=job.payload.thread_id or "",
                chat_type=job.payload.chat_type or "private",
            )
            response = _render_group_targeted_content(job, response or "")
            if job.payload.deliver and job.payload.to:
                from getall.bus.events import OutboundMessage
                outbound_metadata = _build_group_targeted_metadata(job)
                await bus.publish_outbound(OutboundMessage(
                    channel=job.payload.channel or "cli",
                    chat_id=job.payload.to,
                    content=response or "",
                    metadata=outbound_metadata,
                ))
            return response

        cron_agent_prompt = _build_cron_agent_prompt(job, reminder_prompt)

        # ── User-scoped job (has principal_id) ──
        if job.payload.principal_id:
            # Use the target chat's session so cron output appears in the
            # conversation context when the user replies in the same chat.
            cron_channel = job.payload.channel or "cli"
            cron_chat_id = job.payload.to or "direct"
            target_session_key = f"{cron_channel}:{cron_chat_id}"
            response = await agent.process_direct(
                cron_agent_prompt,
                session_key=target_session_key,
                channel=cron_channel,
                chat_id=cron_chat_id,
                sender_id=job.payload.sender_id or "user",
                tenant_id=job.payload.tenant_id or "default",
                principal_id=job.payload.principal_id,
                agent_identity_id=job.payload.agent_identity_id or "",
                thread_id=job.payload.thread_id or "",
                chat_type=job.payload.chat_type or "private",
                metadata={
                    "synthetic": True,
                    "source": "cron",
                    "cron_job_id": job.id,
                    "cron_payload_kind": job.payload.kind,
                },
            )
            response = _render_group_targeted_content(job, response or "")
            if job.payload.deliver and job.payload.to:
                from getall.bus.events import OutboundMessage
                outbound_metadata = _build_group_targeted_metadata(job)
                await bus.publish_outbound(OutboundMessage(
                    channel=job.payload.channel or "cli",
                    chat_id=job.payload.to,
                    content=response or "",
                    metadata=outbound_metadata,
                ))
            return response

        # ── Global job (no principal_id, e.g. cron seed trading tasks) ──
        # Execute once per registered principal so each user gets their own
        # isolated agent turn + notification in the correct channel/chat.
        routes = load_all_routes()
        if not routes:
            # No users registered yet — run with defaults (output to logs only)
            return await agent.process_direct(
                cron_agent_prompt,
                session_key=f"cron:{job.id}",
                metadata={
                    "synthetic": True,
                    "source": "cron",
                    "cron_job_id": job.id,
                    "cron_payload_kind": job.payload.kind,
                },
            )

        last_response: str | None = None
        for principal_id, route in routes.items():
            # Use the target chat's session so cron output appears in the
            # conversation context when the user replies in the same chat.
            target_session_key = f"{route.channel}:{route.chat_id}"
            response = await agent.process_direct(
                cron_agent_prompt,
                session_key=target_session_key,
                channel=route.channel,
                chat_id=route.chat_id,
                sender_id=f"cron:{job.id}",
                tenant_id=job.payload.tenant_id or "default",
                principal_id=principal_id,
                chat_type="private",
                metadata={
                    "synthetic": True,
                    "source": "cron",
                    "cron_job_id": job.id,
                    "cron_payload_kind": job.payload.kind,
                },
            )
            last_response = response
        return last_response
    cron.on_job = on_cron_job
    
    # Create heartbeat service
    async def on_heartbeat(prompt: str) -> str:
        """Execute heartbeat through the agent."""
        return await agent.process_direct(prompt, session_key="heartbeat")
    
    heartbeat = HeartbeatService(
        workspace=config.workspace_path,
        on_heartbeat=on_heartbeat,
        interval_s=30 * 60,  # 30 minutes
        enabled=True
    )
    
    # Create channel manager
    channels = ChannelManager(config, bus, session_manager=session_manager)
    
    if channels.enabled_channels:
        console.print(f"[green]✓[/green] Channels enabled: {', '.join(channels.enabled_channels)}")
    else:
        console.print("[yellow]Warning: No channels enabled[/yellow]")

    # Wire direct send for MessageTool so it gets delivery errors
    # (the queue-based bus.publish_outbound is fire-and-forget)
    from getall.bus.events import OutboundMessage as _OM
    from getall.agent.tools.message import MessageTool as _MT
    _msg_tool = agent.tools.get("message")
    if isinstance(_msg_tool, _MT):
        async def _direct_send(msg: _OM) -> None:
            ch = channels.get_channel(msg.channel)
            if not ch:
                raise ValueError(f"Channel '{msg.channel}' not available")
            await ch.send(msg)
        _msg_tool.set_send_callback(_direct_send)

        # Wire recipient resolver: resolve display names → platform IDs
        # using feishu member cache (search all known group caches)
        def _resolve_recipient(channel: str, name_or_id: str) -> str | None:
            ch = channels.get_channel(channel)
            if ch is None:
                return None
            from getall.channels.feishu import FeishuChannel as _FC
            if isinstance(ch, _FC):
                # Search all cached group member lists for this name
                for chat_id in list(ch._members._by_name.keys()):
                    resolved = ch._members.resolve_name(chat_id, name_or_id)
                    if resolved:
                        return resolved
            return None
        _msg_tool.set_recipient_resolver(_resolve_recipient)
    
    cron_status = cron.status()
    if cron_status["jobs"] > 0:
        console.print(f"[green]✓[/green] Cron: {cron_status['jobs']} scheduled jobs")
    
    console.print(f"[green]✓[/green] Heartbeat: every 30m")
    
    # ── Wire up Lark webhook if feishu channel is in webhook mode ──
    _feishu_webhook_mode = False
    feishu_ch = channels.get_channel("feishu")
    if feishu_ch is not None:
        from getall.channels.feishu import FeishuChannel
        if isinstance(feishu_ch, FeishuChannel) and feishu_ch.config.use_webhook:
            from getall.api.routes.lark_webhook import set_event_handler
            handler = feishu_ch.get_event_handler()
            if handler:
                set_event_handler(handler, encrypt_key=feishu_ch.config.encrypt_key)
                _feishu_webhook_mode = True
                logger.info("Lark webhook handler registered on /lark/event")

    async def _run_http_server(bind_port: int) -> None:
        """Run the FastAPI HTTP server (for Lark webhook + health)."""
        import uvicorn
        from getall.api.app import create_app
        uvi_config = uvicorn.Config(
            create_app(),
            host="0.0.0.0",
            port=bind_port,
            log_level="warning",
        )
        server = uvicorn.Server(uvi_config)
        await server.serve()

    async def run():
        # Auto-create database tables on startup (idempotent)
        try:
            from getall.storage.database import create_all_tables
            await create_all_tables()
            logger.info("Database tables ensured")
        except Exception as e:
            logger.warning(f"Database auto-migrate skipped: {e}")

        # Bootstrap admin principals from GETALL_ADMIN_IFTS env var
        admin_ifts: list[str] = getattr(config, "_admin_ifts", [])
        if admin_ifts:
            try:
                from getall.storage.database import get_session_factory
                from getall.storage.repository import IdentityRepo
                factory = get_session_factory()
                async with factory() as db_session:
                    repo = IdentityRepo(db_session)
                    for ift in admin_ifts:
                        p = await repo.get_by_ift(ift)
                        if p and p.role != "admin":
                            await repo.set_role(p.id, "admin")
                            logger.info(f"Admin bootstrapped: {p.pet_name or p.ift}")
                        elif p:
                            logger.debug(f"Already admin: {p.pet_name or p.ift}")
                        else:
                            logger.info(f"Admin IFT not found (user not yet registered): {ift}")
                    await db_session.commit()
            except Exception as e:
                logger.warning(f"Admin bootstrap failed: {e}")

        tasks = [agent.run(), channels.start_all()]

        # Start HTTP server when Lark webhook is enabled
        if _feishu_webhook_mode:
            tasks.append(_run_http_server(port))
            console.print(f"[green]✓[/green] HTTP server on :{port} (Lark webhook)")

        _shutdown_done = False

        async def _graceful_shutdown() -> None:
            """Send lifecycle notifications and tear down services."""
            nonlocal _shutdown_done
            if _shutdown_done:
                return
            _shutdown_done = True
            console.print("\nShutting down...")
            heartbeat.stop()
            cron.stop()
            agent.stop()
            await channels.stop_all()

        # Register SIGTERM handler for Docker / systemd graceful stop
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(
                sig,
                lambda: asyncio.ensure_future(_graceful_shutdown()),
            )

        try:
            await cron.start()
            await heartbeat.start()
            await asyncio.gather(*tasks)
        except (KeyboardInterrupt, asyncio.CancelledError):
            await _graceful_shutdown()
    
    asyncio.run(run())




# ============================================================================
# Agent Commands
# ============================================================================


@app.command()
def agent(
    message: str = typer.Option(None, "--message", "-m", help="Message to send to the agent"),
    session_id: str = typer.Option("cli:default", "--session", "-s", help="Session ID"),
    markdown: bool = typer.Option(True, "--markdown/--no-markdown", help="Render assistant output as Markdown"),
    logs: bool = typer.Option(False, "--logs/--no-logs", help="Show GetAll runtime logs during chat"),
):
    """Interact with the agent directly."""
    from getall.config.loader import load_config
    from getall.bus.queue import MessageBus
    from getall.agent.loop import AgentLoop
    from loguru import logger
    
    config = load_config()
    
    bus = MessageBus()
    provider = _make_provider(config)

    if logs:
        logger.enable("getall")
    else:
        logger.disable("getall")
    
    agent_loop = AgentLoop(
        bus=bus,
        provider=provider,
        workspace=config.workspace_path,
        model=config.agents.defaults.model,
        max_tokens=config.agents.defaults.max_tokens,
        max_iterations=config.agents.defaults.max_tool_iterations,
        memory_window=config.agents.defaults.memory_window,
        brave_api_key=config.tools.web.search.api_key or None,
        web_search_provider=config.tools.web.search.provider,
        web_search_api_key=config.tools.web.search.api_key or None,
        web_search_openai_api_key=(
            config.tools.web.search.openai_api_key
            or config.providers.openai.api_key
            or None
        ),
        web_search_openai_model=config.tools.web.search.openai_model,
        exec_config=config.tools.exec,
        restrict_to_workspace=config.tools.restrict_to_workspace,
        trading_config=config.trading,
        max_concurrent_workers=config.agents.defaults.max_concurrent_workers,
        reasoning_effort=config.agents.defaults.reasoning_effort,
    )
    
    # Show spinner when logs are off (no output to miss); skip when logs are on
    def _thinking_ctx():
        if logs:
            from contextlib import nullcontext
            return nullcontext()
        # Animated spinner is safe to use with prompt_toolkit input handling
        return console.status("[dim]GetAll is thinking...[/dim]", spinner="dots")

    if message:
        # Single message mode
        async def run_once():
            with _thinking_ctx():
                response = await agent_loop.process_direct(message, session_id)
            _print_agent_response(response, render_markdown=markdown)
        
        asyncio.run(run_once())
    else:
        # Interactive mode
        _init_prompt_session()
        console.print(f"{__logo__} Interactive mode (type [bold]exit[/bold] or [bold]Ctrl+C[/bold] to quit)\n")

        def _exit_on_sigint(signum, frame):
            _restore_terminal()
            console.print("\nGoodbye!")
            os._exit(0)

        signal.signal(signal.SIGINT, _exit_on_sigint)
        
        async def run_interactive():
            while True:
                try:
                    _flush_pending_tty_input()
                    user_input = await _read_interactive_input_async()
                    command = user_input.strip()
                    if not command:
                        continue

                    if _is_exit_command(command):
                        _restore_terminal()
                        console.print("\nGoodbye!")
                        break
                    
                    with _thinking_ctx():
                        response = await agent_loop.process_direct(user_input, session_id)
                    _print_agent_response(response, render_markdown=markdown)
                except KeyboardInterrupt:
                    _restore_terminal()
                    console.print("\nGoodbye!")
                    break
                except EOFError:
                    _restore_terminal()
                    console.print("\nGoodbye!")
                    break
        
        asyncio.run(run_interactive())


# ============================================================================
# Channel Commands
# ============================================================================


channels_app = typer.Typer(help="Manage channels")
app.add_typer(channels_app, name="channels")


@channels_app.command("status")
def channels_status():
    """Show channel status."""
    from getall.config.loader import load_config

    config = load_config()

    table = Table(title="Channel Status")
    table.add_column("Channel", style="cyan")
    table.add_column("Enabled", style="green")
    table.add_column("Configuration", style="yellow")

    # WhatsApp
    wa = config.channels.whatsapp
    table.add_row(
        "WhatsApp",
        "✓" if wa.enabled else "✗",
        wa.bridge_url
    )

    dc = config.channels.discord
    table.add_row(
        "Discord",
        "✓" if dc.enabled else "✗",
        dc.gateway_url
    )

    # Feishu
    fs = config.channels.feishu
    fs_config = f"app_id: {fs.app_id[:10]}..." if fs.app_id else "[dim]not configured[/dim]"
    table.add_row(
        "Feishu",
        "✓" if fs.enabled else "✗",
        fs_config
    )

    # Mochat
    mc = config.channels.mochat
    mc_base = mc.base_url or "[dim]not configured[/dim]"
    table.add_row(
        "Mochat",
        "✓" if mc.enabled else "✗",
        mc_base
    )
    
    # Telegram
    tg = config.channels.telegram
    tg_config = f"token: {tg.token[:10]}..." if tg.token else "[dim]not configured[/dim]"
    table.add_row(
        "Telegram",
        "✓" if tg.enabled else "✗",
        tg_config
    )

    # Slack
    slack = config.channels.slack
    slack_config = "socket" if slack.app_token and slack.bot_token else "[dim]not configured[/dim]"
    table.add_row(
        "Slack",
        "✓" if slack.enabled else "✗",
        slack_config
    )

    console.print(table)


def _get_bridge_dir() -> Path:
    """Get the bridge directory, setting it up if needed."""
    import shutil
    import subprocess
    
    # User's bridge location
    user_bridge = Path.home() / ".getall" / "bridge"
    
    # Check if already built
    if (user_bridge / "dist" / "index.js").exists():
        return user_bridge
    
    # Check for npm
    if not shutil.which("npm"):
        console.print("[red]npm not found. Please install Node.js >= 18.[/red]")
        raise typer.Exit(1)
    
    # Find source bridge: first check package data, then source dir
    pkg_bridge = Path(__file__).parent.parent / "bridge"  # getall/bridge (installed)
    src_bridge = Path(__file__).parent.parent.parent / "bridge"  # repo root/bridge (dev)
    
    source = None
    if (pkg_bridge / "package.json").exists():
        source = pkg_bridge
    elif (src_bridge / "package.json").exists():
        source = src_bridge
    
    if not source:
        console.print("[red]Bridge source not found.[/red]")
        console.print("Try reinstalling: pip install --force-reinstall getall-ai")
        raise typer.Exit(1)
    
    console.print(f"{__logo__} Setting up bridge...")
    
    # Copy to user directory
    user_bridge.parent.mkdir(parents=True, exist_ok=True)
    if user_bridge.exists():
        shutil.rmtree(user_bridge)
    shutil.copytree(source, user_bridge, ignore=shutil.ignore_patterns("node_modules", "dist"))
    
    # Install and build
    try:
        console.print("  Installing dependencies...")
        subprocess.run(["npm", "install"], cwd=user_bridge, check=True, capture_output=True)
        
        console.print("  Building...")
        subprocess.run(["npm", "run", "build"], cwd=user_bridge, check=True, capture_output=True)
        
        console.print("[green]✓[/green] Bridge ready\n")
    except subprocess.CalledProcessError as e:
        console.print(f"[red]Build failed: {e}[/red]")
        if e.stderr:
            console.print(f"[dim]{e.stderr.decode()[:500]}[/dim]")
        raise typer.Exit(1)
    
    return user_bridge


@channels_app.command("login")
def channels_login():
    """Link device via QR code."""
    import subprocess
    
    bridge_dir = _get_bridge_dir()
    
    console.print(f"{__logo__} Starting bridge...")
    console.print("Scan the QR code to connect.\n")
    
    try:
        subprocess.run(["npm", "start"], cwd=bridge_dir, check=True)
    except subprocess.CalledProcessError as e:
        console.print(f"[red]Bridge failed: {e}[/red]")
    except FileNotFoundError:
        console.print("[red]npm not found. Please install Node.js.[/red]")


# ============================================================================
# Cron Commands
# ============================================================================

cron_app = typer.Typer(help="Manage scheduled tasks")
app.add_typer(cron_app, name="cron")


@cron_app.command("list")
def cron_list(
    all: bool = typer.Option(False, "--all", "-a", help="Include disabled jobs"),
):
    """List scheduled jobs."""
    from getall.config.loader import get_data_dir
    from getall.cron.service import CronService
    
    store_path = get_data_dir() / "cron" / "jobs.json"
    service = CronService(store_path)
    
    jobs = service.list_jobs(include_disabled=all)
    
    if not jobs:
        console.print("No scheduled jobs.")
        return
    
    table = Table(title="Scheduled Jobs")
    table.add_column("ID", style="cyan")
    table.add_column("Name")
    table.add_column("Schedule")
    table.add_column("Status")
    table.add_column("Next Run")
    
    import time
    for job in jobs:
        # Format schedule
        if job.schedule.kind == "every":
            sched = f"every {(job.schedule.every_ms or 0) // 1000}s"
        elif job.schedule.kind == "cron":
            sched = job.schedule.expr or ""
        else:
            sched = "one-time"
        
        # Format next run
        next_run = ""
        if job.state.next_run_at_ms:
            next_time = time.strftime("%Y-%m-%d %H:%M", time.localtime(job.state.next_run_at_ms / 1000))
            next_run = next_time
        
        status = "[green]enabled[/green]" if job.enabled else "[dim]disabled[/dim]"
        
        table.add_row(job.id, job.name, sched, status, next_run)
    
    console.print(table)


@cron_app.command("add")
def cron_add(
    name: str = typer.Option(..., "--name", "-n", help="Job name"),
    message: str = typer.Option(..., "--message", "-m", help="Message for agent"),
    every: int = typer.Option(None, "--every", "-e", help="Run every N seconds"),
    cron_expr: str = typer.Option(None, "--cron", "-c", help="Cron expression (e.g. '0 9 * * *')"),
    at: str = typer.Option(None, "--at", help="Run once at time (ISO format)"),
    deliver: bool = typer.Option(False, "--deliver", "-d", help="Deliver response to channel"),
    to: str = typer.Option(None, "--to", help="Recipient for delivery"),
    channel: str = typer.Option(None, "--channel", help="Channel for delivery (e.g. 'telegram', 'whatsapp')"),
):
    """Add a scheduled job."""
    from getall.config.loader import get_data_dir
    from getall.cron.service import CronService
    from getall.cron.types import CronSchedule
    
    # Determine schedule type
    if every:
        schedule = CronSchedule(kind="every", every_ms=every * 1000)
    elif cron_expr:
        schedule = CronSchedule(kind="cron", expr=cron_expr)
    elif at:
        import datetime
        dt = datetime.datetime.fromisoformat(at)
        schedule = CronSchedule(kind="at", at_ms=int(dt.timestamp() * 1000))
    else:
        console.print("[red]Error: Must specify --every, --cron, or --at[/red]")
        raise typer.Exit(1)
    
    store_path = get_data_dir() / "cron" / "jobs.json"
    service = CronService(store_path)
    
    job = service.add_job(
        name=name,
        schedule=schedule,
        message=message,
        deliver=deliver,
        to=to,
        channel=channel,
    )
    
    console.print(f"[green]✓[/green] Added job '{job.name}' ({job.id})")


@cron_app.command("remove")
def cron_remove(
    job_id: str = typer.Argument(..., help="Job ID to remove"),
):
    """Remove a scheduled job."""
    from getall.config.loader import get_data_dir
    from getall.cron.service import CronService
    
    store_path = get_data_dir() / "cron" / "jobs.json"
    service = CronService(store_path)
    
    if service.remove_job(job_id):
        console.print(f"[green]✓[/green] Removed job {job_id}")
    else:
        console.print(f"[red]Job {job_id} not found[/red]")


@cron_app.command("enable")
def cron_enable(
    job_id: str = typer.Argument(..., help="Job ID"),
    disable: bool = typer.Option(False, "--disable", help="Disable instead of enable"),
):
    """Enable or disable a job."""
    from getall.config.loader import get_data_dir
    from getall.cron.service import CronService
    
    store_path = get_data_dir() / "cron" / "jobs.json"
    service = CronService(store_path)
    
    job = service.enable_job(job_id, enabled=not disable)
    if job:
        status = "disabled" if disable else "enabled"
        console.print(f"[green]✓[/green] Job '{job.name}' {status}")
    else:
        console.print(f"[red]Job {job_id} not found[/red]")


@cron_app.command("run")
def cron_run(
    job_id: str = typer.Argument(..., help="Job ID to run"),
    force: bool = typer.Option(False, "--force", "-f", help="Run even if disabled"),
):
    """Manually run a job."""
    from getall.config.loader import get_data_dir
    from getall.cron.service import CronService
    
    store_path = get_data_dir() / "cron" / "jobs.json"
    service = CronService(store_path)
    
    async def run():
        return await service.run_job(job_id, force=force)
    
    if asyncio.run(run()):
        console.print(f"[green]✓[/green] Job executed")
    else:
        console.print(f"[red]Failed to run job {job_id}[/red]")


# ============================================================================
# Status Commands
# ============================================================================


@app.command()
def status():
    """Show GetAll status."""
    from getall.config.loader import load_config, get_config_path

    config_path = get_config_path()
    config = load_config()
    workspace = config.workspace_path

    console.print(f"{__logo__} GetAll Status\n")

    console.print(f"Config: {config_path} {'[green]✓[/green]' if config_path.exists() else '[red]✗[/red]'}")
    console.print(f"Workspace: {workspace} {'[green]✓[/green]' if workspace.exists() else '[red]✗[/red]'}")

    if config_path.exists():
        from getall.providers.registry import PROVIDERS

        console.print(f"Model: {config.agents.defaults.model}")
        
        # Check API keys from registry
        for spec in PROVIDERS:
            p = getattr(config.providers, spec.name, None)
            if p is None:
                continue
            if spec.is_local:
                # Local deployments show api_base instead of api_key
                if p.api_base:
                    console.print(f"{spec.label}: [green]✓ {p.api_base}[/green]")
                else:
                    console.print(f"{spec.label}: [dim]not set[/dim]")
            else:
                has_key = bool(p.api_key)
                console.print(f"{spec.label}: {'[green]✓[/green]' if has_key else '[dim]not set[/dim]'}")


# ============================================================================
# Serve (FastAPI HTTP)
# ============================================================================


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", help="Bind host"),
    port: int = typer.Option(8080, "--port", "-p", help="Bind port"),
    reload: bool = typer.Option(False, "--reload", help="Auto-reload on code changes"),
):
    """Start the GetAll HTTP API server (FastAPI + Uvicorn)."""
    import uvicorn

    console.print(f"{__logo__} Starting GetAll API on {host}:{port} ...")
    uvicorn.run(
        "getall.api.app:create_app",
        host=host,
        port=port,
        reload=reload,
        factory=True,
    )


if __name__ == "__main__":
    app()
