"""Admin tool — manage users, switch models, set roles.

Only executable by principals with role='admin'. The agent loop sets
the context per-message so the tool knows who the caller is.
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable

from loguru import logger

from getall.agent.tools.base import Tool
from getall.bus.events import OutboundMessage
from getall.config.schema import ALLOWED_MODELS


@dataclass(frozen=True, slots=True)
class _AdminContext:
    principal_id: str = ""
    role: str = "user"
    session_factory: Any = None
    send_callback: Callable[[OutboundMessage], Awaitable[None]] | None = None
    chat_type: str = ""       # "private" | "group"
    sender_id: str = ""       # platform sender_id for DM routing
    channel: str = ""         # "feishu" | "telegram" etc.


class AdminTool(Tool):
    """Admin-only operations: manage users, switch model, set roles, view costs, system config."""

    name = "admin"
    description = (
        "Admin-only tool. Actions: "
        "list_users — show all registered users; "
        "set_role — promote/demote a user (principal_id + role); "
        "switch_model — switch the LLM model for private or group chats; "
        "current_model — show current active models; "
        "broadcast — send a message to ALL groups the bot has joined; "
        "cost_report — show LLM token usage and cost statistics (period: today/7d/30d/all); "
        "set_config — set a system config (key + value), e.g. welcome_message_dm; "
        "get_config — get one or all system configs (optional key); "
        "delete_config — remove a system config key."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "list_users", "set_role", "switch_model", "current_model",
                    "broadcast", "cost_report", "set_config", "get_config",
                    "delete_config",
                ],
                "description": "The admin action to perform",
            },
            "principal_id": {
                "type": "string",
                "description": "(set_role) Target user's principal ID",
            },
            "role": {
                "type": "string",
                "enum": ["user", "admin"],
                "description": "(set_role) Role to assign",
            },
            "model": {
                "type": "string",
                "description": f"(switch_model) Model to switch to. Allowed: {', '.join(ALLOWED_MODELS.keys())}",
            },
            "scope": {
                "type": "string",
                "enum": ["private", "group", "all"],
                "description": "(switch_model) Apply to private chats, group chats, or all. Default: all",
            },
            "content": {
                "type": "string",
                "description": "(broadcast) The message content to send to all groups",
            },
            "period": {
                "type": "string",
                "enum": ["today", "7d", "30d", "all"],
                "description": "(cost_report) Time range for cost report. Default: 7d",
            },
            "group_by": {
                "type": "string",
                "enum": ["date", "model", "user"],
                "description": "(cost_report) Group results by date, model, or user. Default: date",
            },
            "key": {
                "type": "string",
                "description": (
                    "(set_config / get_config / delete_config) Config key name. "
                    "Common keys: welcome_message_dm, group_reply_policy, etc."
                ),
            },
            "value": {
                "type": "string",
                "description": "(set_config) The value to set for the config key.",
            },
        },
        "required": ["action"],
    }

    def __init__(self) -> None:
        self._context: ContextVar[_AdminContext] = ContextVar(
            "admin_tool_context", default=_AdminContext(),
        )

    def set_context(
        self,
        principal_id: str,
        role: str,
        session_factory: Any,
        send_callback: Callable[[OutboundMessage], Awaitable[None]] | None = None,
        chat_type: str = "",
        sender_id: str = "",
        channel: str = "",
    ) -> None:
        self._context.set(_AdminContext(
            principal_id=principal_id,
            role=role,
            session_factory=session_factory,
            send_callback=send_callback,
            chat_type=chat_type,
            sender_id=sender_id,
            channel=channel,
        ))

    def clear_context(self) -> None:
        self._context.set(_AdminContext())

    # Actions whose output contains sensitive data and must NOT be shown in groups.
    # When called from a group chat, result is DM'd to the requesting admin instead.
    _SENSITIVE_ACTIONS: frozenset[str] = frozenset({
        "cost_report", "list_users", "set_role", "get_config",
    })

    def _check_admin(self) -> _AdminContext:
        ctx = self._context.get()
        if ctx.role != "admin":
            raise PermissionError("Admin access required")
        return ctx

    async def _dm_admin(self, ctx: _AdminContext, content: str) -> bool:
        """Send *content* to the requesting admin via private message.

        Returns True if successfully sent, False otherwise.
        """
        if not ctx.send_callback:
            logger.warning("Admin DM skipped: no send_callback")
            return False

        from getall.routing import load_last_route

        route = load_last_route(principal_id=ctx.principal_id, prefer_private=True)
        if not route:
            # Fallback: send to current sender_id directly (works for feishu open_id)
            if ctx.sender_id and ctx.channel:
                try:
                    await ctx.send_callback(OutboundMessage(
                        channel=ctx.channel,
                        chat_id=ctx.sender_id,
                        content=content,
                    ))
                    return True
                except Exception as exc:
                    logger.warning(f"Admin DM fallback failed: {exc}")
            return False

        try:
            await ctx.send_callback(OutboundMessage(
                channel=route.channel,
                chat_id=route.chat_id,
                content=content,
            ))
            return True
        except Exception as exc:
            logger.warning(f"Admin DM failed: {exc}")
            return False

    async def execute(self, action: str = "", **kwargs: Any) -> str:
        try:
            ctx = self._check_admin()
        except PermissionError:
            return "Error: 你没有管理员权限。"

        # ── Sensitive-action group guard ──
        # If a sensitive action is invoked from a group chat, execute it
        # but deliver the result via DM instead of leaking to the group.
        is_group = ctx.chat_type == "group"
        redirect_to_dm = is_group and action in self._SENSITIVE_ACTIONS

        result = await self._dispatch(ctx, action, kwargs)

        if redirect_to_dm:
            dm_ok = await self._dm_admin(ctx, f"🔐 {result}")
            if dm_ok:
                return (
                    "[SENSITIVE_REDIRECTED] 敏感数据已通过私聊发送给管理员。"
                    "请在群里用你自己的风格简短回复，告知已私聊，不要透露任何具体数据。"
                )
            else:
                return (
                    "[SENSITIVE_BLOCKED] 该操作包含敏感数据，无法在群聊展示，且私聊发送失败。"
                    "请告知管理员私聊你查看。"
                )

        return result

    async def _dispatch(self, ctx: _AdminContext, action: str, kwargs: dict[str, Any]) -> str:
        """Route to the correct action handler."""
        if action == "list_users":
            return await self._list_users(ctx)
        elif action == "set_role":
            return await self._set_role(ctx, kwargs.get("principal_id", ""), kwargs.get("role", ""))
        elif action == "switch_model":
            return await self._switch_model(
                ctx,
                kwargs.get("model", ""),
                kwargs.get("scope", "all"),
            )
        elif action == "current_model":
            return await self._current_model(ctx)
        elif action == "broadcast":
            return await self._broadcast(ctx, kwargs.get("content", ""))
        elif action == "cost_report":
            return await self._cost_report(
                ctx,
                kwargs.get("period", "7d"),
                kwargs.get("group_by", "date"),
            )
        elif action == "set_config":
            return await self._set_config(
                ctx, kwargs.get("key", ""), kwargs.get("value", ""),
            )
        elif action == "get_config":
            return await self._get_config(ctx, kwargs.get("key", ""))
        elif action == "delete_config":
            return await self._delete_config(ctx, kwargs.get("key", ""))
        else:
            return f"Error: unknown action '{action}'"

    async def _list_users(self, ctx: _AdminContext) -> str:
        from getall.storage.repository import IdentityRepo

        async with ctx.session_factory() as session:
            repo = IdentityRepo(session)
            principals = await repo.list_all(limit=200)

        if not principals:
            return "No registered users."

        lines = [f"共 {len(principals)} 个用户:\n"]
        for p in principals:
            name = p.pet_name or "(未命名)"
            status = "✅" if p.onboarded else "⏳"
            role_tag = " [ADMIN]" if p.role == "admin" else ""
            lines.append(
                f"- {status} {name}{role_tag} | ID: {p.id[:8]}… | "
                f"IFT: {p.ift} | 注册: {p.created_at.strftime('%m-%d %H:%M')}"
            )
        return "\n".join(lines)

    async def _set_role(self, ctx: _AdminContext, principal_id: str, role: str) -> str:
        if not principal_id or not role:
            return "Error: need principal_id and role"
        if role not in ("user", "admin"):
            return f"Error: role must be 'user' or 'admin', got '{role}'"

        from getall.storage.repository import IdentityRepo

        async with ctx.session_factory() as session:
            repo = IdentityRepo(session)
            p = await repo.set_role(principal_id, role)
            await session.commit()

        if p is None:
            return f"Error: principal {principal_id} not found"
        name = p.pet_name or p.ift
        return f"已将 {name} 的角色设置为 {role}"

    async def _switch_model(self, ctx: _AdminContext, model: str, scope: str) -> str:
        if not model:
            models_list = "\n".join(f"- {k} ({v})" for k, v in ALLOWED_MODELS.items())
            return f"请指定要切换的模型。可选:\n{models_list}"

        if model not in ALLOWED_MODELS:
            models_list = "\n".join(f"- {k} ({v})" for k, v in ALLOWED_MODELS.items())
            return f"Error: 不支持的模型 '{model}'。可选:\n{models_list}"

        from getall.storage.repository import SystemConfigRepo

        scopes = []
        if scope in ("private", "all"):
            scopes.append("model:private")
        if scope in ("group", "all"):
            scopes.append("model:group")

        async with ctx.session_factory() as session:
            repo = SystemConfigRepo(session)
            for key in scopes:
                await repo.set(key, model, updated_by=ctx.principal_id)
            await session.commit()

        display = ALLOWED_MODELS[model]
        scope_label = {"private": "私聊", "group": "群聊", "all": "所有会话"}[scope]

        # Broadcast model-switch card to all affected chats
        await self._broadcast_model_switch(ctx, model, display, scope)

        return f"已将 {scope_label} 模型切换为 {display} ({model})"

    async def _current_model(self, ctx: _AdminContext) -> str:
        from getall.storage.repository import SystemConfigRepo

        async with ctx.session_factory() as session:
            repo = SystemConfigRepo(session)
            values = await repo.get_multi(["model:private", "model:group"])

        private_model = values.get("model:private", "(默认)")
        group_model = values.get("model:group", "(默认)")

        private_display = ALLOWED_MODELS.get(private_model, private_model)
        group_display = ALLOWED_MODELS.get(group_model, group_model)

        return (
            f"当前模型配置:\n"
            f"- 私聊: {private_display} ({private_model})\n"
            f"- 群聊: {group_display} ({group_model})"
        )

    async def _broadcast(self, ctx: _AdminContext, content: str) -> str:
        """Send a message to all groups the bot has joined."""
        if not content:
            return "Error: content is required for broadcast"
        if not ctx.send_callback:
            return "Error: send_callback not available"

        from getall.routing import load_all_group_routes

        targets = load_all_group_routes()
        if not targets:
            return "没有找到任何群聊记录。bot 可能还没有在任何群里收到过消息。"

        sent = 0
        failed = 0
        stale_cleaned = 0
        for channel, chat_id in targets:
            try:
                await ctx.send_callback(OutboundMessage(
                    channel=channel,
                    chat_id=chat_id,
                    content=content,
                ))
                sent += 1
                logger.info(f"Broadcast sent to {channel}:{chat_id}")
            except Exception as exc:
                failed += 1
                exc_str = str(exc)
                if any(code in exc_str for code in ("230002", "403", "chat not found")):
                    from getall.routing import remove_stale_route
                    n = remove_stale_route(channel, chat_id)
                    stale_cleaned += n
                    logger.info(f"Cleaned stale route {channel}:{chat_id} ({n} entries)")
                else:
                    logger.warning(f"Broadcast failed for {channel}:{chat_id}: {exc}")

        result = f"广播完成：成功发送到 {sent} 个群"
        if failed:
            result += f"，{failed} 个群发送失败"
        if stale_cleaned:
            result += f"（已清理 {stale_cleaned} 条过期路由）"
        return result

    async def _cost_report(self, ctx: _AdminContext, period: str, group_by: str) -> str:
        """Generate LLM usage & cost report."""
        from datetime import UTC, datetime, timedelta
        from getall.storage.repository import LLMUsageRepo

        now = datetime.now(tz=UTC)
        if period == "today":
            start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            label = "今天"
        elif period == "30d":
            start = now - timedelta(days=30)
            label = "近 30 天"
        elif period == "all":
            start = datetime(2020, 1, 1, tzinfo=UTC)
            label = "全部"
        else:  # default 7d
            start = now - timedelta(days=7)
            label = "近 7 天"

        async with ctx.session_factory() as session:
            repo = LLMUsageRepo(session)

            # Always include the total summary line
            total = await repo.total_cost(start, now)

            if group_by == "model":
                rows = await repo.summary_by_model(start, now)
            elif group_by == "user":
                rows = await repo.summary_by_principal(start, now)
            else:
                rows = await repo.summary_by_date(start, now)

        # ── Format output ──
        lines: list[str] = [f"📊 LLM 成本报告（{label}）\n"]

        total_calls = total["total_calls"]
        if total_calls == 0:
            lines.append("暂无 LLM 调用记录。")
            return "\n".join(lines)

        lines.append(
            f"总计: {total_calls} 次调用 | "
            f"{total['prompt_tokens']:,} prompt + {total['completion_tokens']:,} completion tokens | "
            f"${total['cost_usd']} USD\n"
        )

        if group_by == "model":
            lines.append("按模型:")
            for r in rows:
                lines.append(
                    f"  • {r['model']} — {r['total_calls']} 次 | "
                    f"{r['prompt_tokens']:,}+{r['completion_tokens']:,} tokens | "
                    f"${r['cost_usd']}"
                )
        elif group_by == "user":
            # Resolve principal names
            principal_names = await self._resolve_principal_names(
                ctx, [str(r["principal_id"]) for r in rows]
            )
            lines.append("按用户:")
            for r in rows:
                pid = str(r["principal_id"])
                name = principal_names.get(pid, pid[:8] + "…")
                lines.append(
                    f"  • {name} — {r['total_calls']} 次 | "
                    f"{r['total_tokens']:,} tokens | ${r['cost_usd']}"
                )
        else:
            lines.append("按日期:")
            for r in rows:
                lines.append(
                    f"  • {r['date']} — {r['total_calls']} 次 | "
                    f"{r['prompt_tokens']:,}+{r['completion_tokens']:,} tokens | "
                    f"${r['cost_usd']}"
                )

        return "\n".join(lines)

    async def _resolve_principal_names(
        self, ctx: _AdminContext, principal_ids: list[str],
    ) -> dict[str, str]:
        """Map principal IDs to display names (pet_name or IFT)."""
        if not principal_ids:
            return {}
        from getall.storage.repository import IdentityRepo
        result: dict[str, str] = {}
        async with ctx.session_factory() as session:
            repo = IdentityRepo(session)
            for pid in principal_ids:
                if not pid:
                    continue
                p = await repo.get_by_id(pid)
                if p:
                    result[pid] = p.pet_name or p.ift or pid[:8] + "…"
                else:
                    result[pid] = pid[:8] + "…"
        return result

    # ── System Config actions ──

    async def _set_config(self, ctx: _AdminContext, key: str, value: str) -> str:
        """Set a system configuration value."""
        if not key:
            return "Error: key is required"
        if value is None:
            return "Error: value is required"

        from getall.storage.repository import SystemConfigRepo

        async with ctx.session_factory() as session:
            repo = SystemConfigRepo(session)
            await repo.set(key, value, updated_by=ctx.principal_id)
            await session.commit()

        logger.info(f"Config set: {key} = {value[:100]}... (by {ctx.principal_id[:8]})")
        return f"配置已保存: {key} = {value[:200]}"

    async def _get_config(self, ctx: _AdminContext, key: str) -> str:
        """Get one or all system config values."""
        from getall.storage.repository import SystemConfigRepo

        async with ctx.session_factory() as session:
            repo = SystemConfigRepo(session)

            if key:
                val = await repo.get(key)
                if val is None:
                    return f"配置 '{key}' 不存在"
                return f"{key} = {val}"
            else:
                configs = await repo.list_all()
                if not configs:
                    return "当前没有系统配置"
                lines = ["当前系统配置:\n"]
                for c in configs:
                    val_preview = c.value[:100] + ("..." if len(c.value) > 100 else "")
                    lines.append(f"- {c.key} = {val_preview}")
                return "\n".join(lines)

    async def _delete_config(self, ctx: _AdminContext, key: str) -> str:
        """Delete a system configuration key."""
        if not key:
            return "Error: key is required"

        from getall.storage.repository import SystemConfigRepo

        async with ctx.session_factory() as session:
            repo = SystemConfigRepo(session)
            deleted = await repo.delete(key)
            await session.commit()

        if deleted:
            logger.info(f"Config deleted: {key} (by {ctx.principal_id[:8]})")
            return f"配置已删除: {key}"
        return f"配置 '{key}' 不存在"

    async def _notify_admins(self, ctx: _AdminContext, content: str) -> None:
        """Send a private message to all admin principals."""
        if not ctx.send_callback:
            logger.warning("Admin notification skipped: no send_callback")
            return

        try:
            from getall.storage.repository import IdentityRepo
            from getall.routing import load_last_route

            async with ctx.session_factory() as session:
                repo = IdentityRepo(session)
                admins = await repo.get_admins()

            for admin in admins:
                route = load_last_route(principal_id=admin.id, prefer_private=True)
                if not route:
                    continue
                try:
                    await ctx.send_callback(OutboundMessage(
                        channel=route.channel,
                        chat_id=route.chat_id,
                        content=content,
                    ))
                    logger.info(f"Admin notification sent to {admin.pet_name or admin.id[:8]}")
                except Exception as exc:
                    logger.warning(f"Failed to notify admin {admin.id[:8]}: {exc}")
        except Exception as exc:
            logger.warning(f"Admin notification failed: {exc}")

    async def _broadcast_model_switch(
        self,
        ctx: _AdminContext,
        model: str,
        display: str,
        scope: str,
    ) -> None:
        """Send a model-switch card to all affected chats (groups / private / both)."""
        if not ctx.send_callback:
            logger.warning("Model switch broadcast skipped: no send_callback")
            return

        import datetime as _dt
        from getall.routing import load_all_group_routes, load_all_private_routes

        scope_label = {"private": "私聊", "group": "群聊", "all": "所有会话"}[scope]
        now = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        admin_name = await self._resolve_admin_name(ctx)

        content = (
            f"# 🔄 模型已切换\n\n"
            f"**新模型：** {display}\n"
            f"**范围：** {scope_label}\n"
            f"**时间：** {now}\n"
            f"**操作者：** {admin_name}"
        )

        # Collect targets based on scope
        targets: list[tuple[str, str]] = []
        if scope in ("group", "all"):
            targets.extend(load_all_group_routes())
        if scope in ("private", "all"):
            targets.extend(load_all_private_routes())

        # Deduplicate
        seen: set[tuple[str, str]] = set()
        unique: list[tuple[str, str]] = []
        for t in targets:
            if t not in seen:
                seen.add(t)
                unique.append(t)

        sent = 0
        failed = 0
        stale_cleaned = 0
        for channel, chat_id in unique:
            try:
                await ctx.send_callback(OutboundMessage(
                    channel=channel,
                    chat_id=chat_id,
                    content=content,
                    metadata={"event_card_theme": "turquoise"},
                ))
                sent += 1
            except Exception as exc:
                failed += 1
                exc_str = str(exc)
                # Permanent delivery failures → remove stale route
                # Feishu 230002: bot/user not in chat
                # Telegram 403: bot was blocked by the user
                if any(code in exc_str for code in ("230002", "403", "chat not found")):
                    from getall.routing import remove_stale_route
                    n = remove_stale_route(channel, chat_id)
                    stale_cleaned += n
                    logger.info(f"Cleaned stale route {channel}:{chat_id} ({n} entries)")
                else:
                    logger.warning(f"Model switch card failed for {channel}:{chat_id}: {exc}")

        parts = [f"{sent} sent"]
        if failed:
            parts.append(f"{failed} failed")
        if stale_cleaned:
            parts.append(f"{stale_cleaned} stale routes cleaned")
        logger.info(f"Model switch broadcast: {', '.join(parts)}")

    async def _resolve_admin_name(self, ctx: _AdminContext) -> str:
        """Resolve the display name for the current admin principal."""
        try:
            from getall.storage.repository import IdentityRepo

            async with ctx.session_factory() as session:
                repo = IdentityRepo(session)
                p = await repo.get_by_id(ctx.principal_id)
                if p:
                    return p.pet_name or p.ift or ctx.principal_id[:8] + "…"
        except Exception:
            pass
        return ctx.principal_id[:8] + "…"
