"""Credential tool — lets the LLM save / check / delete user exchange API keys.

Security: HARD-CODED to reject any call that is NOT in a private chat.
The tool result NEVER contains the raw key material.
"""

from __future__ import annotations

from typing import Any

from getall.agent.tools.base import Tool


class CredentialTool(Tool):
    """Manage user exchange API credentials (private chat only)."""

    name = "credential"
    description = (
        "Save, check, or delete the user's exchange API credentials. "
        "ONLY works in private/DM chats — will refuse in group chats. "
        "Actions: save (store new credentials), check (see if credentials exist), delete (remove credentials)."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["save", "check", "delete"],
                "description": "What to do with credentials",
            },
            "provider": {
                "type": "string",
                "description": "Exchange name, default 'bitget'",
            },
            "api_key": {
                "type": "string",
                "description": "API key (only for save action)",
            },
            "api_secret": {
                "type": "string",
                "description": "API secret (only for save action)",
            },
            "passphrase": {
                "type": "string",
                "description": "API passphrase (only for save action)",
            },
        },
        "required": ["action"],
    }

    def __init__(self) -> None:
        self._principal_id: str = ""
        self._session_factory: Any = None
        self._chat_type: str = "private"

    def set_context(
        self,
        principal_id: str,
        session_factory: Any,
        chat_type: str = "private",
    ) -> None:
        """Called by agent loop before each message."""
        self._principal_id = principal_id
        self._session_factory = session_factory
        self._chat_type = chat_type

    async def execute(
        self,
        action: str = "check",
        provider: str = "bitget",
        api_key: str = "",
        api_secret: str = "",
        passphrase: str = "",
        **kw: Any,
    ) -> str:
        # ── HARD SECURITY GATE ──
        if self._chat_type != "private":
            return (
                "Error: credential operations are only allowed in private chats. "
                "Please DM me directly to bind your exchange account."
            )

        if not self._principal_id:
            return "Error: no identity resolved — cannot manage credentials"

        if self._session_factory is None:
            return "Error: database session not available"

        from getall.storage.repository import CredentialRepo

        async with self._session_factory() as session:
            repo = CredentialRepo(session)

            if action == "save":
                if not api_key or not api_secret or not passphrase:
                    return "Error: api_key, api_secret, and passphrase are all required for save"
                await repo.save(
                    principal_id=self._principal_id,
                    provider=provider,
                    api_key=api_key,
                    api_secret=api_secret,
                    passphrase=passphrase,
                )
                await session.commit()
                return f"Credentials for {provider} saved successfully. You can now use trading features."

            elif action == "check":
                exists = await repo.exists(self._principal_id, provider)
                if exists:
                    return f"Credentials for {provider} are configured. Trading features are available."
                return f"No credentials found for {provider}. Ask the user to provide API key, secret, and passphrase."

            elif action == "delete":
                deleted = await repo.delete(self._principal_id, provider)
                await session.commit()
                if deleted:
                    return f"Credentials for {provider} deleted."
                return f"No credentials found for {provider} to delete."

            else:
                return f"Error: unknown action '{action}'"
