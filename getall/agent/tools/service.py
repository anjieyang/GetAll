"""Service tool for managing background WebSocket services."""

from typing import Any

from getall.agent.tools.base import Tool


class ServiceTool(Tool):
    """Let the agent query and manage background WebSocket services.

    Wraps a ``WSManager`` to expose:
    - ``status``  -- list all background services and their states
    - ``enable``  -- start a stopped service
    - ``disable`` -- stop a running service

    Cron jobs are managed separately via the ``reminders`` / ``cron`` tool.
    """

    def __init__(self, ws_manager: Any) -> None:
        self._ws = ws_manager

    @property
    def name(self) -> str:
        return "service"

    @property
    def description(self) -> str:
        return (
            "Manage background WebSocket services. "
            "Available services: ws:account (exchange position/order/balance stream), "
            "ws:breaking-news (real-time news). "
            "Actions: status (list all), enable (start), disable (stop). "
            "Cron tasks are managed via the separate 'reminders' tool."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["status", "enable", "disable"],
                    "description": "Action to perform.",
                },
                "name": {
                    "type": "string",
                    "description": (
                        "Service name to enable/disable, e.g. 'ws:account', "
                        "'ws:breaking-news'. Not required for 'status'."
                    ),
                },
            },
            "required": ["action"],
        }

    async def execute(self, action: str, name: str | None = None, **kwargs: Any) -> str:
        if action == "status":
            return self._status()
        if action == "enable":
            return await self._enable(name)
        if action == "disable":
            return await self._disable(name)
        return f"Unknown action: {action}"

    # ------------------------------------------------------------------

    def _status(self) -> str:
        statuses = self._ws.get_service_status()
        if not statuses:
            return "No WebSocket services registered."
        lines = ["Background WebSocket services:"]
        for svc_name, status in sorted(statuses.items()):
            icon = _status_icon(status)
            lines.append(f"  {icon} {svc_name}: {status}")
        return "\n".join(lines)

    async def _enable(self, name: str | None) -> str:
        if not name:
            return "Error: 'name' is required for enable (e.g. 'ws:account')."
        return await self._ws.enable_service(name)

    async def _disable(self, name: str | None) -> str:
        if not name:
            return "Error: 'name' is required for disable (e.g. 'ws:account')."
        return await self._ws.disable_service(name)


def _status_icon(status: str) -> str:
    return {"running": "[ON]", "stopped": "[OFF]", "connecting": "[..]", "error": "[ERR]"}.get(
        status, "[?]"
    )
