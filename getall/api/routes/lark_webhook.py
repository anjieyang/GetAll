"""Lark/Feishu event subscription webhook endpoint.

Receives HTTP POST events from Lark Open Platform and delegates to
the FeishuChannel's EventDispatcherHandler.
"""

from __future__ import annotations

from fastapi import APIRouter, Request, Response
from loguru import logger

router = APIRouter()

# Will be set by the gateway when starting in webhook mode
_event_handler = None


def set_event_handler(handler: object) -> None:
    """Register the lark-oapi EventDispatcherHandler for webhook processing."""
    global _event_handler
    _event_handler = handler


@router.post("/lark/event")
async def lark_event(request: Request) -> Response:
    """Handle Lark event subscription callbacks.

    Supports:
    - URL verification challenge (returns challenge token)
    - im.message.receive_v1 events (dispatched to FeishuChannel)
    """
    if _event_handler is None:
        logger.warning("Lark webhook received but no event handler registered")
        return Response(
            content='{"error":"handler not configured"}',
            status_code=503,
            media_type="application/json",
        )

    try:
        import lark_oapi as lark

        body = await request.body()
        headers: dict[str, str] = {}
        for key, value in request.headers.items():
            headers[key] = value

        raw_req = lark.RawRequest()
        raw_req.uri = str(request.url.path)
        raw_req.headers = headers
        raw_req.body = body

        raw_resp: lark.RawResponse = _event_handler.do(raw_req)

        return Response(
            content=raw_resp.content,
            status_code=raw_resp.status_code,
            media_type="application/json; charset=utf-8",
        )
    except Exception as exc:
        logger.error(f"Lark webhook error: {exc}")
        return Response(
            content=f'{{"error":"{exc}"}}',
            status_code=500,
            media_type="application/json",
        )
