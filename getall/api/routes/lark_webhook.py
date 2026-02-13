"""Lark/Feishu event subscription webhook endpoint.

Handles encrypted v2 events by decrypting with Encrypt Key and
dispatching via EventDispatcherHandler.do_without_validation().
This avoids signature header issues (some Lark events don't include
X-Lark-Request-Timestamp / X-Lark-Signature headers).
"""

from __future__ import annotations

import json

from fastapi import APIRouter, Request, Response
from loguru import logger

router = APIRouter()

# Set by gateway at startup
_event_handler = None
_encrypt_key: str = ""


def set_event_handler(handler: object, encrypt_key: str = "") -> None:
    """Register the lark-oapi EventDispatcherHandler + encrypt key."""
    global _event_handler, _encrypt_key
    _event_handler = handler
    _encrypt_key = encrypt_key


@router.post("/lark/event")
async def lark_event(request: Request) -> Response:
    """Handle Lark event subscription callbacks (v2 encrypted events)."""
    body = await request.body()
    logger.info(f"Lark webhook: {len(body)} bytes")

    if _event_handler is None:
        return _json(503, {"error": "handler not configured"})

    try:
        # ── 1. Parse body ──
        raw = json.loads(body)

        # ── 2. Decrypt if encrypted ──
        plaintext: str
        if "encrypt" in raw and _encrypt_key:
            from lark_oapi import AESCipher
            plaintext = AESCipher(_encrypt_key).decrypt_str(raw["encrypt"])
            logger.debug(f"Decrypted event: {plaintext[:300]}")
        else:
            plaintext = body.decode("utf-8")

        # ── 3. URL verification challenge ──
        event_data = json.loads(plaintext)
        if event_data.get("type") == "url_verification":
            challenge = event_data.get("challenge", "")
            logger.info(f"URL verification challenge: {challenge[:20]}...")
            return _json(200, {"challenge": challenge})

        # ── 4. Dispatch via SDK (skip signature verification) ──
        _event_handler.do_without_validation(plaintext.encode("utf-8"))
        return _json(200, {"msg": "success"})

    except Exception as exc:
        logger.error(f"Lark webhook error: {exc}", exc_info=True)
        # Always return 200 to Lark so it doesn't keep retrying
        return _json(200, {"msg": "ok"})


def _json(status: int, data: dict) -> Response:
    return Response(
        content=json.dumps(data),
        status_code=status,
        media_type="application/json; charset=utf-8",
    )
