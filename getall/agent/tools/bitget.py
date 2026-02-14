"""Bitget tools exposed to the LLM.

Routing policy:
- `bitget_market`: public market endpoints
- `bitget_account` / `bitget_trade`: common private shortcuts
- `bitget_uta`: generic private/public endpoint caller for full API coverage
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
import json
from typing import Any

from getall.agent.tools.base import Tool
from getall.integrations.bitget.uta_catalog import (
    build_uta_coverage_report,
    describe_endpoint,
    list_uta_catalog,
)
from getall.integrations.bitget.uta_gateway import (
    BitgetCredentials,
    BitgetError,
    BitgetGateway,
)

_LEGACY_FUTURES_SIDE: dict[str, tuple[str, str]] = {
    "open_long": ("buy", "open"),
    "open_short": ("sell", "open"),
    "close_long": ("buy", "close"),
    "close_short": ("sell", "close"),
}

_NO_CREDENTIALS_MSG = (
    "No exchange credentials bound for this user. "
    "Please bind your Bitget API key first by sending your "
    "API Key, API Secret, and Passphrase in a private chat."
)

_market_gw: BitgetGateway | None = None


def _market_gateway() -> BitgetGateway:
    global _market_gw
    if _market_gw is None:
        _market_gw = BitgetGateway()
    return _market_gw


@dataclass(frozen=True, slots=True)
class _PrivateBitgetContext:
    principal_id: str = ""
    session_factory: Any = None


async def _resolve_user_gateway(
    principal_id: str,
    session_factory: Any,
) -> BitgetGateway | None:
    if not principal_id:
        return None
    if session_factory is None:
        raise BitgetError("database session not available for credential lookup")
    try:
        from getall.storage.repository import CredentialRepo

        async with session_factory() as session:
            repo = CredentialRepo(session)
            creds = await repo.get(principal_id, "bitget")
    except Exception as exc:
        raise BitgetError(f"failed to load bitget credentials for principal '{principal_id}': {exc}") from exc

    if creds is None:
        return None
    return BitgetGateway(
        credentials=BitgetCredentials(
            api_key=creds["api_key"],
            api_secret=creds["api_secret"],
            passphrase=creds["passphrase"],
        )
    )


def _json_result(data: Any, limit: int) -> str:
    return json.dumps(data, ensure_ascii=False)[:limit]


def _derive_futures_side(action: str, side: str) -> str:
    normalized = (side or "").strip().lower()
    if normalized:
        return normalized
    return "open_long" if action == "futures_open" else "close_long"


def _build_futures_order_args(
    *,
    action: str,
    symbol: str,
    size: str,
    price: str,
    side: str,
    trade_side: str,
    order_type: str,
    product_type: str,
    margin_mode: str,
) -> dict[str, Any]:
    args: dict[str, Any] = {
        "symbol": symbol,
        "productType": product_type,
        "marginMode": margin_mode,
        "orderType": order_type,
        "size": size,
    }
    normalized_side = _derive_futures_side(action, side)
    normalized_trade_side = (trade_side or "").strip().lower()
    buy_sell = normalized_side
    resolved_trade_side = normalized_trade_side or ("open" if action == "futures_open" else "close")

    if normalized_side in _LEGACY_FUTURES_SIDE:
        mapped_side, mapped_trade_side = _LEGACY_FUTURES_SIDE[normalized_side]
        buy_sell = mapped_side
        if not normalized_trade_side:
            resolved_trade_side = mapped_trade_side

    args["side"] = buy_sell
    args["tradeSide"] = resolved_trade_side
    if (order_type or "").strip().lower() == "limit":
        args["price"] = price
    return args


class BitgetMarketTool(Tool):
    """Public market data shortcuts."""

    name = "bitget_market"
    description = (
        "Fetch public market data from Bitget. "
        "Actions: tickers, candles, depth, funding_rate."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["tickers", "candles", "depth", "funding_rate"],
            },
            "symbol": {"type": "string"},
            "product_type": {"type": "string"},
            "granularity": {"type": "string"},
            "limit": {"type": "integer", "minimum": 1, "maximum": 200},
        },
        "required": ["action"],
    }

    async def execute(
        self,
        action: str,
        symbol: str = "",
        product_type: str = "USDT-FUTURES",
        granularity: str = "1H",
        limit: int = 20,
        **kw: Any,
    ) -> str:
        gw = _market_gateway()
        try:
            if action == "tickers":
                data = await gw.get_tickers(product_type=product_type)
            elif action == "candles":
                if not symbol:
                    return "Error: symbol is required for candles"
                data = await gw.get_candles(
                    symbol=symbol,
                    granularity=granularity,
                    product_type=product_type,
                    limit=limit,
                )
            elif action == "depth":
                if not symbol:
                    return "Error: symbol is required for depth"
                data = await gw.get_depth(symbol=symbol, product_type=product_type, limit=limit)
            elif action == "funding_rate":
                if not symbol:
                    return "Error: symbol is required for funding_rate"
                data = await gw.get_funding_rate(symbol=symbol, product_type=product_type)
            else:
                return f"Error: unknown action '{action}'"
            return _json_result(data, limit=4000)
        except Exception as exc:
            return f"Error: bitget_market action '{action}' failed: {exc}"


class BitgetAccountTool(Tool):
    """Common account shortcuts on top of direct UTA REST."""

    name = "bitget_account"
    description = (
        "Shortcut wrapper for common account queries over direct Bitget UTA REST. "
        "Actions: all_assets, spot_assets, futures_assets, positions, bills. "
        "For any endpoint beyond shortcuts, use bitget_uta."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["all_assets", "spot_assets", "futures_assets", "positions", "bills", "assets"],
                "default": "all_assets",
            },
            "product_type": {"type": "string"},
            "symbol": {"type": "string"},
            "coin": {"type": "string"},
        },
        "required": [],
    }

    def __init__(self) -> None:
        self._context: ContextVar[_PrivateBitgetContext] = ContextVar(
            "bitget_account_tool_context",
            default=_PrivateBitgetContext(),
        )

    def set_context(self, principal_id: str, session_factory: Any) -> None:
        self._context.set(_PrivateBitgetContext(principal_id=principal_id, session_factory=session_factory))

    async def execute(
        self,
        action: str = "all_assets",
        product_type: str = "USDT-FUTURES",
        symbol: str = "",
        coin: str = "",
        **kw: Any,
    ) -> str:
        context = self._context.get()
        try:
            gw = await _resolve_user_gateway(context.principal_id, context.session_factory)
        except BitgetError as exc:
            return f"Error: {exc}"
        if gw is None:
            return f"Error: {_NO_CREDENTIALS_MSG}"

        action_key = (action or "all_assets").strip() or "all_assets"
        if action_key == "assets":
            action_key = "all_assets"

        try:
            if action_key == "all_assets":
                data = await gw.get_account_assets(account_type="all", coin=coin)
            elif action_key == "spot_assets":
                data = await gw.get_account_assets(account_type="spot", coin=coin)
            elif action_key == "futures_assets":
                data = await gw.get_account_assets(
                    account_type="futures",
                    coin=coin,
                    product_type=product_type,
                )
            elif action_key == "positions":
                data = await gw.get_positions(product_type=product_type, symbol=symbol)
            elif action_key == "bills":
                data = await gw.get_account_bills(
                    account_type="futures",
                    product_type=product_type,
                )
            else:
                return (
                    f"Error: unknown action '{action_key}'. "
                    "Use bitget_uta for full UTA endpoint coverage."
                )
            return _json_result(data, limit=8000)
        except BitgetError as exc:
            return f"Error: {exc}"
        except Exception as exc:
            return f"Error: bitget_account action '{action_key}' failed: {exc}"
        finally:
            await gw.close()


class BitgetTradeTool(Tool):
    """Common trade shortcuts on top of direct UTA REST."""

    name = "bitget_trade"
    description = (
        "Shortcut wrapper for common trading operations over direct Bitget UTA REST. "
        "Actions: spot_buy, spot_sell, futures_open, futures_close, cancel. "
        "For any endpoint beyond shortcuts, use bitget_uta."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["spot_buy", "spot_sell", "futures_open", "futures_close", "cancel"],
            },
            "symbol": {"type": "string"},
            "size": {"type": "string"},
            "price": {"type": "string"},
            "side": {"type": "string"},
            "trade_side": {"type": "string"},
            "order_type": {"type": "string"},
            "product_type": {"type": "string"},
            "margin_mode": {"type": "string"},
            "order_id": {"type": "string"},
            "order_ids": {
                "type": "array",
                "items": {"type": "string"},
            },
            "cancel_all": {"type": "boolean"},
            "margin_coin": {"type": "string"},
        },
        "required": ["action", "symbol"],
    }

    def __init__(self) -> None:
        self._context: ContextVar[_PrivateBitgetContext] = ContextVar(
            "bitget_trade_tool_context",
            default=_PrivateBitgetContext(),
        )

    def set_context(self, principal_id: str, session_factory: Any) -> None:
        self._context.set(_PrivateBitgetContext(principal_id=principal_id, session_factory=session_factory))

    async def execute(
        self,
        action: str,
        symbol: str,
        size: str = "",
        price: str = "",
        side: str = "",
        trade_side: str = "",
        order_type: str = "market",
        product_type: str = "USDT-FUTURES",
        margin_mode: str = "crossed",
        order_id: str = "",
        order_ids: list[str] | None = None,
        cancel_all: bool = False,
        margin_coin: str = "",
        **kw: Any,
    ) -> str:
        context = self._context.get()
        try:
            gw = await _resolve_user_gateway(context.principal_id, context.session_factory)
        except BitgetError as exc:
            return f"Error: {exc}"
        if gw is None:
            return f"Error: {_NO_CREDENTIALS_MSG}"

        action_key = (action or "").strip()
        try:
            if action_key in {"spot_buy", "spot_sell"}:
                if not size:
                    return "Error: size is required for spot orders"
                if (order_type or "").strip().lower() == "limit" and not price:
                    return "Error: price is required for limit spot orders"
                order_payload: dict[str, Any] = {
                    "symbol": symbol,
                    "side": "buy" if action_key == "spot_buy" else "sell",
                    "orderType": order_type,
                    "size": size,
                }
                if price:
                    order_payload["price"] = price
                data = await gw.spot_place_order_orders([order_payload])
                return _json_result(data, limit=4000)

            if action_key in {"futures_open", "futures_close"}:
                if not size:
                    return "Error: size is required for futures orders"
                if (order_type or "").strip().lower() == "limit" and not price:
                    return "Error: price is required for limit futures orders"
                order_payload = _build_futures_order_args(
                    action=action_key,
                    symbol=symbol,
                    size=size,
                    price=price,
                    side=side,
                    trade_side=trade_side,
                    order_type=order_type,
                    product_type=product_type,
                    margin_mode=margin_mode,
                )
                data = await gw.futures_place_order_orders([order_payload])
                return _json_result(data, limit=4000)

            if action_key == "cancel":
                data = await gw.futures_cancel_orders(
                    symbol=symbol,
                    product_type=product_type,
                    order_id=order_id,
                    order_ids=order_ids,
                    cancel_all=cancel_all,
                    margin_coin=margin_coin,
                )
                return _json_result(data, limit=4000)

            return (
                f"Error: unknown action '{action_key}'. "
                "Use bitget_uta for full UTA endpoint coverage."
            )
        except BitgetError as exc:
            return f"Error: {exc}"
        except Exception as exc:
            return f"Error: bitget_trade action '{action_key}' failed: {exc}"
        finally:
            await gw.close()


class BitgetUtaTool(Tool):
    """Generic direct UTA endpoint caller."""

    name = "bitget_uta"
    description = (
        "Primary interface for full Bitget UTA REST API coverage. "
        "Actions: describe (look up required params BEFORE calling), call, list_catalog, check_coverage. "
        "ALWAYS use describe first for unfamiliar endpoints to see required fields. "
        "GET → params object. POST/PUT/DELETE → body object. Path must be clean (no query string)."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["call", "describe", "list_catalog", "check_coverage"],
                "description": "describe: show required/optional params for an endpoint. call: execute. list_catalog: list all endpoints.",
            },
            "method": {
                "type": "string",
                "enum": ["GET", "POST", "PUT", "DELETE"],
                "description": "HTTP method, required for action=call",
            },
            "path": {
                "type": "string",
                "description": (
                    "Clean Bitget API path WITHOUT query string. "
                    "Example: /api/v2/mix/order/place-plan-order"
                ),
            },
            "params": {
                "type": "object",
                "description": "Query parameters object (for GET requests)",
            },
            "body": {
                "type": "object",
                "description": (
                    "JSON body object — REQUIRED for POST/PUT/DELETE. "
                    "Must contain all mandatory fields per Bitget API docs."
                ),
            },
            "auth_required": {
                "type": "boolean",
                "description": "Set false for public endpoints (action=call)",
                "default": True,
            },
        },
        "required": [],
    }

    def __init__(self) -> None:
        self._context: ContextVar[_PrivateBitgetContext] = ContextVar(
            "bitget_uta_tool_context",
            default=_PrivateBitgetContext(),
        )

    def set_context(self, principal_id: str, session_factory: Any) -> None:
        self._context.set(_PrivateBitgetContext(principal_id=principal_id, session_factory=session_factory))

    async def execute(
        self,
        action: str = "call",
        method: str = "",
        path: str = "",
        params: dict[str, Any] | None = None,
        body: dict[str, Any] | None = None,
        auth_required: bool = True,
        **kw: Any,
    ) -> str:
        action_key = (action or "call").strip().lower()
        if action_key == "list_catalog":
            catalog = list_uta_catalog()
            return _json_result({"count": len(catalog), "endpoints": catalog}, limit=12000)
        if action_key == "check_coverage":
            report = build_uta_coverage_report()
            return _json_result(report, limit=12000)
        if action_key == "describe":
            if not path.strip():
                return "Error: path is required for action='describe'"
            info = describe_endpoint(method or "GET", path.strip())
            if info is None:
                return (
                    f"No schema found for {method or '?'} {path}. "
                    "The endpoint may still work via action='call' — "
                    "check Bitget docs for required fields."
                )
            return _json_result(info, limit=8000)
        if action_key != "call":
            return f"Error: unknown action '{action_key}'"
        if not method.strip() or not path.strip():
            return "Error: method and path are required for action='call'"

        # All path/body/params sanitisation is handled at gateway level.
        context = self._context.get()
        use_auth = bool(auth_required)
        gateway: BitgetGateway | None = None
        if use_auth:
            try:
                gateway = await _resolve_user_gateway(context.principal_id, context.session_factory)
            except BitgetError as exc:
                return f"Error: {exc}"
            if gateway is None:
                return f"Error: {_NO_CREDENTIALS_MSG}"
        else:
            gateway = BitgetGateway()

        try:
            query = params if isinstance(params, dict) else None
            payload = body if isinstance(body, dict) else None
            data = await gateway.request_uta(
                method=method.strip().upper(),
                path=path.strip(),
                params=query,
                body=payload,
                auth_required=use_auth,
            )
            return _json_result(data, limit=12000)
        except BitgetError as exc:
            return f"Error: {exc}"
        except Exception as exc:
            return f"Error: bitget_uta call failed: {exc}"
        finally:
            await gateway.close()
