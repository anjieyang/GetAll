"""Bitget trading tools – exposed to the LLM via tool calling.

The agent decides when and how to call these based on natural language
understanding. No hardcoded keyword routing.

Market tools use public endpoints (no credentials required).
Account and Trade tools MUST use per-user credentials resolved from
the database. If the user hasn't bound their API key, the operation
is rejected — there is no global fallback for personal data.
"""

from __future__ import annotations

import json
from typing import Any

from getall.agent.tools.base import Tool
from getall.integrations.bitget.gateway import BitgetCredentials, BitgetGateway


# ---------------------------------------------------------------------------
# Market gateway singleton — public endpoints, no credentials needed
# ---------------------------------------------------------------------------

_market_gw: BitgetGateway | None = None


def _market_gateway() -> BitgetGateway:
    """Lazy singleton for public market data (no API key required)."""
    global _market_gw
    if _market_gw is None:
        _market_gw = BitgetGateway()
    return _market_gw


# ---------------------------------------------------------------------------
# Per-user gateway — personal credentials from DB
# ---------------------------------------------------------------------------

_NO_CREDENTIALS_MSG = (
    "No exchange credentials bound for this user. "
    "Please bind your Bitget API key first by sending your "
    "API Key, API Secret, and Passphrase in a private chat."
)


async def _resolve_user_gateway(
    principal_id: str,
    session_factory: Any,
) -> BitgetGateway | None:
    """Try to build a BitgetGateway with the user's own credentials."""
    if not principal_id or session_factory is None:
        return None
    try:
        from getall.storage.repository import CredentialRepo

        async with session_factory() as session:
            repo = CredentialRepo(session)
            creds = await repo.get(principal_id, "bitget")
        if creds is None:
            return None
        return BitgetGateway(
            credentials=BitgetCredentials(
                api_key=creds["api_key"],
                api_secret=creds["api_secret"],
                passphrase=creds["passphrase"],
            )
        )
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Market tool — public endpoints, no personal credentials needed
# ---------------------------------------------------------------------------


class BitgetMarketTool(Tool):
    """Query Bitget market data: tickers, candles, depth, funding rate."""

    name = "bitget_market"
    description = (
        "Fetch real-time crypto market data from Bitget. "
        "Actions: tickers, candles, depth, funding_rate. "
        "Default product_type is USDT-FUTURES."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["tickers", "candles", "depth", "funding_rate"],
                "description": "Which market data to fetch",
            },
            "symbol": {
                "type": "string",
                "description": "Trading pair symbol, e.g. BTCUSDT",
            },
            "product_type": {
                "type": "string",
                "description": "Product type: USDT-FUTURES, COIN-FUTURES, USDC-FUTURES, or SPOT",
            },
            "granularity": {
                "type": "string",
                "description": "Candle granularity: 1m,5m,15m,30m,1H,4H,1D,1W",
            },
            "limit": {
                "type": "integer",
                "description": "Number of results to return",
                "minimum": 1,
                "maximum": 200,
            },
        },
        "required": ["action"],
    }

    async def execute(self, action: str, symbol: str = "", product_type: str = "USDT-FUTURES", granularity: str = "1H", limit: int = 20, **kw: Any) -> str:
        gw = _market_gateway()
        try:
            if action == "tickers":
                data = await gw.get_tickers(product_type=product_type)
            elif action == "candles":
                if not symbol:
                    return "Error: symbol is required for candles"
                data = await gw.get_candles(symbol=symbol, granularity=granularity, product_type=product_type, limit=limit)
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
            return json.dumps(data, ensure_ascii=False)[:4000]
        except Exception as e:
            return f"Error: {e}"


# ---------------------------------------------------------------------------
# Account tool — per-user credentials only, no fallback
# ---------------------------------------------------------------------------


class BitgetAccountTool(Tool):
    """Query Bitget account info: all wallets, positions, bills."""

    name = "bitget_account"
    description = (
        "Fetch Bitget account information. "
        "Actions: all_assets (spot+futures+funding in one call), spot_assets, futures_assets, positions, bills. "
        "For any 'do I have money / balance / funds' question, always use all_assets first before any narrower action. "
        "Never conclude 'no money' from futures-only data."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["all_assets", "spot_assets", "futures_assets", "positions", "bills", "assets"],
                "description": "What account data to fetch. Use all_assets to see everything. 'assets' is treated as all_assets.",
                "default": "all_assets",
            },
            "product_type": {
                "type": "string",
                "description": "For futures_assets/positions/bills: USDT-FUTURES, COIN-FUTURES, USDC-FUTURES",
            },
            "symbol": {
                "type": "string",
                "description": "Symbol filter for positions",
            },
            "coin": {
                "type": "string",
                "description": "Coin filter for spot_assets (e.g. USDT, BTC)",
            },
        },
        "required": [],
    }

    def __init__(self) -> None:
        self._principal_id: str = ""
        self._session_factory: Any = None

    def set_context(self, principal_id: str, session_factory: Any) -> None:
        self._principal_id = principal_id
        self._session_factory = session_factory

    async def execute(self, action: str = "all_assets", product_type: str = "USDT-FUTURES", symbol: str = "", coin: str = "", **kw: Any) -> str:
        gw = await _resolve_user_gateway(self._principal_id, self._session_factory)
        if gw is None:
            return f"Error: {_NO_CREDENTIALS_MSG}"

        try:
            action_key = (action or "all_assets").strip() or "all_assets"
            if action_key == "assets":
                action_key = "all_assets"

            if action_key == "all_assets":
                data = await gw.get_all_assets()
            elif action_key == "spot_assets":
                data = await gw.get_spot_assets(coin=coin)
            elif action_key == "futures_assets":
                data = await gw.get_futures_assets(product_type=product_type)
            elif action_key == "positions":
                data = await gw.get_positions(product_type=product_type, symbol=symbol)
            elif action_key == "bills":
                data = await gw.get_bills(product_type=product_type)
            else:
                return f"Error: unknown action '{action_key}'"
            return json.dumps(data, ensure_ascii=False)[:8000]
        except Exception as e:
            return f"Error: {e}"


# ---------------------------------------------------------------------------
# Trade tool — MUST use per-user credentials, never global
# ---------------------------------------------------------------------------


class BitgetTradeTool(Tool):
    """Execute trades on Bitget: spot and futures orders."""

    name = "bitget_trade"
    description = (
        "Place or cancel orders on Bitget. "
        "Actions: spot_buy, spot_sell, futures_open, futures_close, cancel. "
        "IMPORTANT: Requires the user to have bound their own exchange API credentials first. "
        "If no credentials are found, tell the user to bind via private chat."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["spot_buy", "spot_sell", "futures_open", "futures_close", "cancel"],
                "description": "Trade action to execute",
            },
            "symbol": {
                "type": "string",
                "description": "Trading pair, e.g. BTCUSDT",
            },
            "size": {
                "type": "string",
                "description": "Order size/quantity",
            },
            "price": {
                "type": "string",
                "description": "Limit price (empty for market order)",
            },
            "side": {
                "type": "string",
                "description": "For futures: open_long, open_short, close_long, close_short",
            },
            "order_type": {
                "type": "string",
                "description": "market or limit",
            },
            "product_type": {
                "type": "string",
                "description": "USDT-FUTURES, COIN-FUTURES, USDC-FUTURES",
            },
            "margin_mode": {
                "type": "string",
                "description": "crossed or isolated",
            },
            "order_id": {
                "type": "string",
                "description": "Order ID for cancel action",
            },
        },
        "required": ["action", "symbol"],
    }

    def __init__(self) -> None:
        self._principal_id: str = ""
        self._session_factory: Any = None

    def set_context(self, principal_id: str, session_factory: Any) -> None:
        self._principal_id = principal_id
        self._session_factory = session_factory

    async def execute(
        self,
        action: str,
        symbol: str,
        size: str = "",
        price: str = "",
        side: str = "",
        order_type: str = "market",
        product_type: str = "USDT-FUTURES",
        margin_mode: str = "crossed",
        order_id: str = "",
        **kw: Any,
    ) -> str:
        gw = await _resolve_user_gateway(self._principal_id, self._session_factory)
        if gw is None:
            return f"Error: {_NO_CREDENTIALS_MSG}"

        try:
            if action == "spot_buy":
                data = await gw.spot_place_order(symbol=symbol, side="buy", order_type=order_type, size=size, price=price)
            elif action == "spot_sell":
                data = await gw.spot_place_order(symbol=symbol, side="sell", order_type=order_type, size=size, price=price)
            elif action == "futures_open":
                data = await gw.futures_place_order(
                    symbol=symbol, product_type=product_type, margin_mode=margin_mode,
                    side=side or "open_long", order_type=order_type, size=size, price=price,
                )
            elif action == "futures_close":
                data = await gw.futures_place_order(
                    symbol=symbol, product_type=product_type, margin_mode=margin_mode,
                    side=side or "close_long", order_type=order_type, size=size, price=price,
                )
            elif action == "cancel":
                if not order_id:
                    return "Error: order_id is required for cancel"
                data = await gw.futures_cancel_order(symbol=symbol, product_type=product_type, order_id=order_id)
            else:
                return f"Error: unknown action '{action}'"
            return json.dumps(data, ensure_ascii=False)[:4000]
        except Exception as e:
            return f"Error: {e}"
