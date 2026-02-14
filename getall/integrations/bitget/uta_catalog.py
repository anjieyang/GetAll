"""Bitget UTA endpoint catalog, parameter docs, and coverage helpers.

Full catalog of Bitget v2 REST API endpoints with parameter schemas.
The agent calls ``describe_endpoint()`` before invoking unfamiliar endpoints
so it knows exactly what fields to pass.
"""

from __future__ import annotations

from typing import Any

_GATEWAY_METHODS = frozenset({"GET", "POST", "PUT", "DELETE"})

# ---------------------------------------------------------------------------
# Parameter schema type: (name, type, required, description)
# ---------------------------------------------------------------------------
_P = tuple[str, str, bool, str]

def _ps(*fields: _P) -> list[dict[str, Any]]:
    """Build a parameter list from compact tuples."""
    return [
        {"name": n, "type": t, "required": r, "description": d}
        for n, t, r, d in fields
    ]


# ── Reusable field groups ────────────────────────────────────────────────

_PRODUCT_TYPE = ("productType", "string", True, "USDT-FUTURES | COIN-FUTURES | USDC-FUTURES")
_SYMBOL = ("symbol", "string", True, "Trading pair, e.g. BTCUSDT")
_MARGIN_COIN = ("marginCoin", "string", True, "Margin coin, e.g. USDT")
_MARGIN_MODE = ("marginMode", "string", True, "isolated | crossed")
_ORDER_TYPE = ("orderType", "string", True, "limit | market")
_SIDE = ("side", "string", True, "buy | sell")
_SIZE = ("size", "string", True, "Order quantity (base coin)")
_PRICE = ("price", "string", False, "Limit price (required if orderType=limit)")
_CLIENT_OID = ("clientOid", "string", False, "Custom order ID")
_COIN_OPT = ("coin", "string", False, "Coin filter, e.g. USDT")
_LIMIT = ("limit", "string", False, "Number of results (default varies)")
_ID_AFTER = ("idLessThan", "string", False, "Pagination cursor (older)")
_PRODUCT_TYPE_OPT = ("productType", "string", False, "USDT-FUTURES | COIN-FUTURES | USDC-FUTURES")


# ---------------------------------------------------------------------------
# Endpoint registry: (method, path, module, auth, params_schema)
# ---------------------------------------------------------------------------

_KNOWN_ENDPOINTS: tuple[tuple[str, str, str, str, list[dict[str, Any]]], ...] = (

    # ── Common / Public ──────────────────────────────────────────────────
    ("GET", "/api/v2/public/time", "common", "public", []),
    ("GET", "/api/v2/spot/public/coins", "common", "public",
     _ps(_COIN_OPT)),

    # ── Funding / Account ────────────────────────────────────────────────
    ("GET", "/api/v2/account/funding-assets", "account", "private",
     _ps(_COIN_OPT)),
    ("GET", "/api/v2/account/all-account-balance", "account", "private", []),

    # ── Wallet ───────────────────────────────────────────────────────────
    ("POST", "/api/v2/spot/wallet/transfer", "wallet", "private",
     _ps(("fromType", "string", True, "Source account: spot | mix_usdt | mix_usd | mix_usdc"),
         ("toType", "string", True, "Target account: spot | mix_usdt | mix_usd | mix_usdc"),
         ("amount", "string", True, "Transfer amount"),
         ("coin", "string", True, "Coin, e.g. USDT"),
         _CLIENT_OID)),
    ("POST", "/api/v2/spot/wallet/subaccount-transfer", "wallet", "private",
     _ps(("fromType", "string", True, "spot | mix_usdt etc."),
         ("toType", "string", True, "spot | mix_usdt etc."),
         ("amount", "string", True, "Transfer amount"),
         ("coin", "string", True, "Coin"),
         ("fromUserId", "string", True, "Source UID"),
         ("toUserId", "string", True, "Target UID"),
         _CLIENT_OID)),
    ("GET", "/api/v2/spot/account/transferRecords", "wallet", "private",
     _ps(("coin", "string", False, "Coin filter"),
         ("fromType", "string", False, "Source account type"),
         _LIMIT, _ID_AFTER)),
    ("GET", "/api/v2/spot/wallet/transfer-coin-info", "wallet", "private",
     _ps(("coin", "string", False, "Coin filter"),
         ("fromType", "string", False, "Source type"),
         ("toType", "string", False, "Target type"))),
    ("GET", "/api/v2/spot/wallet/deposit-address", "wallet", "private",
     _ps(("coin", "string", True, "Coin"),
         ("chain", "string", False, "Chain name"))),
    ("GET", "/api/v2/spot/wallet/deposit-records", "wallet", "private",
     _ps(("coin", "string", False, "Coin filter"), _LIMIT, _ID_AFTER)),
    ("GET", "/api/v2/spot/wallet/withdrawal-records", "wallet", "private",
     _ps(("coin", "string", False, "Coin filter"), _LIMIT, _ID_AFTER)),
    ("POST", "/api/v2/spot/wallet/withdrawal", "wallet", "private",
     _ps(("coin", "string", True, "Coin"),
         ("transferType", "string", True, "on_chain | internal"),
         ("address", "string", True, "Withdrawal address"),
         ("size", "string", True, "Amount"),
         ("chain", "string", False, "Chain (required for on_chain)"),
         ("tag", "string", False, "Memo/tag"),
         _CLIENT_OID)),
    ("POST", "/api/v2/spot/wallet/modify-deposit-account", "wallet", "private",
     _ps(("accountType", "string", True, "Target account after deposit"),
         ("coin", "string", False, "Coin filter"))),

    # ── Convert ──────────────────────────────────────────────────────────
    ("POST", "/api/v2/convert/quoted-price", "convert", "private",
     _ps(("fromCoin", "string", True, "Source coin"),
         ("toCoin", "string", True, "Target coin"),
         ("fromCoinSize", "string", False, "Amount of source coin"),
         ("toCoinSize", "string", False, "Amount of target coin"))),

    # ── Spot Account ─────────────────────────────────────────────────────
    ("GET", "/api/v2/spot/account/assets", "spot", "private",
     _ps(_COIN_OPT,
         ("assetType", "string", False, "hold_only | all (default hold_only)"))),
    ("GET", "/api/v2/spot/account/subaccount-assets", "spot", "private", []),
    ("GET", "/api/v2/spot/account/bills", "spot", "private",
     _ps(_COIN_OPT, _LIMIT, _ID_AFTER)),

    # ── Spot Market ──────────────────────────────────────────────────────
    ("GET", "/api/v2/spot/public/symbols", "spot", "public",
     _ps(("symbol", "string", False, "Filter single symbol"))),
    ("GET", "/api/v2/spot/market/tickers", "spot", "public",
     _ps(("symbol", "string", False, "Filter single symbol"))),
    ("GET", "/api/v2/spot/market/orderbook", "spot", "public",
     _ps(_SYMBOL, ("type", "string", False, "step0-step5"), _LIMIT)),
    ("GET", "/api/v2/spot/market/candles", "spot", "public",
     _ps(_SYMBOL, ("granularity", "string", True, "1min|5min|15min|30min|1h|4h|1day|1week"),
         _LIMIT)),
    ("GET", "/api/v2/spot/market/history-candles", "spot", "public",
     _ps(_SYMBOL, ("granularity", "string", True, "Same as candles"),
         _LIMIT, ("endTime", "string", False, "End time ms"))),
    ("GET", "/api/v2/spot/market/fills", "spot", "public",
     _ps(_SYMBOL, _LIMIT)),
    ("GET", "/api/v2/spot/market/fills-history", "spot", "public",
     _ps(_SYMBOL, _LIMIT, _ID_AFTER)),
    ("GET", "/api/v2/spot/market/vip-fee-rate", "spot", "public", []),

    # ── Spot Trade ───────────────────────────────────────────────────────
    ("POST", "/api/v2/spot/trade/place-order", "spot", "private",
     _ps(_SYMBOL, _SIDE, _ORDER_TYPE,
         ("force", "string", True, "gtc | ioc | fok | post_only"),
         _SIZE, _PRICE, _CLIENT_OID,
         ("stpMode", "string", False, "none | cancel_taker | cancel_maker | cancel_both"))),
    ("POST", "/api/v2/spot/trade/cancel-order", "spot", "private",
     _ps(_SYMBOL,
         ("orderId", "string", False, "Order ID (orderId or clientOid required)"),
         _CLIENT_OID)),
    ("POST", "/api/v2/spot/trade/batch-orders", "spot", "private",
     _ps(_SYMBOL,
         ("orderList", "array", True, "Array of order objects (same fields as place-order)"))),
    ("POST", "/api/v2/spot/trade/batch-cancel-order", "spot", "private",
     _ps(_SYMBOL,
         ("orderList", "array", True, "Array of {orderId} or {clientOid}"))),
    ("POST", "/api/v2/spot/trade/cancel-symbol-order", "spot", "private",
     _ps(_SYMBOL)),
    ("GET", "/api/v2/spot/trade/orderInfo", "spot", "private",
     _ps(("orderId", "string", False, "Order ID"), _CLIENT_OID)),
    ("GET", "/api/v2/spot/trade/unfilled-orders", "spot", "private",
     _ps(("symbol", "string", False, "Filter symbol"), _LIMIT, _ID_AFTER)),
    ("GET", "/api/v2/spot/trade/history-orders", "spot", "private",
     _ps(("symbol", "string", False, "Filter symbol"), _LIMIT, _ID_AFTER)),
    ("GET", "/api/v2/spot/trade/fills", "spot", "private",
     _ps(("symbol", "string", False, "Filter symbol"), _LIMIT, _ID_AFTER,
         ("orderId", "string", False, "Filter by order ID"))),

    # ── Spot Plan ────────────────────────────────────────────────────────
    ("POST", "/api/v2/spot/trade/place-plan-order", "spot", "private",
     _ps(_SYMBOL, _SIDE, _ORDER_TYPE,
         _SIZE, _PRICE,
         ("triggerPrice", "string", True, "Trigger price"),
         ("triggerType", "string", False, "fill_price | mark_price"),
         _CLIENT_OID)),
    ("POST", "/api/v2/spot/trade/modify-plan-order", "spot", "private",
     _ps(("orderId", "string", True, "Plan order ID"),
         _SIZE, _PRICE,
         ("triggerPrice", "string", False, "New trigger price"),
         ("orderType", "string", False, "limit | market"))),
    ("POST", "/api/v2/spot/trade/cancel-plan-order", "spot", "private",
     _ps(("orderId", "string", True, "Plan order ID"))),
    ("POST", "/api/v2/spot/trade/batch-cancel-plan-order", "spot", "private",
     _ps(("orderList", "array", True, "Array of {orderId}"))),
    ("GET", "/api/v2/spot/trade/current-plan-order", "spot", "private",
     _ps(("symbol", "string", False, "Filter symbol"), _LIMIT)),
    ("GET", "/api/v2/spot/trade/history-plan-order", "spot", "private",
     _ps(("symbol", "string", False, "Filter symbol"), _LIMIT, _ID_AFTER)),

    # ── Contract Account ─────────────────────────────────────────────────
    ("GET", "/api/v2/mix/account/account", "contract", "private",
     _ps(_SYMBOL, _PRODUCT_TYPE, _MARGIN_COIN)),
    ("GET", "/api/v2/mix/account/accounts", "contract", "private",
     _ps(_PRODUCT_TYPE)),
    ("GET", "/api/v2/mix/account/bill", "contract", "private",
     _ps(_PRODUCT_TYPE, _COIN_OPT, _LIMIT, _ID_AFTER)),
    ("POST", "/api/v2/mix/account/set-leverage", "contract", "private",
     _ps(_SYMBOL, _PRODUCT_TYPE, _MARGIN_COIN,
         ("leverage", "string", True, "Leverage value"),
         ("holdSide", "string", False, "long | short (hedge mode)"))),
    ("POST", "/api/v2/mix/account/set-all-leverage", "contract", "private",
     _ps(_PRODUCT_TYPE,
         ("leverage", "string", True, "Leverage for all pairs"))),
    ("POST", "/api/v2/mix/account/set-margin", "contract", "private",
     _ps(_SYMBOL, _PRODUCT_TYPE, _MARGIN_COIN,
         ("amount", "string", True, "Margin amount to add/reduce"),
         ("holdSide", "string", True, "long | short"))),
    ("POST", "/api/v2/mix/account/set-margin-mode", "contract", "private",
     _ps(_SYMBOL, _PRODUCT_TYPE, _MARGIN_COIN, _MARGIN_MODE)),
    ("POST", "/api/v2/mix/account/set-position-mode", "contract", "private",
     _ps(_PRODUCT_TYPE,
         ("posMode", "string", True, "one_way_mode | hedge_mode"))),

    # ── Contract Market ──────────────────────────────────────────────────
    ("GET", "/api/v2/mix/market/contracts", "contract", "public",
     _ps(_PRODUCT_TYPE, ("symbol", "string", False, "Filter single symbol"))),
    ("GET", "/api/v2/mix/market/ticker", "contract", "public",
     _ps(_SYMBOL, _PRODUCT_TYPE)),
    ("GET", "/api/v2/mix/market/tickers", "contract", "public",
     _ps(_PRODUCT_TYPE)),
    ("GET", "/api/v2/mix/market/depth", "contract", "public",
     _ps(_SYMBOL, _PRODUCT_TYPE, _LIMIT)),
    ("GET", "/api/v2/mix/market/candles", "contract", "public",
     _ps(_SYMBOL, _PRODUCT_TYPE,
         ("granularity", "string", True, "1m|5m|15m|30m|1H|4H|1D|1W"),
         _LIMIT)),
    ("GET", "/api/v2/mix/market/fills-history", "contract", "public",
     _ps(_SYMBOL, _PRODUCT_TYPE, _LIMIT)),
    ("GET", "/api/v2/mix/market/open-interest", "contract", "public",
     _ps(_SYMBOL, _PRODUCT_TYPE)),
    ("GET", "/api/v2/mix/market/funding-time", "contract", "public",
     _ps(_SYMBOL, _PRODUCT_TYPE)),
    ("GET", "/api/v2/mix/market/history-fund-rate", "contract", "public",
     _ps(_SYMBOL, _PRODUCT_TYPE, _LIMIT)),
    ("GET", "/api/v2/mix/market/current-fund-rate", "contract", "public",
     _ps(_SYMBOL, _PRODUCT_TYPE)),
    ("GET", "/api/v2/mix/market/mark-price", "contract", "public",
     _ps(_SYMBOL, _PRODUCT_TYPE)),
    ("GET", "/api/v2/mix/market/leverage-range", "contract", "public",
     _ps(_SYMBOL, _PRODUCT_TYPE)),
    ("GET", "/api/v2/mix/market/discount-rate", "contract", "public", []),

    # ── Contract Trade ───────────────────────────────────────────────────
    ("POST", "/api/v2/mix/order/place-order", "contract", "private",
     _ps(_SYMBOL, _PRODUCT_TYPE, _MARGIN_MODE, _MARGIN_COIN,
         _SIZE, _PRICE, _SIDE,
         ("tradeSide", "string", False, "open | close (required in hedge_mode)"),
         _ORDER_TYPE,
         ("force", "string", False, "ioc | fok | gtc (default) | post_only"),
         _CLIENT_OID,
         ("reduceOnly", "string", False, "YES | NO (one_way_mode only)"),
         ("presetStopSurplusPrice", "string", False, "Take-profit price"),
         ("presetStopLossPrice", "string", False, "Stop-loss price"),
         ("stpMode", "string", False, "none | cancel_taker | cancel_maker | cancel_both"))),
    ("POST", "/api/v2/mix/order/batch-place-order", "contract", "private",
     _ps(_SYMBOL, _PRODUCT_TYPE, _MARGIN_MODE, _MARGIN_COIN,
         ("orderList", "array", True, "Array of order objects"))),
    ("POST", "/api/v2/mix/order/modify-order", "contract", "private",
     _ps(_SYMBOL, _PRODUCT_TYPE,
         ("orderId", "string", False, "Order ID (orderId or clientOid)"),
         _CLIENT_OID,
         ("newSize", "string", False, "New size"),
         ("newPrice", "string", False, "New price"))),
    ("POST", "/api/v2/mix/order/cancel-order", "contract", "private",
     _ps(_SYMBOL, _PRODUCT_TYPE, _MARGIN_COIN,
         ("orderId", "string", False, "Order ID (orderId or clientOid)"),
         _CLIENT_OID)),
    ("POST", "/api/v2/mix/order/batch-cancel-orders", "contract", "private",
     _ps(_SYMBOL, _PRODUCT_TYPE, _MARGIN_COIN,
         ("orderIdList", "array", True, "Array of orderId strings"))),
    ("POST", "/api/v2/mix/order/cancel-all-orders", "contract", "private",
     _ps(_PRODUCT_TYPE, ("symbol", "string", False, "Cancel for one symbol only"),
         _MARGIN_COIN)),
    ("POST", "/api/v2/mix/order/close-positions", "contract", "private",
     _ps(_SYMBOL, _PRODUCT_TYPE, ("holdSide", "string", False, "long | short"))),
    ("POST", "/api/v2/mix/order/click-backhand", "contract", "private",
     _ps(_SYMBOL, _PRODUCT_TYPE, _MARGIN_COIN, _SIDE, _SIZE,
         ("tradeSide", "string", False, "open | close"))),
    ("GET", "/api/v2/mix/order/detail", "contract", "private",
     _ps(_SYMBOL, _PRODUCT_TYPE,
         ("orderId", "string", False, "Order ID"), _CLIENT_OID)),
    ("GET", "/api/v2/mix/order/fills", "contract", "private",
     _ps(_PRODUCT_TYPE, ("orderId", "string", False, "Filter"), _LIMIT, _ID_AFTER)),
    ("GET", "/api/v2/mix/order/orders-pending", "contract", "private",
     _ps(_PRODUCT_TYPE, ("symbol", "string", False, "Filter"), _LIMIT, _ID_AFTER)),
    ("GET", "/api/v2/mix/order/orders-history", "contract", "private",
     _ps(_PRODUCT_TYPE, ("symbol", "string", False, "Filter"), _LIMIT, _ID_AFTER)),

    # ── Contract Plan (Trigger Orders) ───────────────────────────────────
    ("POST", "/api/v2/mix/order/place-plan-order", "contract", "private",
     _ps(_SYMBOL, _PRODUCT_TYPE, _MARGIN_MODE, _MARGIN_COIN,
         _SIZE, _SIDE,
         ("tradeSide", "string", False, "open | close (hedge_mode)"),
         _ORDER_TYPE, _PRICE,
         ("triggerPrice", "string", True, "Trigger price"),
         ("triggerType", "string", False, "fill_price | mark_price"),
         ("planType", "string", False, "normal_plan (default) | track_plan"),
         _CLIENT_OID,
         ("reduceOnly", "string", False, "YES | NO"),
         ("presetStopSurplusPrice", "string", False, "TP price after triggered"),
         ("presetStopLossPrice", "string", False, "SL price after triggered"),
         ("callbackRatio", "string", False, "Trailing stop callback ratio"))),
    ("POST", "/api/v2/mix/order/place-stop-order", "contract", "private",
     _ps(_SYMBOL, _PRODUCT_TYPE, _MARGIN_MODE,
         ("planType", "string", True, "profit_plan | loss_plan | moving_plan | pos_profit | pos_loss"),
         ("triggerPrice", "string", True, "Trigger price"),
         ("triggerType", "string", False, "fill_price | mark_price"),
         ("holdSide", "string", False, "long | short"),
         _SIZE)),
    ("POST", "/api/v2/mix/order/modify-plan-order", "contract", "private",
     _ps(("orderId", "string", True, "Plan order ID"), _PRODUCT_TYPE,
         ("newTriggerPrice", "string", False, "New trigger price"),
         ("newSize", "string", False, "New size"),
         ("newPrice", "string", False, "New price"))),
    ("POST", "/api/v2/mix/order/modify-stop-order", "contract", "private",
     _ps(("orderId", "string", True, "Stop order ID"), _PRODUCT_TYPE,
         _MARGIN_COIN,
         ("newTriggerPrice", "string", False, "New trigger price"),
         ("newSize", "string", False, "New size"))),
    ("POST", "/api/v2/mix/order/cancel-plan-order", "contract", "private",
     _ps(("orderId", "string", True, "Plan order ID"), _PRODUCT_TYPE,
         _MARGIN_COIN)),
    ("POST", "/api/v2/mix/order/cancel-all-plan-orders", "contract", "private",
     _ps(_PRODUCT_TYPE, ("symbol", "string", False, "Symbol filter"),
         ("planType", "string", False, "Filter by plan type"))),
    ("GET", "/api/v2/mix/order/orders-plan-pending", "contract", "private",
     _ps(_PRODUCT_TYPE, ("symbol", "string", False, "Filter"), _LIMIT, _ID_AFTER)),
    ("GET", "/api/v2/mix/order/orders-plan-history", "contract", "private",
     _ps(_PRODUCT_TYPE, ("symbol", "string", False, "Filter"), _LIMIT, _ID_AFTER)),
    ("GET", "/api/v2/mix/order/orders-stop-pending", "contract", "private",
     _ps(_PRODUCT_TYPE, ("symbol", "string", False, "Filter"), _LIMIT)),
    ("GET", "/api/v2/mix/order/orders-stop-history", "contract", "private",
     _ps(_PRODUCT_TYPE, ("symbol", "string", False, "Filter"), _LIMIT, _ID_AFTER)),

    # ── Contract Position ────────────────────────────────────────────────
    ("GET", "/api/v2/mix/position/single-position", "contract", "private",
     _ps(_SYMBOL, _PRODUCT_TYPE, _MARGIN_COIN)),
    ("GET", "/api/v2/mix/position/all-position", "contract", "private",
     _ps(_PRODUCT_TYPE, _MARGIN_COIN)),
    ("GET", "/api/v2/mix/position/history-position", "contract", "private",
     _ps(_PRODUCT_TYPE, ("symbol", "string", False, "Filter"), _LIMIT)),

    # ── Margin (Cross) ───────────────────────────────────────────────────
    ("POST", "/api/v2/margin/cross/account/borrow", "margin", "private",
     _ps(("coin", "string", True, "Borrow coin"),
         ("borrowAmount", "string", True, "Borrow amount"),
         _CLIENT_OID)),
    ("POST", "/api/v2/margin/cross/account/repay", "margin", "private",
     _ps(("coin", "string", True, "Repay coin"),
         ("repayAmount", "string", True, "Repay amount"),
         _CLIENT_OID)),
    ("POST", "/api/v2/margin/cross/account/flash-repay", "margin", "private",
     _ps(("coin", "string", False, "Coin (empty = all)"))),
    ("POST", "/api/v2/margin/cross/account/flash-repay-status", "margin", "private",
     _ps(("idList", "array", False, "Flash repay ID list"))),
    ("GET", "/api/v2/margin/cross/account/assets", "margin", "private",
     _ps(("coin", "string", False, "Coin filter"))),
    ("GET", "/api/v2/margin/cross/account/max-borrowable-amount", "margin", "private",
     _ps(("coin", "string", True, "Coin"))),
    ("GET", "/api/v2/margin/cross/account/max-transfer-out-amount", "margin", "private",
     _ps(("coin", "string", True, "Coin"))),
    ("GET", "/api/v2/margin/cross/borrow-history", "margin", "private",
     _ps(("coin", "string", False, "Filter"), _LIMIT, _ID_AFTER)),
    ("GET", "/api/v2/margin/cross/repay-history", "margin", "private",
     _ps(("coin", "string", False, "Filter"), _LIMIT, _ID_AFTER)),
    ("GET", "/api/v2/margin/cross/interest-history", "margin", "private",
     _ps(("coin", "string", False, "Filter"), _LIMIT, _ID_AFTER)),
    ("GET", "/api/v2/margin/cross/liquidation-history", "margin", "private",
     _ps(_LIMIT, _ID_AFTER)),
    ("GET", "/api/v2/margin/cross/financial-records", "margin", "private",
     _ps(("coin", "string", False, "Filter"), _LIMIT, _ID_AFTER)),
    ("GET", "/api/v2/margin/cross/interest-rate-and-limit", "margin", "private",
     _ps(("coin", "string", False, "Filter"))),
    ("GET", "/api/v2/margin/cross/tier-data", "margin", "private",
     _ps(("coin", "string", False, "Filter"))),

    # ── Margin (Isolated) ────────────────────────────────────────────────
    ("POST", "/api/v2/margin/isolated/account/borrow", "margin", "private",
     _ps(_SYMBOL, ("coin", "string", True, "Borrow coin"),
         ("borrowAmount", "string", True, "Amount"), _CLIENT_OID)),
    ("POST", "/api/v2/margin/isolated/account/repay", "margin", "private",
     _ps(_SYMBOL, ("coin", "string", True, "Repay coin"),
         ("repayAmount", "string", True, "Amount"), _CLIENT_OID)),
    ("POST", "/api/v2/margin/isolated/account/flash-repay", "margin", "private",
     _ps(_SYMBOL, ("coin", "string", False, "Coin (empty = all)"))),
    ("POST", "/api/v2/margin/isolated/account/query-flash-repay-status", "margin", "private",
     _ps(("idList", "array", False, "Flash repay ID list"))),
    ("GET", "/api/v2/margin/isolated/account/assets", "margin", "private",
     _ps(("symbol", "string", False, "Filter"), ("coin", "string", False, "Filter"))),
    ("GET", "/api/v2/margin/isolated/account/max-borrowable-amount", "margin", "private",
     _ps(_SYMBOL, ("coin", "string", True, "Coin"))),
    ("GET", "/api/v2/margin/isolated/account/max-transfer-out-amount", "margin", "private",
     _ps(_SYMBOL, ("coin", "string", True, "Coin"))),
    ("GET", "/api/v2/margin/isolated/borrow-history", "margin", "private",
     _ps(("symbol", "string", False, "Filter"), ("coin", "string", False, "Filter"), _LIMIT, _ID_AFTER)),
    ("GET", "/api/v2/margin/isolated/repay-history", "margin", "private",
     _ps(("symbol", "string", False, "Filter"), ("coin", "string", False, "Filter"), _LIMIT, _ID_AFTER)),
    ("GET", "/api/v2/margin/isolated/interest-history", "margin", "private",
     _ps(("symbol", "string", False, "Filter"), ("coin", "string", False, "Filter"), _LIMIT, _ID_AFTER)),
    ("GET", "/api/v2/margin/isolated/liquidation-history", "margin", "private",
     _ps(("symbol", "string", False, "Filter"), _LIMIT, _ID_AFTER)),
    ("GET", "/api/v2/margin/isolated/financial-records", "margin", "private",
     _ps(("symbol", "string", False, "Filter"), ("coin", "string", False, "Filter"), _LIMIT, _ID_AFTER)),
    ("GET", "/api/v2/margin/isolated/interest-rate-and-limit", "margin", "private",
     _ps(("symbol", "string", False, "Filter"), ("coin", "string", False, "Filter"))),
    ("GET", "/api/v2/margin/isolated/tier-data", "margin", "private",
     _ps(("symbol", "string", False, "Filter"), ("coin", "string", False, "Filter"))),
    ("GET", "/api/v2/margin/currencies", "margin", "public", []),
)


# Shortcut-wrapped endpoints
_SHORTCUT_ENDPOINTS = frozenset(
    {
        ("GET",  "/api/v2/account/funding-assets"),
        ("GET",  "/api/v2/mix/account/accounts"),
        ("GET",  "/api/v2/mix/account/bill"),
        ("GET",  "/api/v2/mix/market/candles"),
        ("GET",  "/api/v2/mix/market/current-fund-rate"),
        ("GET",  "/api/v2/mix/market/depth"),
        ("GET",  "/api/v2/mix/market/tickers"),
        ("GET",  "/api/v2/mix/position/all-position"),
        ("GET",  "/api/v2/spot/account/assets"),
        ("POST", "/api/v2/mix/order/cancel-order"),
        ("POST", "/api/v2/mix/order/place-order"),
        ("POST", "/api/v2/spot/trade/place-order"),
    }
)

# Fast lookup index: (method, path) → full entry
_INDEX: dict[tuple[str, str], tuple[str, str, str, str, list[dict[str, Any]]]] = {
    (m, p): entry for entry in _KNOWN_ENDPOINTS for m, p, *_ in [entry]
}


def describe_endpoint(method: str, path: str) -> dict[str, Any] | None:
    """Return parameter schema for a specific endpoint, or None if not found."""
    key = (method.upper().strip(), path.strip())
    entry = _INDEX.get(key)
    if entry is None:
        # fuzzy: try path only
        for (m, p), e in _INDEX.items():
            if p == key[1]:
                entry = e
                break
    if entry is None:
        return None
    m, p, mod, auth, params = entry
    required = [f for f in params if f["required"]]
    optional = [f for f in params if not f["required"]]
    return {
        "method": m,
        "path": p,
        "module": mod,
        "auth": auth,
        "required_fields": required,
        "optional_fields": optional,
    }


def list_uta_catalog(
    module: str = "",
    auth: str = "",
) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    for method, path, mod, vis, _params in _KNOWN_ENDPOINTS:
        if module and mod != module:
            continue
        if auth and vis != auth:
            continue
        result.append({"method": method, "path": path, "module": mod, "auth": vis})
    result.sort(key=lambda e: (e["module"], e["method"], e["path"]))
    return result


def build_uta_coverage_report(module: str = "") -> dict[str, Any]:
    catalog = list_uta_catalog(module=module)
    unsupported_method: list[dict[str, str]] = []
    invalid_path: list[dict[str, str]] = []
    for endpoint in catalog:
        if endpoint["method"].upper() not in _GATEWAY_METHODS:
            unsupported_method.append(endpoint)
        if not endpoint["path"].startswith("/api/"):
            invalid_path.append(endpoint)
    shortcut_count = sum(
        1 for e in catalog if (e["method"], e["path"]) in _SHORTCUT_ENDPOINTS
    )
    uncovered = len(unsupported_method) + len(invalid_path)
    modules: dict[str, int] = {}
    for e in catalog:
        modules[e["module"]] = modules.get(e["module"], 0) + 1
    return {
        "catalog_total": len(catalog),
        "catalog_by_module": modules,
        "generic_gateway_coverage": "100%" if uncovered == 0 else "partial",
        "shortcut_coverage": f"{shortcut_count}/{len(catalog)}",
    }
