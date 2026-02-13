"""BitgetGateway – unified async Bitget API client.

Self-contained HMAC-SHA256 signing (no third-party SDK dependency).
Adds: idempotency keys, rate-limit guard, domain grouping, error mapping.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import httpx

from getall.settings import GetAllSettings, get_settings


class BitgetError(RuntimeError):
    """Wrapper for all Bitget API errors."""


@dataclass(frozen=True, slots=True)
class BitgetCredentials:
    api_key: str
    api_secret: str
    passphrase: str


class _RateLimiter:
    """Simple token-bucket rate limiter (per-instance)."""

    def __init__(self, rps: float = 10.0) -> None:
        self._interval = 1.0 / rps
        self._last = 0.0
        self._lock = asyncio.Lock()

    async def wait(self) -> None:
        async with self._lock:
            now = time.monotonic()
            gap = self._interval - (now - self._last)
            if gap > 0:
                await asyncio.sleep(gap)
            self._last = time.monotonic()


class BitgetGateway:
    """Async gateway to all Bitget API domains.

    Capability domains (mirrors Bitget V2 API structure):
      - market:  tickers, candles, depth, funding rates
      - account: balances, bills, transfers
      - spot:    place/cancel/query spot orders
      - futures: place/cancel/query futures (USDT-M / Coin-M) orders
      - position: open positions, margin, leverage
      - plan:    trigger / TP-SL plan orders
      - ws:      websocket subscription helpers (public + private)
    """

    def __init__(
        self,
        credentials: BitgetCredentials | None = None,
        settings: GetAllSettings | None = None,
        rps: float = 10.0,
    ) -> None:
        s = settings or get_settings()
        self._creds = credentials or BitgetCredentials(
            api_key="", api_secret="", passphrase="",
        )
        self._base = s.bitget_base_url.rstrip("/")
        self._limiter = _RateLimiter(rps)
        self._http = httpx.AsyncClient(timeout=15.0, base_url=self._base)

    # ── low-level request ────────────────────────────────────────────────

    async def _request(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        body: dict[str, Any] | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        await self._limiter.wait()
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._creds.api_key:
            ts = str(int(time.time() * 1000))
            query = "&".join(f"{k}={v}" for k, v in sorted((params or {}).items()))
            path_query = f"{path}?{query}" if query else path
            raw_body = json.dumps(body, separators=(",", ":"), sort_keys=True) if body else ""
            sign_payload = f"{ts}{method.upper()}{path_query}{raw_body}"
            sig = base64.b64encode(
                hmac.new(
                    self._creds.api_secret.encode("utf-8"),
                    sign_payload.encode("utf-8"),
                    hashlib.sha256,
                ).digest()
            ).decode("utf-8")
            headers.update({
                "ACCESS-KEY": self._creds.api_key,
                "ACCESS-SIGN": sig,
                "ACCESS-TIMESTAMP": ts,
                "ACCESS-PASSPHRASE": self._creds.passphrase,
            })
            if method.upper() == "POST":
                headers["clientOid"] = idempotency_key or str(uuid.uuid4())

        resp = await self._http.request(method, path, params=params, json=body, headers=headers)
        if resp.status_code >= 400:
            raise BitgetError(f"HTTP {resp.status_code}: {resp.text[:500]}")
        data = resp.json()
        if not isinstance(data, dict):
            raise BitgetError("unexpected response shape")
        if data.get("code") and str(data["code"]) != "00000":
            raise BitgetError(f"API error {data.get('code')}: {data.get('msg', '')}")
        return data

    # ── market ───────────────────────────────────────────────────────────

    async def get_tickers(self, product_type: str = "USDT-FUTURES") -> dict[str, Any]:
        return await self._request("GET", "/api/v2/mix/market/tickers", params={"productType": product_type})

    async def get_candles(self, symbol: str, granularity: str = "1H", product_type: str = "USDT-FUTURES", limit: int = 100) -> dict[str, Any]:
        return await self._request("GET", "/api/v2/mix/market/candles", params={
            "symbol": symbol, "productType": product_type, "granularity": granularity, "limit": str(limit),
        })

    async def get_depth(self, symbol: str, product_type: str = "USDT-FUTURES", limit: int = 15) -> dict[str, Any]:
        return await self._request("GET", "/api/v2/mix/market/depth", params={
            "symbol": symbol, "productType": product_type, "limit": str(limit),
        })

    async def get_funding_rate(self, symbol: str, product_type: str = "USDT-FUTURES") -> dict[str, Any]:
        return await self._request("GET", "/api/v2/mix/market/current-fund-rate", params={
            "symbol": symbol, "productType": product_type,
        })

    # ── account ──────────────────────────────────────────────────────────

    async def get_spot_assets(self, coin: str = "") -> dict[str, Any]:
        """GET /api/v2/spot/account/assets — spot wallet balances."""
        params: dict[str, str] = {}
        if coin:
            params["coin"] = coin
        return await self._request("GET", "/api/v2/spot/account/assets", params=params or None)

    async def get_futures_assets(self, product_type: str = "USDT-FUTURES") -> dict[str, Any]:
        """GET /api/v2/mix/account/accounts — futures margin balances."""
        return await self._request("GET", "/api/v2/mix/account/accounts", params={"productType": product_type})

    async def get_funding_assets(self) -> dict[str, Any]:
        """GET /api/v2/account/funding-assets — funding/main account balances."""
        return await self._request("GET", "/api/v2/account/funding-assets")

    async def get_all_assets(self) -> dict[str, Any]:
        """Fetch spot + funding + all futures account balances in one call."""
        results: dict[str, Any] = {}
        for name, coro in [
            ("spot", self.get_spot_assets()),
            ("funding", self.get_funding_assets()),
            ("usdt_futures", self.get_futures_assets("USDT-FUTURES")),
            ("usdc_futures", self.get_futures_assets("USDC-FUTURES")),
            ("coin_futures", self.get_futures_assets("COIN-FUTURES")),
        ]:
            try:
                results[name] = await coro
            except Exception as e:
                results[name] = {"error": str(e)}
        return results

    async def get_bills(self, product_type: str = "USDT-FUTURES", limit: int = 20) -> dict[str, Any]:
        return await self._request("GET", "/api/v2/mix/account/bill", params={"productType": product_type, "limit": str(limit)})

    # ── spot trading ─────────────────────────────────────────────────────

    async def spot_place_order(self, symbol: str, side: str, order_type: str, size: str, price: str = "", client_oid: str | None = None) -> dict[str, Any]:
        body: dict[str, Any] = {"symbol": symbol, "side": side, "orderType": order_type, "size": size}
        if price:
            body["price"] = price
        return await self._request("POST", "/api/v2/spot/trade/place-order", body=body, idempotency_key=client_oid)

    async def spot_cancel_order(self, symbol: str, order_id: str) -> dict[str, Any]:
        return await self._request("POST", "/api/v2/spot/trade/cancel-order", body={"symbol": symbol, "orderId": order_id})

    # ── futures trading ──────────────────────────────────────────────────

    async def futures_place_order(
        self, symbol: str, product_type: str, margin_mode: str, side: str,
        order_type: str, size: str, price: str = "", client_oid: str | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "symbol": symbol, "productType": product_type, "marginMode": margin_mode,
            "side": side, "orderType": order_type, "size": size,
        }
        if price:
            body["price"] = price
        return await self._request("POST", "/api/v2/mix/order/place-order", body=body, idempotency_key=client_oid)

    async def futures_cancel_order(self, symbol: str, product_type: str, order_id: str) -> dict[str, Any]:
        return await self._request("POST", "/api/v2/mix/order/cancel-order", body={
            "symbol": symbol, "productType": product_type, "orderId": order_id,
        })

    # ── position ─────────────────────────────────────────────────────────

    async def get_positions(self, product_type: str = "USDT-FUTURES", symbol: str = "") -> dict[str, Any]:
        params: dict[str, str] = {"productType": product_type}
        if symbol:
            params["symbol"] = symbol
        return await self._request("GET", "/api/v2/mix/position/all-position-v2", params=params)

    async def set_leverage(self, symbol: str, product_type: str, margin_mode: str, leverage: str) -> dict[str, Any]:
        return await self._request("POST", "/api/v2/mix/account/set-leverage", body={
            "symbol": symbol, "productType": product_type, "marginMode": margin_mode, "leverage": leverage,
        })

    # ── plan / trigger orders ────────────────────────────────────────────

    async def place_plan_order(self, symbol: str, product_type: str, margin_mode: str, side: str, size: str, trigger_price: str, order_type: str = "market", client_oid: str | None = None) -> dict[str, Any]:
        body: dict[str, Any] = {
            "symbol": symbol, "productType": product_type, "marginMode": margin_mode,
            "side": side, "size": size, "triggerPrice": trigger_price, "orderType": order_type,
        }
        return await self._request("POST", "/api/v2/mix/order/place-plan-order", body=body, idempotency_key=client_oid)

    # ── lifecycle ────────────────────────────────────────────────────────

    async def close(self) -> None:
        await self._http.aclose()
