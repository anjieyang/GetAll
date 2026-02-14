"""Bitget UTA REST gateway.

This client is the single Bitget integration path:
- Public market shortcuts (no credentials required)
- Private account/trade shortcuts (credentials required)
- Generic UTA endpoint caller for full API coverage
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import time
import uuid
from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse

import httpx

from getall.settings import GetAllSettings, get_settings

_FUTURES_PRODUCT_TYPES = ("USDT-FUTURES", "USDC-FUTURES", "COIN-FUTURES")


class BitgetError(RuntimeError):
    """Typed Bitget gateway error with contextual message."""


@dataclass(frozen=True, slots=True)
class BitgetCredentials:
    api_key: str
    api_secret: str
    passphrase: str


class _RateLimiter:
    """Simple token-bucket limiter per gateway instance."""

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


def _normalize_path(
    path: str,
    params: dict[str, Any] | None,
) -> tuple[str, dict[str, Any] | None]:
    """Clean path and auto-extract any mistakenly appended query string.

    Returns (clean_path, merged_params).
    """
    raw = (path or "").strip()
    if not raw:
        raise BitgetError("empty Bitget API path")

    # Auto-extract query string from path (LLM sometimes appends ?key=val)
    if "?" in raw:
        parsed = urlparse(raw)
        raw = parsed.path or raw
        if parsed.query:
            extracted = {k: v[0] for k, v in parse_qs(parsed.query).items()}
            if params is None:
                params = extracted
            else:
                merged = dict(extracted)
                merged.update(params)  # explicit params win
                params = merged

    clean = raw if raw.startswith("/") else f"/{raw}"
    return clean, params


def _encode_query(params: dict[str, Any] | None) -> str:
    if not params:
        return ""
    pairs: list[tuple[str, str]] = []
    for key, value in sorted(params.items(), key=lambda item: str(item[0])):
        if value is None:
            continue
        pairs.append((str(key), str(value)))
    return urlencode(pairs)


class BitgetGateway:
    """Async Bitget UTA REST gateway."""

    def __init__(
        self,
        credentials: BitgetCredentials | None = None,
        settings: GetAllSettings | None = None,
        rps: float = 10.0,
    ) -> None:
        s = settings or get_settings()
        self._creds = credentials or BitgetCredentials("", "", "")
        self._base = s.bitget_base_url.rstrip("/")
        self._limiter = _RateLimiter(rps)
        self._http = httpx.AsyncClient(timeout=15.0, base_url=self._base)

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        body: dict[str, Any] | None = None,
        auth_required: bool = False,
        idempotency_key: str | None = None,
    ) -> Any:
        await self._limiter.wait()
        method_upper = (method or "").upper().strip()
        if method_upper not in {"GET", "POST", "PUT", "DELETE"}:
            raise BitgetError(f"unsupported HTTP method: {method}")

        # ── Path sanitisation ────────────────────────────────────────────
        request_path, params = _normalize_path(path, params)

        # ── Auto-fix: POST/PUT/DELETE without body ─────────────────────────
        # Promote params → body if the caller put fields in the wrong slot.
        # If truly nothing provided, send empty {} so Bitget returns a real
        # error with the exact missing-field list (more useful than a local block).
        if method_upper in {"POST", "PUT", "DELETE"} and body is None:
            if params:
                body = dict(params)
                params = None
            else:
                body = {}

        # ── Auto-fix: GET with body → promote body → params ──────────────
        if method_upper == "GET" and body is not None:
            if params is None:
                params = dict(body)
            else:
                merged = dict(body)
                merged.update(params)
                params = merged
            body = None

        query = _encode_query(params)
        path_query = f"{request_path}?{query}" if query else request_path

        payload_body = dict(body) if body is not None else None
        if method_upper == "POST" and idempotency_key and payload_body is not None:
            payload_body.setdefault("clientOid", idempotency_key)
        raw_body = (
            json.dumps(payload_body, ensure_ascii=False, separators=(",", ":"))
            if payload_body is not None
            else ""
        )

        headers: dict[str, str] = {"Content-Type": "application/json", "locale": "en-US"}
        if auth_required:
            if not self._creds.api_key or not self._creds.api_secret:
                raise BitgetError(
                    f"private Bitget request requires credentials: {method_upper} {request_path}"
                )
            ts = str(int(time.time() * 1000))
            pre_hash = f"{ts}{method_upper}{path_query}{raw_body}"
            signature = base64.b64encode(
                hmac.new(
                    self._creds.api_secret.encode("utf-8"),
                    pre_hash.encode("utf-8"),
                    hashlib.sha256,
                ).digest()
            ).decode("utf-8")
            headers.update(
                {
                    "ACCESS-KEY": self._creds.api_key,
                    "ACCESS-SIGN": signature,
                    "ACCESS-TIMESTAMP": ts,
                    "ACCESS-PASSPHRASE": self._creds.passphrase,
                }
            )

        response = await self._http.request(
            method_upper,
            path_query,
            content=raw_body if raw_body else None,
            headers=headers,
        )
        if response.status_code >= 400:
            raise BitgetError(
                f"HTTP {response.status_code} for {method_upper} {request_path}: {response.text[:500]}"
            )

        try:
            data = response.json()
        except Exception as exc:
            raise BitgetError(f"invalid JSON response for {method_upper} {request_path}") from exc
        if isinstance(data, dict) and data.get("code") and str(data["code"]) != "00000":
            raise BitgetError(
                f"API error {data.get('code')} for {method_upper} {request_path}: {data.get('msg', '')}"
            )
        return data

    async def get_tickers(self, product_type: str = "USDT-FUTURES") -> Any:
        return await self._request(
            "GET",
            "/api/v2/mix/market/tickers",
            params={"productType": product_type},
            auth_required=False,
        )

    async def get_candles(
        self,
        symbol: str,
        granularity: str = "1H",
        product_type: str = "USDT-FUTURES",
        limit: int = 100,
    ) -> Any:
        return await self._request(
            "GET",
            "/api/v2/mix/market/candles",
            params={
                "symbol": symbol,
                "productType": product_type,
                "granularity": granularity,
                "limit": limit,
            },
            auth_required=False,
        )

    async def get_depth(
        self,
        symbol: str,
        product_type: str = "USDT-FUTURES",
        limit: int = 15,
    ) -> Any:
        return await self._request(
            "GET",
            "/api/v2/mix/market/depth",
            params={"symbol": symbol, "productType": product_type, "limit": limit},
            auth_required=False,
        )

    async def get_funding_rate(self, symbol: str, product_type: str = "USDT-FUTURES") -> Any:
        return await self._request(
            "GET",
            "/api/v2/mix/market/current-fund-rate",
            params={"symbol": symbol, "productType": product_type},
            auth_required=False,
        )

    async def get_account_assets(
        self,
        account_type: str = "all",
        coin: str = "",
        product_type: str = "USDT-FUTURES",
    ) -> Any:
        kind = (account_type or "all").strip().lower()
        if kind == "spot":
            params = {"coin": coin} if coin else None
            return await self._request(
                "GET",
                "/api/v2/spot/account/assets",
                params=params,
                auth_required=True,
            )
        if kind == "funding":
            return await self._request("GET", "/api/v2/account/funding-assets", auth_required=True)
        if kind == "futures":
            return await self._request(
                "GET",
                "/api/v2/mix/account/accounts",
                params={"productType": product_type},
                auth_required=True,
            )
        if kind != "all":
            raise BitgetError(f"unsupported account_type: {account_type}")

        results: dict[str, Any] = {}
        results["spot"] = await self.get_account_assets(account_type="spot", coin=coin)
        results["funding"] = await self.get_account_assets(account_type="funding")
        futures: dict[str, Any] = {}
        for pt in _FUTURES_PRODUCT_TYPES:
            futures[pt] = await self.get_account_assets(account_type="futures", product_type=pt)
        results["futures"] = futures
        return results

    async def get_account_bills(
        self,
        account_type: str = "futures",
        product_type: str = "USDT-FUTURES",
        limit: int = 20,
    ) -> Any:
        kind = (account_type or "futures").strip().lower()
        if kind != "futures":
            raise BitgetError("spot bills shortcut is not implemented; use bitget_uta for this endpoint")
        return await self._request(
            "GET",
            "/api/v2/mix/account/bill",
            params={"productType": product_type, "limit": limit},
            auth_required=True,
        )

    async def get_positions(self, product_type: str = "USDT-FUTURES", symbol: str = "") -> Any:
        params: dict[str, Any] = {"productType": product_type}
        if symbol:
            params["symbol"] = symbol
        return await self._request(
            "GET",
            "/api/v2/mix/position/all-position",
            params=params,
            auth_required=True,
        )

    async def spot_place_order_orders(self, orders: list[dict[str, Any]]) -> Any:
        if not orders:
            raise BitgetError("spot_place_order_orders requires at least one order")
        results: list[Any] = []
        for order in orders:
            payload = dict(order)
            payload.setdefault("clientOid", str(uuid.uuid4()))
            result = await self._request(
                "POST",
                "/api/v2/spot/trade/place-order",
                body=payload,
                auth_required=True,
            )
            results.append(result)
        return results[0] if len(results) == 1 else {"results": results}

    async def futures_place_order_orders(self, orders: list[dict[str, Any]]) -> Any:
        if not orders:
            raise BitgetError("futures_place_order_orders requires at least one order")
        results: list[Any] = []
        for order in orders:
            payload = dict(order)
            payload.setdefault("clientOid", str(uuid.uuid4()))
            result = await self._request(
                "POST",
                "/api/v2/mix/order/place-order",
                body=payload,
                auth_required=True,
            )
            results.append(result)
        return results[0] if len(results) == 1 else {"results": results}

    async def futures_cancel_orders(
        self,
        symbol: str,
        product_type: str,
        order_id: str = "",
        order_ids: list[str] | None = None,
        cancel_all: bool = False,
        margin_coin: str = "",
    ) -> Any:
        if cancel_all:
            raise BitgetError("cancel_all shortcut is not implemented; use bitget_uta")

        ids: list[str] = []
        if order_id:
            ids.append(order_id)
        if order_ids:
            ids.extend([oid for oid in order_ids if oid])
        if not ids:
            raise BitgetError("futures_cancel_orders requires order_id or order_ids")

        results: list[Any] = []
        for oid in ids:
            payload: dict[str, Any] = {
                "symbol": symbol,
                "productType": product_type,
                "orderId": oid,
            }
            if margin_coin:
                payload["marginCoin"] = margin_coin
            result = await self._request(
                "POST",
                "/api/v2/mix/order/cancel-order",
                body=payload,
                auth_required=True,
            )
            results.append(result)
        return results[0] if len(results) == 1 else {"results": results}

    async def request_uta(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        body: dict[str, Any] | None = None,
        auth_required: bool = True,
    ) -> Any:
        return await self._request(
            method=method,
            path=path,
            params=params,
            body=body,
            auth_required=auth_required,
        )

    async def close(self) -> None:
        await self._http.aclose()
