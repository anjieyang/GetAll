"""Followin API 新闻 / KOL 观点 / 热门话题适配器.

Base: https://api.followin.io
使用 httpx async 发起请求, 支持多语言 (lang 参数).
"""

from typing import Any

import httpx
from loguru import logger


# 默认请求超时 (秒)
_TIMEOUT = 15.0


class FollowinAdapter:
    """
    Followin open API adapter.

    Fetches crypto news, KOL opinions, trending topics, and flash news.
    """

    def __init__(self, base_url: str, api_key: str = "", lang: str = "zh-Hans"):
        """
        Initialize Followin adapter.

        Args:
            base_url: API base URL (no trailing slash).
            api_key: Followin API key.
            lang: Language code, e.g. "zh-Hans", "en".
        """
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._lang = lang

    # ──────────────────────── 底层请求 ────────────────────────

    async def _request(
        self, endpoint: str, params: dict[str, Any] | None = None
    ) -> Any:
        """
        Send GET request to Followin API.

        Args:
            endpoint: API endpoint path.
            params: Query parameters.

        Returns:
            Parsed data payload or error dict.
        """
        url = f"{self._base_url}/{endpoint.lstrip('/')}"
        query = dict(params or {})
        # 注入公共参数
        if self._api_key:
            query["apikey"] = self._api_key
        query.setdefault("lang", self._lang)

        headers = {"accept": "application/json"}

        try:
            async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
                resp = await client.get(url, params=query, headers=headers)
                # Followin 会在非 2xx 时依然返回 JSON 错误体（含 msg/code），
                # 不直接 raise_for_status，优先解析 msg 以便上层可读提示。
                try:
                    data = resp.json()
                except ValueError:
                    data = None

                if resp.status_code >= 400:
                    if isinstance(data, dict):
                        msg = data.get("msg") or data.get("message") or f"HTTP {resp.status_code}"
                        code = data.get("code")
                        logger.error(f"Followin HTTP {resp.status_code}: {endpoint} ({msg})")
                        return {
                            "error": msg,
                            "endpoint": endpoint,
                            "status": resp.status_code,
                            "code": code,
                        }
                    logger.error(f"Followin HTTP {resp.status_code}: {endpoint}")
                    return {"error": f"HTTP {resp.status_code}", "endpoint": endpoint, "status": resp.status_code}

                if data is None:
                    return {"error": "Empty response", "endpoint": endpoint, "status": resp.status_code}
                # Followin 标准响应: {"code": 2000, "msg": "success", "data": ...}
                # 注意: Followin 的成功码是 2000, 不是 0
                if not isinstance(data, dict):
                    return data

                code = data.get("code")
                msg = (data.get("msg") or data.get("message") or "").strip()
                payload = data.get("data") if "data" in data else data

                # 兼容不同 success code（部分端点可能返回 200 / "200"）
                success_codes = {0, "0", 2000, "2000", 200, "200", None}
                if code in success_codes:
                    return payload

                # 容错：如果 payload 非空且 msg 看起来是成功提示，则视为成功
                if payload not in (None, "", []) and msg and "success" in msg.lower():
                    return payload

                logger.warning(f"Followin API error [{endpoint}]: code={code}, {msg or 'Unknown error'}")
                return {"error": msg or "Unknown error", "endpoint": endpoint, "code": code}
        except httpx.HTTPStatusError as e:
            logger.error(f"Followin HTTP {e.response.status_code}: {endpoint}")
            return {"error": f"HTTP {e.response.status_code}", "endpoint": endpoint}
        except httpx.RequestError as e:
            logger.error(f"Followin request failed [{endpoint}]: {e}")
            return {"error": str(e), "endpoint": endpoint}

    # ──────────────────────── 新闻接口 ────────────────────────

    async def get_trending_news(
        self, type: str = "hot_news", count: int = 20
    ) -> Any:
        """
        Get trending news feed.

        Args:
            type: Feed type, e.g. "hot_news".
            count: Number of items.

        Returns:
            List of news items or error dict.
        """
        return await self._request(
            "open/feed/list/trending",
            {"type": type, "count": count},
        )

    async def get_kol_opinions(self, symbol: str, count: int = 20) -> Any:
        """
        Get KOL opinions for a specific coin/tag.

        Args:
            symbol: Coin symbol, e.g. "BTC".
            count: Number of items.

        Returns:
            List of KOL opinion items or error dict.
        """
        return await self._request(
            "open/feed/list/tag/opinions",
            {"symbol": symbol, "count": count},
        )

    async def get_trending_topics(self, count: int = 10) -> Any:
        """
        Get trending topic rankings.

        Args:
            count: Number of topics.

        Returns:
            List of trending topics or error dict.
        """
        return await self._request(
            "open/trending_topic/ranks",
            {"count": count},
        )

    async def get_flash_news(self, count: int = 20) -> Any:
        """
        Get flash / breaking news.

        Args:
            count: Number of flash news items.

        Returns:
            List of flash news or error dict.
        """
        return await self._request(
            "open/feed/news",
            {"count": count},
        )

    async def get_coin_news(
        self, symbol: str, type: str = "news", count: int = 20
    ) -> Any:
        """
        Get news/discussions for a specific coin.

        Args:
            symbol: Coin symbol tag, e.g. "BTC", "ETH".
            type: Content type - currently only "news" is supported.
            count: Number of items.

        Returns:
            List of coin-related news or error dict.
        """
        return await self._request(
            "open/feed/list/tag",
            {"symbol": symbol, "type": type, "count": count},
        )

    # ──────────────────────── 过滤工具 ────────────────────────

    @staticmethod
    def filter_by_symbols(
        news_list: list[dict[str, Any]],
        watched_symbols: list[str],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """
        Filter news by watched symbol tags.

        Splits a list of news items into matched (containing any watched symbol
        in their tags) and unmatched.

        Args:
            news_list: Raw news items (each may have a "tags" list).
            watched_symbols: Symbols to match, e.g. ["BTC", "ETH", "SOL"].

        Returns:
            Tuple of (matched, unmatched) lists.
        """
        # 归一化 watched symbols: 去除 /USDT 等后缀, 转大写
        watch_set = set()
        for s in watched_symbols:
            base = s.split("/")[0].upper()
            watch_set.add(base)

        matched: list[dict[str, Any]] = []
        unmatched: list[dict[str, Any]] = []

        for item in news_list:
            tags = item.get("tags") or []
            # tags 可能是 [{"symbol": "BTC"}, ...] 或 ["BTC", ...]
            item_symbols = set()
            for tag in tags:
                if isinstance(tag, dict):
                    sym = tag.get("symbol") or tag.get("name") or ""
                    item_symbols.add(sym.upper())
                elif isinstance(tag, str):
                    item_symbols.add(tag.upper())

            if item_symbols & watch_set:
                matched.append(item)
            else:
                unmatched.append(item)

        return matched, unmatched
