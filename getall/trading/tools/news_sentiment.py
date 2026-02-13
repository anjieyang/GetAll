"""新闻舆情工具 - 热门新闻、KOL 观点、趋势话题、币种新闻、突发消息 + 批量币种舆情"""

import asyncio
from collections import deque
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

from getall.agent.tools.base import Tool
from getall.trading.data.hub import DataHub
from getall.trading.news_cache import (
    iter_history_snapshots,
    load_last_non_empty_snapshot,
    load_latest_snapshot,
    persist_news_snapshot,
    pick_best_snapshot,
    snapshot_has_content,
    snapshot_ts,
)


class NewsSentimentTool(Tool):
    """从 Followin API 和 bwenews 缓存获取加密货币新闻与舆情数据。
    支持 batch_coin_sentiment 批量操作，一次获取多个币种的新闻+KOL 观点。
    """

    def __init__(self, hub: DataHub, workspace_path: Path | None = None):
        # 注入 DataHub，通过 followin 适配器获取新闻数据
        self.hub = hub
        self.workspace_path = workspace_path
        # latest_news 增量去重缓存（进程内；避免重复播报同一条新闻）
        self._latest_seen: deque[str] = deque()
        self._latest_seen_set: set[str] = set()
        self._latest_seen_max = 500

    @property
    def name(self) -> str:
        return "news_sentiment"

    @property
    def description(self) -> str:
        return (
            "Get crypto news, KOL opinions, trending topics (今日最新热点), and breaking "
            "news from Followin API and bwenews cache.\n"
            "Supports batch_coin_sentiment: get news + KOL opinions for "
            "multiple coins in one call (ideal for cron news-feed tasks).\n"
            "Use action='latest_news' for continuously updating important/latest news (最新新闻, 增量去重).\n"
            "Use action='trending_topics' (or 'hot_events') to get today's hottest topics (热门事件/风向标)."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "The type of news/sentiment data to fetch",
                    "enum": [
                        "batch_coin_sentiment",  # 批量: 多币种新闻+KOL
                        "recent",           # 从本地缓存返回最近新闻（可按窗口筛选，空结果自动回退）
                        "latest_news",      # 最新新闻（增量，优先 important）
                        "trending_news",    # 热门新闻
                        "kol_opinions",     # KOL 观点
                        "hot_events",       # 热门事件（别名：trending_topics）
                        "trending_topics",  # 热门话题（今日最新热点/风向标）
                        "coin_news",        # 币种专属新闻
                        "breaking_news",    # 突发消息
                    ],
                },
                "symbol": {
                    "type": "string",
                    "description": (
                        "Coin symbol, e.g. 'BTC'. Required for kol_opinions and coin_news; "
                        "optional for latest_news (filters to that coin)."
                    ),
                },
                "symbols": {
                    "type": "string",
                    "description": (
                        "Comma-separated coin symbols for batch_coin_sentiment, e.g. 'BTC,ETH,SOL'. "
                        "Fetches KOL opinions + coin news for each symbol."
                    ),
                },
                "count": {
                    "type": "integer",
                    "description": "Number of results per category (default: 5 for batch, 10 for single)",
                    "minimum": 1,
                    "maximum": 50,
                },
                "window_minutes": {
                    "type": "integer",
                    "description": "For action='recent': time window in minutes to search cache history (default: 360).",
                    "minimum": 1,
                    "maximum": 43200,
                },
                "mode": {
                    "type": "string",
                    "description": (
                        "For action='recent': "
                        "'auto' uses cache if fresh else fetches live; "
                        "'cache' never fetches live; "
                        "'live' always fetches live then caches."
                    ),
                    "enum": ["auto", "cache", "live"],
                },
                "prefer_non_empty": {
                    "type": "boolean",
                    "description": "For action='recent': if true, avoid returning empty cache snapshots (default: true).",
                },
            },
            "required": ["action"],
        }

    async def execute(self, **kwargs: Any) -> str:
        action: str = kwargs["action"]
        symbol: str | None = kwargs.get("symbol")
        symbols_str: str | None = kwargs.get("symbols")
        count: int = kwargs.get("count", 10)
        window_minutes: int = kwargs.get("window_minutes", 360)
        mode: str = kwargs.get("mode", "auto")
        prefer_non_empty: bool = kwargs.get("prefer_non_empty", True)

        try:
            # ── 批量操作 ──
            if action == "batch_coin_sentiment":
                return await self._batch_coin_sentiment(
                    symbols_str=symbols_str or "",
                    count=kwargs.get("count", 5),  # 批量默认每个币 5 条
                )
            if action == "recent":
                return await self._recent(
                    symbols_str=symbols_str or "",
                    count=kwargs.get("count", 5),
                    window_minutes=window_minutes,
                    mode=mode,
                    prefer_non_empty=prefer_non_empty,
                )

            handlers = {
                "latest_news": self._latest_news,
                "trending_news": self._trending_news,
                "kol_opinions": self._kol_opinions,
                "hot_events": self._trending_topics,  # alias
                "trending_topics": self._trending_topics,
                "coin_news": self._coin_news,
                "breaking_news": self._breaking_news,
            }
            handler = handlers.get(action)
            if handler is None:
                return f"Error: unknown action '{action}'"
            return await handler(symbol=symbol, count=count)
        except Exception as e:
            return f"Error fetching {action}: {e}"

    # ──────────────────────────────────────────────
    # 通用解析：兼容 Followin 多种返回结构
    # ──────────────────────────────────────────────

    @staticmethod
    def _extract_list_payload(data: Any) -> tuple[list[dict[str, Any]], str | None]:
        """
        Normalize Followin payload to a list of dict items.

        Followin endpoints may return:
        - list[dict]
        - {"list": [...]}
        - {"items": [...]}
        - {"data": {"list": [...]}} (depending on gateway/proxy)
        - {"error": "..."} (our adapter error shape)
        """
        if data is None:
            return [], None

        if isinstance(data, dict) and data.get("error"):
            return [], str(data.get("error"))

        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict)], None

        if isinstance(data, dict):
            for key in ("list", "items", "result", "rows"):
                val = data.get(key)
                if isinstance(val, list):
                    return [x for x in val if isinstance(x, dict)], None

            # 兼容一层嵌套
            for key in ("data", "payload"):
                nested = data.get(key)
                if isinstance(nested, dict):
                    for k2 in ("list", "items", "result", "rows"):
                        val = nested.get(k2)
                        if isinstance(val, list):
                            return [x for x in val if isinstance(x, dict)], None

        return [], None

    @staticmethod
    def _news_uid(item: dict[str, Any]) -> str:
        """Best-effort stable uid for deduplication."""
        for k in ("id", "newsId", "feedId", "uuid"):
            v = item.get(k)
            if v:
                return f"{k}:{v}"
        url = item.get("url") or item.get("page_url") or item.get("pageUrl") or item.get("link")
        if url:
            return f"url:{url}"
        title = item.get("title") or item.get("headline") or ""
        ts = (
            item.get("publishedAt")
            or item.get("timestamp")
            or item.get("publish_time")
            or item.get("publishTime")
            or item.get("published_at")
            or ""
        )
        src = item.get("source") or item.get("source_name") or item.get("sourceName") or ""
        return f"fallback:{src}|{ts}|{title}"

    @staticmethod
    def _looks_important(item: dict[str, Any]) -> bool:
        """
        Determine whether a news item is important.

        Supports common fields:
        - important: bool/int/str
        - importance: "high"/"medium"/"low"
        - level: same as above
        - type/category: may contain "important"
        """
        if "important" in item:
            v = item.get("important")
            if isinstance(v, bool):
                return v
            if isinstance(v, (int, float)):
                return v > 0
            if isinstance(v, str):
                return v.strip().lower() in ("1", "true", "yes", "y", "important", "high")

        for k in ("importance", "level", "priority"):
            v = item.get(k)
            if isinstance(v, str):
                return v.strip().lower() in ("high", "important", "urgent")
            if isinstance(v, (int, float)):
                return v >= 2

        for k in ("type", "category"):
            v = item.get(k)
            if isinstance(v, str) and "important" in v.lower():
                return True

        return False

    # ══════════════════════════════════════════════
    # 批量币种舆情
    # ══════════════════════════════════════════════

    async def _batch_coin_sentiment(self, symbols_str: str, count: int) -> str:
        """批量获取多币种的 KOL 观点 + 币种新闻 (cron:news-feed 专用).

        同时抓取全市场 trending news 和 trending topics (不按币种过滤),
        再按每个币种并发获取 KOL opinions,
        一次 tool call 返回完整舆情快照.
        """
        symbols = [s.strip().upper() for s in symbols_str.split(",") if s.strip()]
        if not symbols:
            return "Error: 'symbols' is required for batch_coin_sentiment, e.g. 'BTC,ETH,SOL'"

        sem = asyncio.Semaphore(4)

        # 1. 全市场新闻 + 热门话题 (与币种查询并发执行)
        trending_coro = self.hub.followin.get_trending_news(count=count * 2)
        topics_coro = self.hub.followin.get_trending_topics(count=5)

        # 2. 每个币种的 KOL 观点
        kol_results: dict[str, Any] = {}

        async def _fetch_kol(sym: str) -> None:
            async with sem:
                try:
                    kol_results[sym] = await self.hub.followin.get_kol_opinions(
                        symbol=sym, count=count
                    )
                except Exception as e:
                    kol_results[sym] = {"error": str(e)}

        # 并发执行全部
        all_tasks = [_fetch_kol(s) for s in symbols]
        trending_task = asyncio.ensure_future(trending_coro)
        topics_task = asyncio.ensure_future(topics_coro)
        await asyncio.gather(*all_tasks, trending_task, topics_task, return_exceptions=True)

        def _task_result(task: asyncio.Future, default: Any) -> Any:
            if task.cancelled():
                return default
            try:
                return task.result()
            except Exception:
                return default

        trending_data = _task_result(trending_task, [])
        topics_data = _task_result(topics_task, [])

        # ── Persist cache snapshot (code-level guarantee) ──
        snapshot = self._build_batch_snapshot(
            symbols=symbols,
            count=count,
            trending_data=trending_data,
            topics_data=topics_data,
            kol_results=kol_results,
        )
        self._persist_snapshot(snapshot)

        # 格式化输出
        lines = [f"📰 Batch Sentiment Report ({len(symbols)} coins)"]
        lines.append("═" * 60)

        # 全市场热门新闻
        lines.append("\n🔥 Trending News:")
        normalized = snapshot.get("data") if isinstance(snapshot, dict) else {}
        trending_news = (
            normalized.get("trending_news") if isinstance(normalized, dict) else None
        )
        topics_flat = (
            normalized.get("trending_topics") if isinstance(normalized, dict) else None
        )
        kol_norm = normalized.get("kol_opinions") if isinstance(normalized, dict) else None
        errors = snapshot.get("errors") if isinstance(snapshot, dict) else None

        if isinstance(trending_news, list) and trending_news:
            for i, item in enumerate(trending_news[:8], 1):
                if not isinstance(item, dict):
                    lines.append(f"  {i}. {str(item)[:100]}")
                    continue
                title = (
                    item.get("title")
                    or item.get("translated_title")
                    or item.get("headline")
                    or "Untitled"
                )
                source = (
                    item.get("source")
                    or item.get("source_name")
                    or item.get("sourceName")
                    or ""
                )
                ts = self._format_time(
                    item.get("publishedAt")
                    or item.get("timestamp")
                    or item.get("publish_time")
                    or item.get("publishTime")
                    or item.get("published_at")
                )
                lines.append(f"  {i}. {str(title)[:100]}")
                if source or ts:
                    lines.append(f"     [{source}] {ts}")
        else:
            err = errors.get("trending_news") if isinstance(errors, dict) else None
            if err:
                lines.append(f"  ❌ {err}")
            else:
                lines.append("  No trending news available.")

        # 热门话题
        lines.append("\n📈 Trending Topics:")
        if isinstance(topics_flat, list) and topics_flat:
            for item in topics_flat[:5]:
                topic = (
                    item.get("topic", item.get("title", "?"))
                    if isinstance(item, dict)
                    else str(item)
                )
                heat = (
                    item.get("heat", item.get("score", ""))
                    if isinstance(item, dict)
                    else ""
                )
                lines.append(f"  • {topic}" + (f" (heat: {heat})" if heat else ""))
        else:
            err = errors.get("trending_topics") if isinstance(errors, dict) else None
            if err:
                lines.append(f"  ❌ {err}")
            else:
                lines.append("  No trending topics available.")

        # 每个币种的 KOL 观点
        for sym in symbols:
            lines.append(f"\n── {sym} KOL Opinions ──")
            kol_data = kol_norm.get(sym) if isinstance(kol_norm, dict) else None
            kol_errors = errors.get("kol_opinions") if isinstance(errors, dict) else None
            if isinstance(kol_errors, dict) and sym in kol_errors:
                lines.append(f"  ❌ {kol_errors[sym]}")
            elif isinstance(kol_data, list) and kol_data:
                bullish = 0
                bearish = 0
                for item in kol_data[:count]:
                    author = item.get("author", item.get("kolName", "?"))
                    sentiment = item.get("sentiment", "neutral")
                    content = item.get("content", item.get("text", ""))
                    truncated = content[:100] + "..." if len(content) > 100 else content

                    if sentiment in ("bullish", "positive", "long"):
                        bullish += 1
                        tag = "🟢"
                    elif sentiment in ("bearish", "negative", "short"):
                        bearish += 1
                        tag = "🔴"
                    else:
                        tag = "⚪"

                    lines.append(f"  {tag} @{author}: {truncated}")
                total = bullish + bearish + (len(kol_data[:count]) - bullish - bearish)
                lines.append(f"  Summary: 🟢{bullish} 🔴{bearish} / {total} total")
            else:
                lines.append("  No KOL opinions found.")

        lines.append(f"\n{'═' * 60}")
        lines.append(f"Coins: {', '.join(symbols)} | Per-coin limit: {count}")
        return "\n".join(lines)

    # ──────────────────────────────────────────────
    # Cache snapshot helpers (batch → latest/history/last_non_empty)
    # ──────────────────────────────────────────────

    def _build_batch_snapshot(
        self,
        *,
        symbols: list[str],
        count: int,
        trending_data: Any,
        topics_data: Any,
        kol_results: dict[str, Any],
    ) -> dict[str, Any]:
        ts = datetime.now(timezone.utc).isoformat()

        trending_news, trending_err = self._extract_list_payload(trending_data)

        # Normalize trending topics to a flat list of topic dicts.
        topics_err: str | None = None
        topics: list[dict[str, Any]] = []
        if isinstance(topics_data, dict) and topics_data.get("error"):
            topics_err = str(topics_data.get("error"))
        elif isinstance(topics_data, dict) and "list" in topics_data:
            for day_group in topics_data.get("list", []) or []:
                if isinstance(day_group, dict):
                    day_topics = day_group.get("topics") or []
                    if isinstance(day_topics, list):
                        topics.extend([x for x in day_topics if isinstance(x, dict)])
        elif isinstance(topics_data, list):
            topics = [x for x in topics_data if isinstance(x, dict)]
        else:
            topics, topics_err = self._extract_list_payload(topics_data)

        kol_opinions: dict[str, list[dict[str, Any]]] = {}
        kol_errors: dict[str, str] = {}
        for sym in symbols:
            lst, err = self._extract_list_payload(kol_results.get(sym))
            kol_opinions[sym] = lst
            if err:
                kol_errors[sym] = err

        snapshot: dict[str, Any] = {
            "ts": ts,
            "source": "news_sentiment.batch_coin_sentiment",
            "symbols": list(symbols),
            "count": count,
            "data": {
                "trending_news": trending_news,
                "trending_topics": topics,
                "kol_opinions": kol_opinions,
            },
        }

        errors: dict[str, Any] = {}
        if trending_err:
            errors["trending_news"] = trending_err
        if topics_err:
            errors["trending_topics"] = topics_err
        if kol_errors:
            errors["kol_opinions"] = kol_errors
        if errors:
            snapshot["errors"] = errors

        snapshot["has_content"] = snapshot_has_content(snapshot)
        return snapshot

    def _persist_snapshot(self, snapshot: dict[str, Any]) -> None:
        if not self.workspace_path:
            return
        try:
            persist_news_snapshot(self.workspace_path, snapshot)
        except Exception:
            # Never break tool output due to cache persistence failures.
            return

    # ──────────────────────────────────────────────
    # Recent (cache-first) query
    # ──────────────────────────────────────────────

    async def _recent(
        self,
        *,
        symbols_str: str,
        count: int,
        window_minutes: int,
        mode: str,
        prefer_non_empty: bool,
    ) -> str:
        symbols = [s.strip().upper() for s in symbols_str.split(",") if s.strip()]
        if not symbols:
            return "Error: 'symbols' is required for recent, e.g. 'BTC,ETH,SOL'"
        if not self.workspace_path:
            return "Error: news cache is not configured (workspace_path missing)."

        now = datetime.now(timezone.utc)
        since = now - timedelta(minutes=window_minutes)

        mode_norm = (mode or "auto").strip().lower()
        if mode_norm not in ("auto", "cache", "live"):
            mode_norm = "auto"

        # ── Decide whether to fetch live ──
        cache_ttl_minutes = 20  # cron default is 15m; keep a small buffer
        latest = load_latest_snapshot(self.workspace_path)
        latest_dt = snapshot_ts(latest) if isinstance(latest, dict) else None
        is_fresh = bool(latest_dt and (now - latest_dt).total_seconds() <= cache_ttl_minutes * 60)

        # If the latest snapshot does not cover requested symbols, treat as stale for auto-mode.
        if is_fresh and isinstance(latest, dict):
            cached_syms = set(latest.get("symbols") or [])
            if not set(symbols).issubset(cached_syms):
                is_fresh = False

        if mode_norm == "live" or (mode_norm == "auto" and not is_fresh):
            # Fetch fresh snapshot and persist (batch handler already persists)
            await self._batch_coin_sentiment(symbols_str=",".join(symbols), count=count)

        # ── Pick best snapshot from history within the window ──
        snapshots = list(iter_history_snapshots(self.workspace_path, since=since))

        wanted = set(symbols)
        covering = [
            s for s in snapshots
            if isinstance(s, dict) and wanted.issubset(set(s.get("symbols") or []))
        ]
        if covering:
            snapshots = covering

        picked = pick_best_snapshot(snapshots, prefer_non_empty=prefer_non_empty)
        fallback_note = ""

        if picked is None or (prefer_non_empty and not snapshot_has_content(picked)):
            last_good = load_last_non_empty_snapshot(self.workspace_path)
            if isinstance(last_good, dict):
                picked = last_good
                dt = snapshot_ts(picked)
                ts_str = dt.isoformat() if dt else (picked.get("ts") or "?")
                fallback_note = (
                    f"\n[Note] No non-empty cache snapshot in last {window_minutes}m; "
                    f"showing last non-empty snapshot @ {ts_str}."
                )

        if not isinstance(picked, dict):
            return "📰 No cached news snapshots available yet."

        picked_dt = snapshot_ts(picked)
        picked_ts = picked_dt.isoformat() if picked_dt else (picked.get("ts") or "unknown")
        data = picked.get("data") if isinstance(picked.get("data"), dict) else {}
        trending_news = data.get("trending_news") if isinstance(data.get("trending_news"), list) else []
        trending_topics = data.get("trending_topics") if isinstance(data.get("trending_topics"), list) else []
        kol = data.get("kol_opinions") if isinstance(data.get("kol_opinions"), dict) else {}

        lines: list[str] = []
        lines.append(f"📰 Recent News (cache window: {window_minutes}m)")
        lines.append(f"Snapshot: {picked_ts}")
        lines.append("═" * 60)

        lines.append("\n🔥 Trending News:")
        if trending_news:
            for i, item in enumerate(trending_news[:8], 1):
                if not isinstance(item, dict):
                    lines.append(f"  {i}. {str(item)[:120]}")
                    continue
                title = (
                    item.get("title")
                    or item.get("translated_title")
                    or item.get("headline")
                    or "Untitled"
                )
                source = item.get("source") or item.get("source_name") or item.get("sourceName") or ""
                tss = self._format_time(
                    item.get("publishedAt")
                    or item.get("timestamp")
                    or item.get("publish_time")
                    or item.get("publishTime")
                    or item.get("published_at")
                )
                lines.append(f"  {i}. {str(title)[:100]}")
                if source or tss:
                    lines.append(f"     [{source}] {tss}")
        else:
            lines.append("  No trending news available.")

        lines.append("\n📈 Trending Topics:")
        if trending_topics:
            for item in trending_topics[:5]:
                if not isinstance(item, dict):
                    lines.append(f"  • {str(item)[:120]}")
                    continue
                topic = item.get("topic") or item.get("title") or "?"
                heat = item.get("heat") or item.get("score") or ""
                lines.append(f"  • {topic}" + (f" (heat: {heat})" if heat else ""))
        else:
            lines.append("  No trending topics available.")

        for sym in symbols:
            lines.append(f"\n── {sym} KOL Opinions ──")
            k = kol.get(sym)
            if isinstance(k, list) and k:
                for item in k[:count]:
                    if not isinstance(item, dict):
                        lines.append(f"  • {str(item)[:120]}")
                        continue
                    author = item.get("author", item.get("kolName", "?"))
                    sentiment = item.get("sentiment", "neutral")
                    content = item.get("content", item.get("text", ""))
                    truncated = content[:100] + "..." if len(content) > 100 else content
                    if sentiment in ("bullish", "positive", "long"):
                        tag = "🟢"
                    elif sentiment in ("bearish", "negative", "short"):
                        tag = "🔴"
                    else:
                        tag = "⚪"
                    lines.append(f"  {tag} @{author}: {truncated}")
            else:
                lines.append("  No KOL opinions found.")

        if fallback_note:
            lines.append(fallback_note)

        return "\n".join(lines)

    # ──────────────────────────────────────────────
    # 最新新闻（增量）
    # ──────────────────────────────────────────────

    async def _latest_news(self, symbol: str | None, count: int, **_) -> str:
        """
        获取“最新新闻”：支持增量去重，并在返回存在 important 字段时优先筛选重要新闻。

        业务语义：
        - latest_news：面向“新闻流”，随时间持续更新；适合频繁查询
        - trending_topics/hot_events：面向“热点事件/话题”，更新频率较低（一天 1~2 条）
        """
        fetch_count = min(max(count * 3, count), 50)
        scope = "GLOBAL"
        if symbol:
            clean_symbol = self._clean_symbol(symbol)
            scope = clean_symbol
            raw = await self.hub.followin.get_coin_news(symbol=clean_symbol, count=fetch_count)
        else:
            raw = await self.hub.followin.get_trending_news(count=fetch_count)
        data, err = self._extract_list_payload(raw)
        if err:
            return f"❌ Error fetching latest news: {err}"
        if not data:
            return "🆕 No latest news available at the moment."

        # 尽量按时间倒序展示
        try:
            data.sort(
                key=lambda x: (
                    x.get("publishedAt")
                    or x.get("timestamp")
                    or x.get("publish_time")
                    or x.get("publishTime")
                    or x.get("published_at")
                    or 0
                ),
                reverse=True,
            )
        except Exception:
            pass

        def _scoped_uid(item: dict[str, Any]) -> str:
            return f"{scope}:{self._news_uid(item)}"

        # 如果返回里存在 important/importance 字段 → 只显示“重要新闻”
        has_importance_field = any(
            any(k in item for k in ("important", "importance", "level", "priority"))
            for item in data
        )
        candidates = [x for x in data if self._looks_important(x)] if has_importance_field else data

        # 计算“本次新增”（在更新去重集合之前）
        items_to_show = candidates[:count]
        new_uids: set[str] = set()
        for x in items_to_show:
            uid = _scoped_uid(x)
            if uid not in self._latest_seen_set:
                new_uids.add(uid)

        # 记住本次拉到的所有新闻（不论是否 important），避免下一次重复统计
        for item in data:
            uid = _scoped_uid(item)
            if uid in self._latest_seen_set:
                continue
            self._latest_seen.append(uid)
            self._latest_seen_set.add(uid)
            while len(self._latest_seen) > self._latest_seen_max:
                old = self._latest_seen.popleft()
                self._latest_seen_set.discard(old)

        if not items_to_show:
            if has_importance_field:
                return "🆕 No important news available at the moment."
            return "🆕 No latest news available at the moment."

        title = "🆕 最新新闻（增量）" if scope == "GLOBAL" else f"🆕 {scope} 最新新闻（增量）"
        if has_importance_field:
            title += "｜仅重要"
        lines = [title]
        lines.append("═" * 60)
        lines.append(
            f"本次新增 {len(new_uids)} 条｜显示 {len(items_to_show)} 条（拉取 {fetch_count} 条去重）\n"
        )

        for i, item in enumerate(items_to_show, 1):
            headline = (
                item.get("title")
                or item.get("translated_title")
                or item.get("headline")
                or "Untitled"
            )
            source = (
                item.get("source")
                or item.get("source_name")
                or item.get("sourceName")
                or "Unknown"
            )
            published = self._format_time(
                item.get("publishedAt")
                or item.get("timestamp")
                or item.get("publish_time")
                or item.get("publishTime")
                or item.get("published_at")
            )
            summary = (
                item.get("summary")
                or item.get("description")
                or item.get("content")
                or item.get("translated_content")
                or ""
            )
            url = item.get("url") or item.get("page_url") or item.get("pageUrl") or item.get("link") or ""

            prefix = "🆕 " if _scoped_uid(item) in new_uids else ""
            lines.append(f"{i}. {prefix}{headline}")
            lines.append(f"   Source: {source} | {published}")
            if summary:
                truncated = summary[:200] + "..." if len(summary) > 200 else summary
                lines.append(f"   {truncated}")
            if url:
                lines.append(f"   Link: {url}")
            lines.append("")

        return "\n".join(lines).rstrip()

    # ──────────────────────────────────────────────
    # 热门新闻
    # ──────────────────────────────────────────────

    async def _trending_news(self, count: int, **_) -> str:
        """获取当前热门加密货币新闻"""
        raw = await self.hub.followin.get_trending_news(count=count)
        data, err = self._extract_list_payload(raw)
        if err:
            return f"❌ Error fetching trending news: {err}"
        if not data:
            return "📰 No trending news available at the moment."

        lines = [f"📰 Trending Crypto News (top {count})"]
        lines.append("═" * 50)

        # 尽量按时间倒序展示
        try:
            data.sort(
                key=lambda x: (
                    x.get("publishedAt")
                    or x.get("timestamp")
                    or x.get("publish_time")
                    or x.get("publishTime")
                    or x.get("published_at")
                    or 0
                ),
                reverse=True,
            )
        except Exception:
            pass

        for i, item in enumerate(data[:count], 1):
            title = (
                item.get("title")
                or item.get("translated_title")
                or item.get("headline")
                or "Untitled"
            )
            source = (
                item.get("source")
                or item.get("source_name")
                or item.get("sourceName")
                or "Unknown"
            )
            published = self._format_time(
                item.get("publishedAt")
                or item.get("timestamp")
                or item.get("publish_time")
                or item.get("publishTime")
                or item.get("published_at")
            )
            summary = (
                item.get("summary")
                or item.get("description")
                or item.get("content")
                or item.get("translated_content")
                or ""
            )
            url = item.get("url") or item.get("page_url") or item.get("pageUrl") or item.get("link") or ""

            lines.append(f"\n{i}. {title}")
            lines.append(f"   Source: {source} | {published}")
            if summary:
                # 截断过长的摘要
                truncated = summary[:200] + "..." if len(summary) > 200 else summary
                lines.append(f"   {truncated}")
            if url:
                lines.append(f"   Link: {url}")

        return "\n".join(lines)

    # ──────────────────────────────────────────────
    # KOL 观点
    # ──────────────────────────────────────────────

    async def _kol_opinions(self, symbol: str | None, count: int, **_) -> str:
        """获取 KOL 对特定币种的观点"""
        if not symbol:
            return "Error: 'symbol' is required for kol_opinions (e.g. 'BTC')"

        clean_symbol = self._clean_symbol(symbol)
        raw = await self.hub.followin.get_kol_opinions(symbol=clean_symbol, count=count)
        data, err = self._extract_list_payload(raw)
        if err:
            return f"❌ Error fetching KOL opinions: {err}"
        if not data:
            return f"🎤 No KOL opinions found for {clean_symbol}."

        lines = [f"🎤 KOL Opinions on {clean_symbol} ({len(data)} results)"]
        lines.append("═" * 50)

        # 统计多空观点
        bullish = 0
        bearish = 0
        neutral = 0

        for i, item in enumerate(data[:count], 1):
            author = item.get("author") or item.get("kolName", "Anonymous")
            content = item.get("content") or item.get("text", "")
            sentiment = item.get("sentiment", "neutral")
            followers = item.get("followers", "N/A")
            published = self._format_time(
                item.get("publishedAt")
                or item.get("timestamp")
                or item.get("publish_time")
                or item.get("publishTime")
                or item.get("published_at")
            )

            # 统计观点方向
            if sentiment in ("bullish", "positive", "long"):
                bullish += 1
                tag = "🟢 Bullish"
            elif sentiment in ("bearish", "negative", "short"):
                bearish += 1
                tag = "🔴 Bearish"
            else:
                neutral += 1
                tag = "⚪ Neutral"

            truncated = content[:180] + "..." if len(content) > 180 else content
            lines.append(f"\n{i}. [{tag}] @{author} ({followers} followers)")
            lines.append(f"   {published}")
            lines.append(f"   {truncated}")

        # 观点汇总
        lines.append(f"\n{'─' * 50}")
        lines.append(f"  Sentiment Summary: 🟢 Bullish: {bullish} | 🔴 Bearish: {bearish} | ⚪ Neutral: {neutral}")
        total = bullish + bearish + neutral
        if total > 0:
            bull_pct = bullish / total * 100
            bear_pct = bearish / total * 100
            lines.append(f"  Bull/Bear Ratio: {bull_pct:.0f}% / {bear_pct:.0f}%")

        return "\n".join(lines)

    # ──────────────────────────────────────────────
    # 热门话题（今日最新热点）
    # ──────────────────────────────────────────────

    async def _trending_topics(self, count: int, **_) -> str:
        """获取当前加密圈热门话题（今日最新热点风向标）"""
        data = await self.hub.followin.get_trending_topics(count=count)

        if not data:
            return "🔥 No trending topics available."

        # API 返回结构: {"list": [{"topics": [...]}, ...]}
        # 提取所有话题（优先今日，然后按热度排序）
        all_topics = []
        
        if isinstance(data, dict) and "list" in data:
            # 遍历所有日期分组，优先取第一个（今日）
            for day_group in data.get("list", []):
                topics = day_group.get("topics", [])
                if topics:
                    all_topics.extend(topics)
        elif isinstance(data, list):
            # 如果直接返回列表，使用它
            all_topics = data
        else:
            # 尝试从错误响应中提取
            if isinstance(data, dict) and "error" in data:
                return f"❌ Error fetching trending topics: {data.get('error')}"
            all_topics = []

        if not all_topics:
            return "🔥 No trending topics available."

        # 按热度排序（如果 API 没有排序）
        try:
            all_topics.sort(key=lambda x: x.get("heat", 0) or 0, reverse=True)
        except Exception:
            pass

        # 限制返回数量
        topics_to_show = all_topics[:count]

        lines = [f"🔥 今日最新热点（热门风向标）"]
        lines.append("═" * 60)
        lines.append(f"共 {len(topics_to_show)} 个热门话题\n")

        for i, topic in enumerate(topics_to_show, 1):
            title = topic.get("title") or topic.get("topic", "Unknown")
            heat = topic.get("heat", 0) or 0
            source_count = topic.get("source_count", 0) or 0
            desc = topic.get("desc", "") or topic.get("description", "")
            
            # 提取标签（币种）
            tags = topic.get("tags", [])
            tag_symbols = []
            if isinstance(tags, list):
                for tag in tags:
                    if isinstance(tag, dict):
                        symbol = tag.get("symbol") or tag.get("name", "")
                        if symbol:
                            tag_symbols.append(symbol)
            
            # 提取原因（如 "KOL热议"）
            reasons = topic.get("reasons", [])
            reason_texts = []
            if isinstance(reasons, list):
                for reason in reasons:
                    if isinstance(reason, dict):
                        text = reason.get("text", "")
                        if text:
                            reason_texts.append(text)

            # 格式化热度（大数字用 K/M 表示）
            heat_str = self._format_heat(heat)
            
            lines.append(f"{i}. {title}")
            
            # 显示标签
            if tag_symbols:
                tag_str = ", ".join(tag_symbols[:3])  # 最多显示3个
                if len(tag_symbols) > 3:
                    tag_str += f" +{len(tag_symbols) - 3}"
                lines.append(f"   🏷️  {tag_str}")
            
            # 显示热度和来源数
            info_parts = [f"热度: {heat_str}"]
            if source_count > 0:
                info_parts.append(f"来源: {source_count}")
            if reason_texts:
                info_parts.append(f"原因: {', '.join(reason_texts)}")
            lines.append(f"   {' | '.join(info_parts)}")
            
            # 显示描述
            if desc:
                truncated = desc[:200] + "..." if len(desc) > 200 else desc
                lines.append(f"   {truncated}")
            
            lines.append("")  # 空行分隔

        return "\n".join(lines)
    
    @staticmethod
    def _format_heat(heat: int | float) -> str:
        """格式化热度数字（如 108172 -> 108K）"""
        try:
            heat_num = float(heat)
            if heat_num >= 1_000_000:
                return f"{heat_num / 1_000_000:.1f}M"
            elif heat_num >= 1_000:
                return f"{heat_num / 1_000:.1f}K"
            else:
                return str(int(heat_num))
        except Exception:
            return str(heat)

    # ──────────────────────────────────────────────
    # 币种新闻
    # ──────────────────────────────────────────────

    async def _coin_news(self, symbol: str | None, count: int, **_) -> str:
        """获取特定币种相关新闻"""
        if not symbol:
            return "Error: 'symbol' is required for coin_news (e.g. 'BTC')"

        clean_symbol = self._clean_symbol(symbol)
        raw = await self.hub.followin.get_coin_news(symbol=clean_symbol, count=count)
        data, err = self._extract_list_payload(raw)
        if err:
            return f"❌ Error fetching coin news: {err}"
        if not data:
            return f"📰 No recent news for {clean_symbol}."

        lines = [f"📰 News for {clean_symbol} ({len(data)} results)"]
        lines.append("═" * 50)

        for i, item in enumerate(data[:count], 1):
            title = (
                item.get("title")
                or item.get("translated_title")
                or item.get("headline")
                or "Untitled"
            )
            source = (
                item.get("source")
                or item.get("source_name")
                or item.get("sourceName")
                or "Unknown"
            )
            published = self._format_time(
                item.get("publishedAt")
                or item.get("timestamp")
                or item.get("publish_time")
                or item.get("publishTime")
                or item.get("published_at")
            )
            summary = (
                item.get("summary")
                or item.get("description")
                or item.get("content")
                or item.get("translated_content")
                or ""
            )
            impact = item.get("impact", "")
            url = item.get("url") or item.get("page_url") or item.get("pageUrl") or item.get("link") or ""

            lines.append(f"\n{i}. {title}")
            lines.append(f"   Source: {source} | {published}")
            if impact:
                lines.append(f"   Impact: {impact}")
            if summary:
                truncated = summary[:200] + "..." if len(summary) > 200 else summary
                lines.append(f"   {truncated}")
            if url:
                lines.append(f"   Link: {url}")

        return "\n".join(lines)

    # ──────────────────────────────────────────────
    # 突发消息
    # ──────────────────────────────────────────────

    async def _breaking_news(self, count: int, **_) -> str:
        """获取突发消息（bwenews 缓存 + Followin 快讯）"""
        # 突发消息来自 Followin 快讯 (flash news) + bwenews WS 缓存
        raw = await self.hub.followin.get_flash_news(count=count)
        data, err = self._extract_list_payload(raw)
        if err:
            return f"❌ Error fetching breaking news: {err}"
        if not data:
            return "⚡ No breaking news at the moment."

        lines = [f"⚡ Breaking News ({len(data)} alerts)"]
        lines.append("═" * 50)

        for i, item in enumerate(data[:count], 1):
            title = item.get("title") or item.get("content", "")
            published = self._format_time(
                item.get("publishedAt")
                or item.get("timestamp")
                or item.get("publish_time")
                or item.get("publishTime")
                or item.get("published_at")
            )
            urgency = item.get("urgency", "normal")

            # 根据紧急程度添加标记
            if urgency == "high":
                prefix = "🚨"
            elif urgency == "medium":
                prefix = "⚡"
            else:
                prefix = "📢"

            truncated = title[:250] + "..." if len(title) > 250 else title
            lines.append(f"\n{prefix} [{published}]")
            lines.append(f"   {truncated}")

        return "\n".join(lines)

    # ──────────────────────────────────────────────
    # 工具方法
    # ──────────────────────────────────────────────

    @staticmethod
    def _clean_symbol(symbol: str) -> str:
        """清理币种符号，提取基础币种"""
        return symbol.split("/")[0].split(":")[0].upper()

    @staticmethod
    def _format_time(ts: Any) -> str:
        """将时间戳格式化为可读字符串"""
        if ts is None:
            return "Unknown time"
        try:
            if isinstance(ts, (int, float)):
                # 自动检测秒 vs 毫秒时间戳
                if ts > 1e12:
                    ts = ts / 1000
                dt = datetime.fromtimestamp(ts, tz=timezone.utc)
            elif isinstance(ts, str):
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            else:
                return str(ts)

            # 计算相对时间
            now = datetime.now(timezone.utc)
            delta = now - dt
            seconds = int(delta.total_seconds())

            if seconds < 60:
                return "just now"
            elif seconds < 3600:
                return f"{seconds // 60}m ago"
            elif seconds < 86400:
                return f"{seconds // 3600}h ago"
            else:
                return dt.strftime("%Y-%m-%d %H:%M UTC")
        except Exception:
            return str(ts)
