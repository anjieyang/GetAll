"""RSS crypto news aggregator — multi-source headlines from major outlets.

Free, no API key required. Parses public RSS/Atom feeds.

This is a NATIVE data capability for the agent.

Sources:
  - CoinTelegraph:    Major crypto outlet, broad coverage.
  - Decrypt:          Web3 / DeFi focused.
  - Bitcoinist:       Bitcoin-centric news.
  - Blockworks:       Institutional, research, macro analysis.
  - The Defiant:      DeFi-specific deep dives.
  - Watcher.Guru:     Breaking news, whale alerts, market-moving events.
  - CoinDesk:         Institutional / market-focused.

Actions:
  - latest:   Fetch recent headlines from all or selected sources.
  - sources:  List available RSS feeds and their focus areas.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any

import httpx
from loguru import logger

from getall.agent.tools.base import Tool

_TIMEOUT = 20.0
_MAX_ITEMS_PER_FEED = 10
_MAX_TOTAL_ITEMS = 30


@dataclass(frozen=True)
class _FeedSource:
    """RSS feed metadata."""

    key: str
    name: str
    url: str
    focus: str
    lang: str


_FEEDS: list[_FeedSource] = [
    _FeedSource(
        key="cointelegraph",
        name="CoinTelegraph",
        url="https://cointelegraph.com/rss",
        focus="Broad crypto coverage, market analysis, regulation",
        lang="en",
    ),
    _FeedSource(
        key="decrypt",
        name="Decrypt",
        url="https://decrypt.co/feed",
        focus="Web3, DeFi, NFT, gaming",
        lang="en",
    ),
    _FeedSource(
        key="bitcoinist",
        name="Bitcoinist",
        url="https://bitcoinist.com/feed/",
        focus="Bitcoin ecosystem, altcoins",
        lang="en",
    ),
    _FeedSource(
        key="blockworks",
        name="Blockworks",
        url="https://blockworks.co/feed",
        focus="Institutional, research, macro, DeFi analysis",
        lang="en",
    ),
    _FeedSource(
        key="the_defiant",
        name="The Defiant",
        url="https://thedefiant.io/feed",
        focus="DeFi protocols, yield, governance",
        lang="en",
    ),
    _FeedSource(
        key="watcherguru",
        name="Watcher.Guru",
        url="https://watcher.guru/news/feed",
        focus="Breaking news, whale alerts, market-moving events",
        lang="en",
    ),
    _FeedSource(
        key="coindesk",
        name="CoinDesk",
        url="https://www.coindesk.com/arc/outboundfeeds/rss/",
        focus="Institutional, market structure, regulation",
        lang="en",
    ),
]

_FEED_MAP: dict[str, _FeedSource] = {f.key: f for f in _FEEDS}


def _parse_feed_xml(raw: str) -> list[dict[str, str]]:
    """Lightweight RSS/Atom XML parser — no external dependency.

    Extracts <item> (RSS) or <entry> (Atom) elements and pulls out
    title, link, pubDate/published/updated, and description/summary.
    """
    import re

    items: list[dict[str, str]] = []

    # RSS <item> blocks
    item_blocks = re.findall(r"<item[\s>](.*?)</item>", raw, re.DOTALL)
    if not item_blocks:
        # Atom <entry> blocks
        item_blocks = re.findall(r"<entry[\s>](.*?)</entry>", raw, re.DOTALL)

    for block in item_blocks:
        title = _extract_tag(block, "title")
        link = _extract_link(block)
        pub_date = (
            _extract_tag(block, "pubDate")
            or _extract_tag(block, "published")
            or _extract_tag(block, "updated")
        )
        description = _extract_tag(block, "description") or _extract_tag(block, "summary")

        # Strip HTML from description
        if description:
            description = re.sub(r"<[^>]+>", "", description).strip()
            description = re.sub(r"\s+", " ", description)
            # Truncate to ~300 chars
            if len(description) > 300:
                description = description[:297] + "..."

        if title:
            items.append({
                "title": title.strip(),
                "link": link.strip() if link else "",
                "published": pub_date.strip() if pub_date else "",
                "summary": description or "",
            })

    return items


def _extract_tag(block: str, tag: str) -> str:
    """Extract text content from an XML tag, handling CDATA."""
    import re

    # Try CDATA first
    m = re.search(
        rf"<{tag}[^>]*>\s*<!\[CDATA\[(.*?)\]\]>\s*</{tag}>",
        block,
        re.DOTALL,
    )
    if m:
        return m.group(1)
    # Plain text
    m = re.search(rf"<{tag}[^>]*>(.*?)</{tag}>", block, re.DOTALL)
    return m.group(1) if m else ""


def _extract_link(block: str) -> str:
    """Extract link — RSS uses <link>text</link>, Atom uses <link href='...'/>."""
    import re

    # Atom: <link href="..." />
    m = re.search(r'<link[^>]+href=["\']([^"\']+)["\']', block)
    if m:
        return m.group(1)
    # RSS: <link>url</link>
    m = re.search(r"<link>(.*?)</link>", block, re.DOTALL)
    return m.group(1) if m else ""


async def _fetch_feed(
    client: httpx.AsyncClient,
    source: _FeedSource,
    max_items: int,
) -> dict[str, Any]:
    """Fetch and parse a single RSS feed."""
    try:
        resp = await client.get(
            source.url,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; GetAll-Bot/1.0)",
                "Accept": "application/rss+xml, application/atom+xml, application/xml, text/xml",
            },
            follow_redirects=True,
        )
        resp.raise_for_status()
        items = _parse_feed_xml(resp.text)
        return {
            "source": source.key,
            "name": source.name,
            "focus": source.focus,
            "count": len(items[:max_items]),
            "articles": items[:max_items],
        }
    except httpx.HTTPStatusError as e:
        logger.warning(f"RSS feed {source.key} HTTP {e.response.status_code}")
        return {"source": source.key, "name": source.name, "error": f"HTTP {e.response.status_code}"}
    except Exception as e:
        logger.warning(f"RSS feed {source.key} failed: {e}")
        return {"source": source.key, "name": source.name, "error": str(e)}


class RssNewsTool(Tool):
    """RSS crypto news — aggregate headlines from 7 major English outlets. No API key."""

    @property
    def name(self) -> str:
        return "rss_news"

    @property
    def description(self) -> str:
        return (
            "Crypto news aggregator via RSS from major English-language outlets "
            "(CoinTelegraph, CoinDesk, Decrypt, Bitcoinist, Blockworks, "
            "The Defiant, Watcher.Guru). No API key needed. "
            "Actions: 'latest' (recent headlines from all or selected sources, "
            "optional keyword filter), 'sources' (list available feeds). "
            "Use for: English news coverage, cross-source signal detection "
            "(same story on multiple outlets = high importance), "
            "whale/breaking alerts (Watcher.Guru), DeFi-specific news "
            "(The Defiant), institutional research (Blockworks). "
            "Complements Followin (Chinese) and Finnhub (limited sources). "
            "Returns title, link, published date, and summary for each article."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "Action to perform",
                    "enum": ["latest", "sources"],
                },
                "feeds": {
                    "type": "string",
                    "description": (
                        "For 'latest': comma-separated feed keys to fetch "
                        "(e.g. 'cointelegraph,decrypt,rekt'). "
                        "Omit or 'all' to fetch from all sources. "
                        f"Available: {', '.join(f.key for f in _FEEDS)}"
                    ),
                },
                "keyword": {
                    "type": "string",
                    "description": (
                        "For 'latest': optional keyword to filter articles "
                        "(case-insensitive match on title + summary). "
                        "E.g. 'ethereum', 'hack', 'SEC', 'solana'"
                    ),
                },
                "limit": {
                    "type": "integer",
                    "description": "For 'latest': max articles per feed. Default: 5, max: 10",
                    "minimum": 1,
                    "maximum": 10,
                },
            },
            "required": ["action"],
        }

    async def execute(self, **kwargs: Any) -> str:
        action = kwargs.get("action", "latest")
        try:
            if action == "latest":
                return await self._latest(**kwargs)
            if action == "sources":
                return self._sources()
            return json.dumps({"error": f"Unknown action '{action}'"})
        except Exception as e:
            logger.error(f"RssNews tool error: {e}")
            return json.dumps({"error": str(e)})

    def _sources(self) -> str:
        """List available RSS feeds."""
        return json.dumps({
            "action": "sources",
            "count": len(_FEEDS),
            "feeds": [
                {
                    "key": f.key,
                    "name": f.name,
                    "focus": f.focus,
                    "lang": f.lang,
                    "url": f.url,
                }
                for f in _FEEDS
            ],
        })

    async def _latest(self, **kwargs: Any) -> str:
        """Fetch recent headlines from selected feeds."""
        feeds_str = (kwargs.get("feeds") or "all").strip().lower()
        keyword = (kwargs.get("keyword") or "").strip().lower()
        limit_per_feed = min(max(kwargs.get("limit", 5) or 5, 1), _MAX_ITEMS_PER_FEED)

        # Resolve which feeds to query
        if feeds_str == "all" or not feeds_str:
            targets = list(_FEEDS)
        else:
            keys = [k.strip() for k in feeds_str.split(",") if k.strip()]
            targets = [_FEED_MAP[k] for k in keys if k in _FEED_MAP]
            if not targets:
                return json.dumps({
                    "error": f"No valid feed keys in '{feeds_str}'",
                    "available": [f.key for f in _FEEDS],
                })

        t0 = time.monotonic()

        # Fetch all feeds concurrently
        async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
            tasks = [_fetch_feed(client, src, limit_per_feed) for src in targets]
            results = await asyncio.gather(*tasks)

        elapsed_ms = int((time.monotonic() - t0) * 1000)

        # Apply keyword filter if provided
        if keyword:
            for result in results:
                if "articles" in result:
                    filtered = [
                        a
                        for a in result["articles"]
                        if keyword in a.get("title", "").lower()
                        or keyword in a.get("summary", "").lower()
                    ]
                    result["articles"] = filtered
                    result["count"] = len(filtered)

        # Cap total items
        total_articles = sum(r.get("count", 0) for r in results)
        successful = sum(1 for r in results if "articles" in r)
        failed = sum(1 for r in results if "error" in r)

        return json.dumps({
            "action": "latest",
            "feeds_queried": len(targets),
            "feeds_ok": successful,
            "feeds_failed": failed,
            "total_articles": total_articles,
            "keyword": keyword or None,
            "elapsed_ms": elapsed_ms,
            "results": results,
        })
