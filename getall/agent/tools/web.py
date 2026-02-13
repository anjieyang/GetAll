"""Web tools: web_search and web_fetch.

web_search supports two providers:
- "brave"  -> Brave Search API (needs api_key / BRAVE_API_KEY)
- "openai" -> OpenAI Responses API built-in web_search (needs openai_api_key)

Auto-fallback: if provider="brave" but no Brave key is set, uses OpenAI.
"""

import html
import json
import os
import re
from typing import Any
from urllib.parse import urlparse

import httpx

from getall.agent.tools.base import Tool

# Shared constants
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_7_2) AppleWebKit/537.36"
MAX_REDIRECTS = 5


def _strip_tags(text: str) -> str:
    """Remove HTML tags and decode entities."""
    text = re.sub(r"<script[\s\S]*?</script>", "", text, flags=re.I)
    text = re.sub(r"<style[\s\S]*?</style>", "", text, flags=re.I)
    text = re.sub(r"<[^>]+>", "", text)
    return html.unescape(text).strip()


def _normalize(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _validate_url(url: str) -> tuple[bool, str]:
    try:
        p = urlparse(url)
        if p.scheme not in ("http", "https"):
            return False, f"Only http/https allowed, got '{p.scheme or 'none'}'"
        if not p.netloc:
            return False, "Missing domain"
        return True, ""
    except Exception as exc:
        return False, str(exc)


# ======================================================================
# web_search
# ======================================================================


class WebSearchTool(Tool):
    """Search the web using Brave Search or OpenAI built-in web search."""

    name = "web_search"
    description = "Search the web. Returns titles, URLs, and snippets."
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "count": {"type": "integer", "description": "Results (1-10)", "minimum": 1, "maximum": 10},
        },
        "required": ["query"],
    }

    def __init__(
        self,
        provider: str = "brave",
        api_key: str | None = None,
        max_results: int = 5,
        openai_api_key: str | None = None,
        openai_model: str = "gpt-4o-mini",
    ):
        self.provider = provider
        self.api_key = api_key or os.environ.get("BRAVE_API_KEY", "")
        self.max_results = max_results
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY", "")
        self.openai_model = openai_model

    async def execute(self, query: str, count: int | None = None, **kwargs: Any) -> str:
        if self.provider == "openai":
            return await self._search_openai(query, count)
        # Brave first; auto-fallback to OpenAI if no Brave key
        if not self.api_key and self.openai_api_key:
            return await self._search_openai(query, count)
        return await self._search_brave(query, count)

    # -- Brave --------------------------------------------------------

    async def _search_brave(self, query: str, count: int | None = None) -> str:
        if not self.api_key:
            return "Error: BRAVE_API_KEY not configured"
        try:
            n = min(max(count or self.max_results, 1), 10)
            async with httpx.AsyncClient() as client:
                r = await client.get(
                    "https://api.search.brave.com/res/v1/web/search",
                    params={"q": query, "count": n},
                    headers={"Accept": "application/json", "X-Subscription-Token": self.api_key},
                    timeout=10.0,
                )
                r.raise_for_status()
            results = r.json().get("web", {}).get("results", [])
            if not results:
                return f"No results for: {query}"
            lines = [f"Results for: {query}\n"]
            for i, item in enumerate(results[:n], 1):
                lines.append(f"{i}. {item.get('title', '')}\n   {item.get('url', '')}")
                if desc := item.get("description"):
                    lines.append(f"   {desc}")
            return "\n".join(lines)
        except Exception as exc:
            return f"Error: {exc}"

    # -- OpenAI Responses API -----------------------------------------

    async def _search_openai(self, query: str, count: int | None = None) -> str:
        """Use OpenAI Responses API with built-in web_search tool."""
        if not self.openai_api_key:
            return "Error: OpenAI API key not configured for web search"
        try:
            async with httpx.AsyncClient() as client:
                r = await client.post(
                    "https://api.openai.com/v1/responses",
                    headers={
                        "Authorization": f"Bearer {self.openai_api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.openai_model,
                        "tools": [{"type": "web_search"}],
                        "input": query,
                    },
                    timeout=30.0,
                )
                r.raise_for_status()

            data = r.json()

            # Extract output text
            output_text = ""
            if isinstance(data.get("output_text"), str):
                output_text = data["output_text"].strip()

            if not output_text:
                parts: list[str] = []
                for item in data.get("output", []) or []:
                    if not isinstance(item, dict) or item.get("type") != "message":
                        continue
                    for cp in item.get("content", []) or []:
                        if not isinstance(cp, dict):
                            continue
                        txt = cp.get("text")
                        if cp.get("type") in ("output_text", "text") and isinstance(txt, str) and txt.strip():
                            parts.append(txt.strip())
                output_text = "\n\n".join(parts).strip()

            # Extract citations
            citations: list[dict[str, str]] = []
            for item in data.get("output", []) or []:
                if not isinstance(item, dict) or item.get("type") != "message":
                    continue
                for cp in item.get("content", []) or []:
                    if not isinstance(cp, dict):
                        continue
                    for ann in cp.get("annotations", []) or []:
                        if isinstance(ann, dict) and ann.get("type") == "url_citation":
                            citations.append({"url": ann.get("url", ""), "title": ann.get("title", "")})

            if not output_text:
                return f"No results for: {query}"

            lines = [f"Results for: {query}\n", output_text]
            if citations:
                lines.append("\nSources:")
                seen: set[str] = set()
                idx = 0
                for c in citations:
                    url = c["url"]
                    if url in seen:
                        continue
                    seen.add(url)
                    idx += 1
                    lines.append(f"{idx}. {c['title'] or url}\n   {url}")
            return "\n".join(lines)
        except httpx.HTTPStatusError as exc:
            body = exc.response.text[:300] if exc.response else ""
            return f"Error: OpenAI web search failed ({exc.response.status_code}): {body}"
        except Exception as exc:
            return f"Error: {exc}"


# ======================================================================
# web_fetch
# ======================================================================


class WebFetchTool(Tool):
    """Fetch and extract content from a URL using Readability."""

    name = "web_fetch"
    description = "Fetch URL and extract readable content (HTML -> markdown/text)."
    parameters = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "URL to fetch"},
            "extractMode": {"type": "string", "enum": ["markdown", "text"], "default": "markdown"},
            "maxChars": {"type": "integer", "minimum": 100},
        },
        "required": ["url"],
    }

    def __init__(self, max_chars: int = 50_000):
        self.max_chars = max_chars

    async def execute(self, url: str, extractMode: str = "markdown", maxChars: int | None = None, **kwargs: Any) -> str:
        max_chars = maxChars or self.max_chars

        is_valid, error_msg = _validate_url(url)
        if not is_valid:
            return json.dumps({"error": f"URL validation failed: {error_msg}", "url": url})

        try:
            async with httpx.AsyncClient(
                follow_redirects=True,
                max_redirects=MAX_REDIRECTS,
                timeout=30.0,
            ) as client:
                r = await client.get(url, headers={"User-Agent": USER_AGENT})
                r.raise_for_status()

            ctype = r.headers.get("content-type", "")

            if "application/json" in ctype:
                text, extractor = json.dumps(r.json(), indent=2), "json"
            elif "text/html" in ctype or r.text[:256].lower().startswith(("<!doctype", "<html")):
                doc_html = r.text
                title = ""
                extractor = "html"
                try:
                    from readability import Document  # type: ignore[import-untyped]
                    doc = Document(r.text)
                    doc_html = doc.summary()
                    title = doc.title() or ""
                    extractor = "readability"
                except Exception:
                    pass
                content = self._to_markdown(doc_html) if extractMode == "markdown" else _strip_tags(doc_html)
                text = f"# {title}\n\n{content}" if title else content
            else:
                text, extractor = r.text, "raw"

            truncated = len(text) > max_chars
            if truncated:
                text = text[:max_chars]

            return json.dumps({
                "url": url,
                "finalUrl": str(r.url),
                "status": r.status_code,
                "extractor": extractor,
                "truncated": truncated,
                "length": len(text),
                "text": text,
            })
        except Exception as exc:
            return json.dumps({"error": str(exc), "url": url})

    def _to_markdown(self, raw_html: str) -> str:
        text = re.sub(
            r'<a\s+[^>]*href=["\']([^"\']+)["\'][^>]*>([\s\S]*?)</a>',
            lambda m: f"[{_strip_tags(m[2])}]({m[1]})",
            raw_html,
            flags=re.I,
        )
        text = re.sub(
            r"<h([1-6])[^>]*>([\s\S]*?)</h\1>",
            lambda m: f'\n{"#" * int(m[1])} {_strip_tags(m[2])}\n',
            text,
            flags=re.I,
        )
        text = re.sub(r"<li[^>]*>([\s\S]*?)</li>", lambda m: f"\n- {_strip_tags(m[1])}", text, flags=re.I)
        text = re.sub(r"</(p|div|section|article)>", "\n\n", text, flags=re.I)
        text = re.sub(r"<(br|hr)\s*/?>", "\n", text, flags=re.I)
        return _normalize(_strip_tags(text))
