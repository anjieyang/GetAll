#!/usr/bin/env python3.11
"""Search meme/reaction images on the web and download one locally.

No third-party dependency required.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

EXT_BY_CONTENT_TYPE: dict[str, str] = {
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/png": ".png",
    "image/gif": ".gif",
    "image/webp": ".webp",
    "image/bmp": ".bmp",
}

ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}


@dataclass(frozen=True)
class Candidate:
    url: str
    source: str
    title: str
    query: str


def _http_bytes(
    url: str,
    *,
    timeout: int = 15,
    headers: dict[str, str] | None = None,
) -> bytes:
    req_headers = {"User-Agent": USER_AGENT, "Accept-Language": "en-US,en;q=0.9"}
    if headers:
        req_headers.update(headers)
    req = urllib.request.Request(url, headers=req_headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _http_json(
    url: str,
    *,
    timeout: int = 15,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    raw = _http_bytes(url, timeout=timeout, headers=headers)
    data = json.loads(raw.decode("utf-8", errors="replace"))
    if not isinstance(data, dict):
        raise ValueError(f"JSON payload is not an object: {url}")
    return data


def _extract_ddg_vqd(page_html: str) -> str | None:
    m = re.search(r"vqd=['\"]([^'\"]+)['\"]", page_html)
    if m:
        return m.group(1)
    m = re.search(r"vqd=([0-9-]+)\&", page_html)
    return m.group(1) if m else None


def _ddg_candidates(query: str, limit: int) -> list[Candidate]:
    q = urllib.parse.quote_plus(query)
    page_url = f"https://duckduckgo.com/?q={q}&iax=images&ia=images"
    page = _http_bytes(page_url).decode("utf-8", errors="replace")
    vqd = _extract_ddg_vqd(page)
    if not vqd:
        return []

    params = urllib.parse.urlencode(
        {
            "l": "wt-wt",
            "o": "json",
            "q": query,
            "vqd": vqd,
            "f": ",,,",
            "p": "1",
        }
    )
    api_url = f"https://duckduckgo.com/i.js?{params}"
    payload = _http_json(
        api_url,
        headers={"Referer": "https://duckduckgo.com/"},
    )
    out: list[Candidate] = []
    for item in payload.get("results", [])[:limit]:
        if not isinstance(item, dict):
            continue
        image = str(item.get("image", "")).strip()
        if image.startswith("http"):
            out.append(
                Candidate(
                    url=image,
                    source="duckduckgo",
                    title=str(item.get("title", "")).strip(),
                    query=query,
                )
            )
    return out


def _wikimedia_candidates(query: str, limit: int) -> list[Candidate]:
    base = "https://commons.wikimedia.org/w/api.php"
    search_params = urllib.parse.urlencode(
        {
            "action": "query",
            "list": "search",
            "format": "json",
            "utf8": "1",
            "srnamespace": "6",
            "srlimit": str(limit),
            "srsearch": f"filetype:bitmap {query}",
        }
    )
    search_data = _http_json(f"{base}?{search_params}")
    rows = search_data.get("query", {}).get("search", [])
    titles = [str(r.get("title", "")).strip() for r in rows if isinstance(r, dict)]
    titles = [t for t in titles if t.startswith("File:")]
    if not titles:
        return []

    detail_params = urllib.parse.urlencode(
        {
            "action": "query",
            "prop": "imageinfo",
            "iiprop": "url",
            "format": "json",
            "utf8": "1",
            "titles": "|".join(titles[:limit]),
        }
    )
    details = _http_json(f"{base}?{detail_params}")
    pages = details.get("query", {}).get("pages", {})
    out: list[Candidate] = []
    if not isinstance(pages, dict):
        return out

    for p in pages.values():
        if not isinstance(p, dict):
            continue
        title = str(p.get("title", "")).strip()
        infos = p.get("imageinfo", [])
        if not isinstance(infos, list) or not infos:
            continue
        first = infos[0]
        if not isinstance(first, dict):
            continue
        url = str(first.get("url", "")).strip()
        if url.startswith("http"):
            out.append(
                Candidate(
                    url=url,
                    source="wikimedia",
                    title=title,
                    query=query,
                )
            )
    return out


def _dedup_candidates(candidates: list[Candidate]) -> list[Candidate]:
    seen: set[str] = set()
    out: list[Candidate] = []
    for c in candidates:
        key = c.url.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def _looks_like_gif(c: Candidate) -> bool:
    lower_url = c.url.lower()
    lower_title = c.title.lower()
    return ".gif" in lower_url or " gif" in lower_title


def _pick_extension(url: str, content_type: str) -> str:
    ctype = content_type.lower().split(";")[0].strip()
    if ctype in EXT_BY_CONTENT_TYPE:
        return EXT_BY_CONTENT_TYPE[ctype]
    path_ext = Path(urllib.parse.urlparse(url).path).suffix.lower()
    if path_ext in ALLOWED_EXTS:
        return ".jpg" if path_ext == ".jpeg" else path_ext
    return ".jpg"


def _download_candidate(
    candidate: Candidate,
    *,
    out_dir: Path,
    max_bytes: int,
    index: int,
) -> Path:
    req = urllib.request.Request(
        candidate.url,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "image/*,*/*;q=0.8",
            "Referer": "https://duckduckgo.com/",
        },
    )
    with urllib.request.urlopen(req, timeout=20) as resp:
        content_type = str(resp.headers.get("Content-Type", "")).strip()
        if not content_type.lower().startswith("image/"):
            raise ValueError(f"non-image content type: {content_type}")
        blob = resp.read(max_bytes + 1)
    if len(blob) > max_bytes:
        raise ValueError(f"image too large: {len(blob)} bytes")

    ext = _pick_extension(candidate.url, content_type)
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = f"meme_{int(time.time() * 1000)}_{index}{ext}"
    out_path = out_dir / filename
    out_path.write_bytes(blob)
    return out_path


def _build_queries(base: str) -> list[str]:
    seed = base.strip()
    if not seed:
        return []
    options = [seed, f"{seed} meme", f"{seed} reaction image"]
    result: list[str] = []
    seen: set[str] = set()
    for q in options:
        key = q.lower().strip()
        if key and key not in seen:
            seen.add(key)
            result.append(q)
    return result


def _search_candidates(query: str, max_candidates: int) -> list[Candidate]:
    all_candidates: list[Candidate] = []
    for q in _build_queries(query):
        try:
            all_candidates.extend(_ddg_candidates(q, max_candidates))
        except Exception:
            pass
        try:
            all_candidates.extend(_wikimedia_candidates(q, max_candidates))
        except Exception:
            pass
    return _dedup_candidates(all_candidates)


def search_and_download(
    query: str,
    *,
    out_dir: Path,
    max_candidates: int,
    max_bytes: int,
    prefer_gif: bool,
) -> dict[str, Any]:
    candidates = _search_candidates(query, max_candidates=max_candidates)
    if not candidates:
        raise RuntimeError("no image candidates found")

    if prefer_gif:
        gif_items = [c for c in candidates if _looks_like_gif(c)]
        other_items = [c for c in candidates if not _looks_like_gif(c)]
        random.shuffle(gif_items)
        random.shuffle(other_items)
        ordered = gif_items + other_items
    else:
        ordered = list(candidates)
        random.shuffle(ordered)

    errors: list[str] = []
    for idx, cand in enumerate(ordered[:max_candidates], start=1):
        try:
            path = _download_candidate(
                cand,
                out_dir=out_dir,
                max_bytes=max_bytes,
                index=idx,
            )
            return {
                "ok": True,
                "path": str(path.resolve()),
                "source": cand.source,
                "query": cand.query,
                "url": cand.url,
                "title": cand.title,
            }
        except Exception as exc:
            errors.append(f"{cand.source}:{cand.url} -> {exc}")

    raise RuntimeError("all candidates failed: " + " | ".join(errors[:3]))


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search and download one meme image.")
    parser.add_argument("--query", required=True, help="Meme/reaction search query.")
    parser.add_argument("--out-dir", default="/tmp/getall_memes", help="Output directory.")
    parser.add_argument("--max-candidates", type=int, default=24, help="Max candidate URLs to try.")
    parser.add_argument("--max-bytes", type=int, default=8_000_000, help="Max downloaded image size.")
    parser.add_argument("--prefer-gif", action="store_true", help="Prefer GIF candidates first.")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    try:
        payload = search_and_download(
            args.query,
            out_dir=Path(args.out_dir),
            max_candidates=max(5, args.max_candidates),
            max_bytes=max(256_000, args.max_bytes),
            prefer_gif=bool(args.prefer_gif),
        )
        print(json.dumps(payload, ensure_ascii=False))
        return 0
    except (RuntimeError, urllib.error.URLError, TimeoutError, ValueError) as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False))
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
