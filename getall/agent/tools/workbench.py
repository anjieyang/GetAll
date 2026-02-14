"""Shared workbench tool for reusable scripts and skills."""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote_plus, unquote, urlparse

import httpx

from getall.agent.tools.base import Tool

_NAME_RE = re.compile(r"^[A-Za-z0-9._-]{1,80}$")


class WorkbenchTool(Tool):
    """Create, run, and reuse shared scripts/skills in workspace."""

    def __init__(self, workspace: Path, timeout: int = 180):
        self._workspace = workspace.expanduser().resolve()
        self._scripts_dir = self._workspace / "shared" / "scripts"
        self._skills_dir = self._workspace / "skills"
        self._capabilities_dir = self._workspace / "shared" / "capabilities"
        self._registry_path = self._capabilities_dir / "registry.json"
        self._timeout = timeout

    @property
    def name(self) -> str:
        return "workbench"

    @property
    def description(self) -> str:
        return (
            "Shared automation workbench. Create/run reusable scripts, create/install skills, "
            "discover capabilities from the web, and persist shared capability registry entries."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "list_scripts",
                        "create_script",
                        "run_script",
                        "delete_script",
                        "list_skills",
                        "create_skill",
                        "list_capabilities",
                        "discover_capabilities",
                        "install_skill_from_url",
                    ],
                    "description": "Workbench action to perform",
                },
                "script_name": {
                    "type": "string",
                    "description": "Script name for script actions (e.g. count_to_ten, rebalance.py)",
                },
                "language": {
                    "type": "string",
                    "enum": ["python", "shell"],
                    "description": "Script language for create_script",
                },
                "content": {
                    "type": "string",
                    "description": "Full script or skill content for create actions",
                },
                "args": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional argument list for run_script",
                },
                "timeout_seconds": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 600,
                    "description": "Optional timeout override for run_script",
                },
                "skill_name": {
                    "type": "string",
                    "description": "Skill directory name for create_skill",
                },
                "description": {
                    "type": "string",
                    "description": "Optional description used in generated skill frontmatter",
                },
                "request": {
                    "type": "string",
                    "description": "Natural-language request for capability discovery or creation",
                },
                "source_url": {
                    "type": "string",
                    "description": "Source URL for install_skill_from_url (raw SKILL.md or docs page)",
                },
                "risk_level": {
                    "type": "string",
                    "enum": ["readonly", "privileged"],
                    "description": "Capability risk label for registry/audit",
                },
                "confirmed": {
                    "type": "boolean",
                    "description": "Set true only after explicit user confirmation for privileged installs",
                },
            },
            "required": ["action"],
        }

    async def execute(
        self,
        action: str,
        script_name: str = "",
        language: str = "python",
        content: str = "",
        args: list[str] | None = None,
        timeout_seconds: int | None = None,
        skill_name: str = "",
        description: str = "",
        request: str = "",
        source_url: str = "",
        risk_level: str = "readonly",
        confirmed: bool = False,
        **kw: Any,
    ) -> str:
        action_key = (action or "").strip()
        if action_key == "list_scripts":
            return self._list_scripts()
        if action_key == "create_script":
            return self._create_script(script_name, language, content)
        if action_key == "run_script":
            return await self._run_script(script_name, args or [], timeout_seconds)
        if action_key == "delete_script":
            return self._delete_script(script_name)
        if action_key == "list_skills":
            return self._list_skills()
        if action_key == "create_skill":
            return self._create_skill(skill_name, content, description)
        if action_key == "list_capabilities":
            return self._list_capabilities()
        if action_key == "discover_capabilities":
            return await self._discover_capabilities(request or description or content)
        if action_key == "install_skill_from_url":
            return await self._install_skill_from_url(
                skill_name=skill_name,
                source_url=source_url,
                description=description or request,
                risk_level=risk_level,
                confirmed=confirmed,
            )
        return f"Error: unknown action '{action_key}'"

    def _valid_name(self, value: str) -> bool:
        return bool(_NAME_RE.fullmatch(value or ""))

    def _ensure_dirs(self) -> None:
        self._scripts_dir.mkdir(parents=True, exist_ok=True)
        self._skills_dir.mkdir(parents=True, exist_ok=True)
        self._capabilities_dir.mkdir(parents=True, exist_ok=True)

    def _resolve_script(self, script_name: str) -> Path | None:
        if not script_name:
            return None
        self._ensure_dirs()
        if not self._valid_name(script_name):
            return None
        candidate = self._scripts_dir / script_name
        if candidate.exists():
            return candidate
        if candidate.suffix:
            return None
        for ext in (".py", ".sh"):
            p = self._scripts_dir / f"{script_name}{ext}"
            if p.exists():
                return p
        return None

    def _create_script(self, script_name: str, language: str, content: str) -> str:
        if not script_name:
            return "Error: script_name is required for create_script"
        if not content:
            return "Error: content is required for create_script"
        if not self._valid_name(script_name):
            return "Error: script_name contains invalid characters"

        self._ensure_dirs()
        ext = ".py" if language == "python" else ".sh"
        path = self._scripts_dir / script_name
        if not path.suffix:
            path = path.with_suffix(ext)

        path.write_text(content, encoding="utf-8")
        if path.suffix == ".sh":
            mode = path.stat().st_mode
            path.chmod(mode | 0o111)
        self._record_capability(
            kind="script",
            name=path.name,
            file_path=str(path),
            source="workbench.create_script",
            summary=f"Shared {language} script",
            risk_level="readonly",
        )
        return f"Created shared script: {path}"

    async def _run_script(self, script_name: str, args: list[str], timeout_seconds: int | None) -> str:
        path = self._resolve_script(script_name)
        if path is None:
            return f"Error: script not found '{script_name}'"
        if not path.is_file():
            return f"Error: not a file '{path.name}'"

        cmd = [sys.executable, str(path)] if path.suffix == ".py" else ["bash", str(path)]
        cmd.extend(str(a) for a in args)
        env = os.environ.copy()
        venv = os.environ.get("VIRTUAL_ENV")
        if venv:
            env["VIRTUAL_ENV"] = venv
            env["PATH"] = f"{Path(venv) / 'bin'}:{env.get('PATH', '')}"

        timeout = timeout_seconds or self._timeout
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(self._workspace),
            env=env,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            return f"Error: script timed out after {timeout} seconds"

        out = stdout.decode("utf-8", errors="replace") if stdout else ""
        err = stderr.decode("utf-8", errors="replace") if stderr else ""
        chunks: list[str] = []
        if out.strip():
            chunks.append(out)
        if err.strip():
            chunks.append(f"STDERR:\n{err}")
        if proc.returncode != 0:
            chunks.append(f"\nExit code: {proc.returncode}")
        result = "\n".join(chunks) if chunks else "(no output)"
        if len(result) > 12000:
            result = result[:12000] + f"\n... (truncated, {len(result) - 12000} more chars)"
        return result

    def _delete_script(self, script_name: str) -> str:
        path = self._resolve_script(script_name)
        if path is None:
            return f"Error: script not found '{script_name}'"
        path.unlink(missing_ok=True)
        return f"Deleted shared script: {path.name}"

    def _list_scripts(self) -> str:
        self._ensure_dirs()
        files = sorted(
            [p for p in self._scripts_dir.iterdir() if p.is_file() and p.suffix in {".py", ".sh"}],
            key=lambda p: p.name,
        )
        if not files:
            return "No shared scripts yet."
        lines = ["Shared scripts:"]
        for p in files:
            size = p.stat().st_size
            lines.append(f"- {p.name} ({size} bytes)")
        return "\n".join(lines)

    def _create_skill(self, skill_name: str, content: str, description: str) -> str:
        if not skill_name:
            return "Error: skill_name is required for create_skill"
        if not content:
            return "Error: content is required for create_skill"
        if not self._valid_name(skill_name):
            return "Error: skill_name contains invalid characters"

        self._ensure_dirs()
        skill_dir = self._skills_dir / skill_name
        skill_dir.mkdir(parents=True, exist_ok=True)
        skill_path = skill_dir / "SKILL.md"
        body = content
        if not content.lstrip().startswith("---"):
            title = description or f"Reusable skill: {skill_name}"
            body = (
                "---\n"
                f"name: {skill_name}\n"
                f"description: {title}\n"
                "---\n\n"
                f"{content}"
            )
        skill_path.write_text(body, encoding="utf-8")
        self._record_capability(
            kind="skill",
            name=skill_name,
            file_path=str(skill_path),
            source="workbench.create_skill",
            summary=description or f"Reusable skill: {skill_name}",
            risk_level="readonly",
        )
        return f"Created shared skill: {skill_path}"

    def _list_skills(self) -> str:
        self._ensure_dirs()
        entries: list[Path] = []
        for d in self._skills_dir.iterdir():
            if not d.is_dir():
                continue
            skill_file = d / "SKILL.md"
            if skill_file.exists():
                entries.append(d)
        entries.sort(key=lambda p: p.name)
        if not entries:
            return "No shared skills yet."
        lines = ["Shared skills:"]
        for d in entries:
            lines.append(f"- {d.name}")
        return "\n".join(lines)

    def _list_capabilities(self) -> str:
        registry = self._load_registry()
        items = registry.get("items", [])
        if not items:
            return "No shared capabilities registered yet."
        lines = ["Shared capabilities:"]
        for item in sorted(items, key=lambda x: x.get("updated_at", ""), reverse=True):
            lines.append(
                "- [{kind}] {name} | risk={risk} | source={source}".format(
                    kind=item.get("kind", "unknown"),
                    name=item.get("name", "unknown"),
                    risk=item.get("risk_level", "readonly"),
                    source=item.get("source", "unknown"),
                )
            )
        return "\n".join(lines)

    async def _discover_capabilities(self, request: str) -> str:
        query = (request or "").strip()
        if not query:
            return "Error: request is required for discover_capabilities"
        queries = [
            query,
            f"{query} site:clawhub.ai",
            f"{query} mcp server",
            f"{query} agent skill",
        ]
        dedup: dict[str, dict[str, str]] = {}
        for q in queries:
            for item in await self._duckduckgo_search(q, limit=4):
                dedup.setdefault(item["url"], item)
        if not dedup:
            return f"No capability candidates found for: {query}"
        ranked = sorted(dedup.values(), key=self._score_candidate, reverse=True)[:10]
        lines = [f"Capability candidates for: {query}"]
        for i, item in enumerate(ranked, 1):
            lines.append(f"{i}. {item['title']}\n   {item['url']}\n   {item['snippet']}")
        lines.append(
            "\nInstall a selected source with: "
            "workbench(action='install_skill_from_url', skill_name='...', source_url='...')"
        )
        return "\n".join(lines)

    async def _install_skill_from_url(
        self,
        skill_name: str,
        source_url: str,
        description: str,
        risk_level: str,
        confirmed: bool,
    ) -> str:
        if not skill_name:
            return "Error: skill_name is required for install_skill_from_url"
        if not self._valid_name(skill_name):
            return "Error: skill_name contains invalid characters"
        if not source_url:
            return "Error: source_url is required for install_skill_from_url"
        if risk_level not in {"readonly", "privileged"}:
            return "Error: risk_level must be readonly or privileged"
        if risk_level == "privileged" and not confirmed:
            return (
                "CONFIRM_REQUIRED: privileged capability install requested. "
                "Ask the user for explicit confirmation, then retry with confirmed=true."
            )

        content = await self._fetch_remote_text(source_url)
        if content.startswith("Error:"):
            return content
        skill_body = self._build_installed_skill_body(
            skill_name=skill_name,
            source_url=source_url,
            description=description,
            fetched_text=content,
        )
        self._ensure_dirs()
        skill_dir = self._skills_dir / skill_name
        skill_dir.mkdir(parents=True, exist_ok=True)
        skill_path = skill_dir / "SKILL.md"
        skill_path.write_text(skill_body, encoding="utf-8")
        self._record_capability(
            kind="skill",
            name=skill_name,
            file_path=str(skill_path),
            source=source_url,
            summary=description or f"Imported skill from {source_url}",
            risk_level=risk_level,
        )
        return f"Installed shared skill: {skill_path}"

    async def _fetch_remote_text(self, url: str) -> str:
        try:
            async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                return resp.text
        except Exception as exc:
            return f"Error: failed to fetch source_url: {exc}"

    async def _duckduckgo_search(self, query: str, limit: int) -> list[dict[str, str]]:
        search_url = f"https://r.jina.ai/http://lite.duckduckgo.com/lite/?q={quote_plus(query)}"
        try:
            async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
                resp = await client.get(search_url)
                resp.raise_for_status()
            return self._parse_duckduckgo_markdown(resp.text, limit)
        except Exception:
            return []

    def _parse_duckduckgo_markdown(self, text: str, limit: int) -> list[dict[str, str]]:
        marker = "Markdown Content:"
        body = text.split(marker, 1)[1] if marker in text else text
        pattern = re.compile(
            r"(?ms)^\s*\d+\.\[(?P<title>.+?)\]\((?P<url>https?://[^\s)]+)\)\s*(?P<snippet>.*?)(?=^\s*\d+\.\[|\Z)"
        )
        results: list[dict[str, str]] = []
        for m in pattern.finditer(body):
            title = m.group("title").strip()
            url = self._unwrap_duckduckgo_redirect(m.group("url").strip())
            snippet = self._clean_snippet(m.group("snippet"))
            if title and url:
                results.append({"title": title, "url": url, "snippet": snippet})
            if len(results) >= limit:
                break
        return results

    @staticmethod
    def _clean_snippet(raw: str) -> str:
        lines = [line.strip() for line in raw.splitlines() if line.strip()]
        if not lines:
            return ""
        return re.sub(r"\*\*(.*?)\*\*", r"\1", lines[0])

    @staticmethod
    def _unwrap_duckduckgo_redirect(url: str) -> str:
        parsed = urlparse(url)
        if "duckduckgo.com" not in parsed.netloc or parsed.path != "/l/":
            return url
        target = parse_qs(parsed.query).get("uddg", [""])[0]
        return unquote(target) if target else url

    @staticmethod
    def _score_candidate(item: dict[str, str]) -> int:
        url = item.get("url", "").lower()
        text = f"{item.get('title', '')} {item.get('snippet', '')}".lower()
        score = 0
        for domain in ("clawhub.ai", "smithery.ai", "github.com", "modelcontextprotocol.io"):
            if domain in url:
                score += 3
        for keyword in ("skill", "agent", "mcp", "tool", "registry"):
            if keyword in text:
                score += 1
        return score

    def _build_installed_skill_body(
        self,
        skill_name: str,
        source_url: str,
        description: str,
        fetched_text: str,
    ) -> str:
        text = fetched_text.strip()
        if text.startswith("---") and "name:" in text and "description:" in text:
            return text
        snippet = text
        if "Markdown Content:" in snippet:
            snippet = snippet.split("Markdown Content:", 1)[1].strip()
        if len(snippet) > 7000:
            snippet = snippet[:7000] + "\n...\n"
        title = description or f"Imported from {source_url}"
        return (
            "---\n"
            f"name: {skill_name}\n"
            f"description: {title}\n"
            "---\n\n"
            f"# {skill_name}\n\n"
            f"- Source: {source_url}\n"
            "- This skill was auto-installed for shared workspace reuse.\n"
            "- Validate instructions before running privileged operations.\n\n"
            "## Reference Snapshot\n\n"
            f"{snippet}\n"
        )

    def _load_registry(self) -> dict[str, Any]:
        self._ensure_dirs()
        if not self._registry_path.exists():
            return {"version": 1, "items": []}
        try:
            data = json.loads(self._registry_path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and isinstance(data.get("items"), list):
                return data
        except Exception:
            pass
        return {"version": 1, "items": []}

    def _save_registry(self, data: dict[str, Any]) -> None:
        self._ensure_dirs()
        self._registry_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _record_capability(
        self,
        kind: str,
        name: str,
        file_path: str,
        source: str,
        summary: str,
        risk_level: str,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        data = self._load_registry()
        items = data.get("items", [])
        target = None
        for item in items:
            if item.get("kind") == kind and item.get("name") == name:
                target = item
                break
        payload = {
            "kind": kind,
            "name": name,
            "file_path": file_path,
            "source": source,
            "summary": summary,
            "risk_level": risk_level,
            "shared": True,
            "updated_at": now,
        }
        if target is None:
            payload["installed_at"] = now
            items.append(payload)
        else:
            installed_at = target.get("installed_at", now)
            target.clear()
            target.update(payload)
            target["installed_at"] = installed_at
        data["items"] = items
        self._save_registry(data)
