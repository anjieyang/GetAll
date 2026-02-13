"""Shared workbench tool for reusable scripts and skills."""

from __future__ import annotations

import asyncio
import os
import re
import sys
from pathlib import Path
from typing import Any

from getall.agent.tools.base import Tool

_NAME_RE = re.compile(r"^[A-Za-z0-9._-]{1,80}$")


class WorkbenchTool(Tool):
    """Create, run, and reuse shared scripts/skills in workspace."""

    def __init__(self, workspace: Path, timeout: int = 180):
        self._workspace = workspace.expanduser().resolve()
        self._scripts_dir = self._workspace / "shared" / "scripts"
        self._skills_dir = self._workspace / "skills"
        self._timeout = timeout

    @property
    def name(self) -> str:
        return "workbench"

    @property
    def description(self) -> str:
        return (
            "Shared automation workbench. Create/run reusable scripts and create/list skills "
            "inside the shared workspace so future agent tasks can reuse them."
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
        return f"Error: unknown action '{action_key}'"

    def _valid_name(self, value: str) -> bool:
        return bool(_NAME_RE.fullmatch(value or ""))

    def _ensure_dirs(self) -> None:
        self._scripts_dir.mkdir(parents=True, exist_ok=True)
        self._skills_dir.mkdir(parents=True, exist_ok=True)

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
