"""Browser automation tool powered by browser-use.

Provides AI-driven browser interaction for tasks that require navigating
real web pages — e.g. scraping Dune Analytics dashboards, DEXScreener,
or any site without a public API.

Heavy by design: each invocation spins up a headless Chromium instance,
runs the task, and tears it down.  Use sparingly — API tools are always
preferred when available.
"""

import asyncio
import time
from pathlib import Path
from typing import Any

from loguru import logger

from getall.agent.tools.base import Tool

# Directory for browser task screenshots
_SCREENSHOT_DIR = Path("/tmp/getall_browser")


class BrowserUseTool(Tool):
    """AI-controlled browser for web scraping and interaction tasks."""

    def __init__(
        self,
        model: str,
        api_key: str,
        api_base: str | None = None,
        extra_headers: dict[str, str] | None = None,
        timeout: int = 180,
    ):
        self._model = model
        self._api_key = api_key
        self._api_base = api_base
        self._extra_headers = extra_headers or {}
        self._timeout = timeout

    @property
    def name(self) -> str:
        return "browser_use"

    @property
    def description(self) -> str:
        return (
            "Use an AI-controlled browser to interact with websites that lack APIs. "
            "The browser can navigate pages, click buttons, fill forms, extract data, "
            "and take screenshots. SLOW (30-120s per task) — only use when API tools "
            "(market_data, coingecko, web_fetch, etc.) cannot provide the data. "
            "Good for: Dune Analytics dashboards, DEXScreener, exchange announcement "
            "pages, social media scraping, airdrop eligibility checks."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": (
                        "Natural language description of what to do in the browser. "
                        "Be specific about what data to extract and in what format. "
                        "Example: 'Go to the Dune query page, wait for the table to "
                        "load, and extract all rows as a JSON array.'"
                    ),
                },
                "url": {
                    "type": "string",
                    "description": (
                        "Optional starting URL. If provided, the browser navigates "
                        "here first before executing the task."
                    ),
                },
                "max_steps": {
                    "type": "integer",
                    "description": (
                        "Maximum number of browser actions (click, type, scroll, etc.). "
                        "Lower = faster but may fail on complex pages. Default: 15."
                    ),
                    "minimum": 3,
                    "maximum": 50,
                },
            },
            "required": ["task"],
        }

    async def execute(
        self,
        task: str,
        url: str | None = None,
        max_steps: int = 15,
        **kwargs: Any,
    ) -> str:
        """Run a browser automation task and return the result."""
        # Lazy import — don't break startup if browser-use isn't installed
        try:
            from browser_use import Agent, Browser, BrowserProfile
            from browser_use.llm.openai.chat import ChatOpenAI
        except ImportError:
            return (
                "Error: browser-use is not installed. "
                "Run: pip install browser-use"
            )

        # Build the full task description
        full_task = task
        if url:
            full_task = f"First navigate to {url}. Then: {task}"

        # Prepare screenshot directory
        _SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

        browser: Any = None
        t_start = time.time()

        try:
            # Create LLM — uses a dedicated model (e.g. gpt-5.2) that supports
            # both structured output and vision for reliable browser automation.
            llm_kwargs: dict[str, Any] = {
                "model": self._model,
                "api_key": self._api_key,
                "temperature": 0,  # Deterministic for browser tasks
            }
            if self._api_base:
                llm_kwargs["base_url"] = self._api_base
            if self._extra_headers:
                llm_kwargs["default_headers"] = self._extra_headers

            llm = ChatOpenAI(**llm_kwargs)

            # Create headless browser session
            browser = Browser(
                browser_profile=BrowserProfile(
                    headless=True,
                    disable_security=False,
                )
            )

            # Create and run the browser-use agent
            agent = Agent(
                task=full_task,
                llm=llm,
                browser=browser,
                max_actions_per_step=4,
            )

            # Run with timeout protection
            history = await asyncio.wait_for(
                agent.run(max_steps=max_steps),
                timeout=self._timeout,
            )

            elapsed = time.time() - t_start
            logger.info(f"browser_use completed in {elapsed:.1f}s ({max_steps} max steps)")

            # Extract result
            result = self._extract_result(history)

            # Try to save a screenshot for visual reference
            screenshot_path = self._save_screenshot(history)
            if screenshot_path:
                result += f"\n[GENERATED_IMAGE:{screenshot_path}]"

            return result

        except asyncio.TimeoutError:
            elapsed = time.time() - t_start
            logger.warning(f"browser_use timed out after {elapsed:.1f}s")
            return f"Error: Browser task timed out after {self._timeout}s. Try simplifying the task or increasing max_steps."

        except Exception as e:
            elapsed = time.time() - t_start
            logger.error(f"browser_use failed after {elapsed:.1f}s: {e}")
            return f"Error: Browser task failed: {e}"

        finally:
            if browser is not None:
                try:
                    await browser.stop()
                except Exception:
                    pass  # Best-effort cleanup

    @staticmethod
    def _extract_result(history: Any) -> str:
        """Extract useful text from browser-use AgentHistory."""
        parts: list[str] = []

        # Primary: final result text
        try:
            final = history.final_result()
            if final:
                parts.append(final)
        except Exception:
            pass

        # Fallback: if no final result, collect action results
        if not parts:
            try:
                for action_result in history.action_results():
                    if action_result and action_result.extracted_content:
                        parts.append(action_result.extracted_content)
            except Exception:
                pass

        # Include errors if any
        try:
            errors = history.errors()
            if errors:
                error_texts = [str(e) for e in errors if e]
                if error_texts:
                    parts.append(f"Browser warnings: {'; '.join(error_texts[:3])}")
        except Exception:
            pass

        if not parts:
            return "(Browser task completed but returned no extractable content)"

        return "\n\n".join(parts)

    @staticmethod
    def _save_screenshot(history: Any) -> str | None:
        """Save the last screenshot from the browser session, if available."""
        try:
            screenshots = history.screenshots()
            if not screenshots:
                return None
            # Save the last screenshot
            last_screenshot = screenshots[-1]
            ts = int(time.time())
            path = _SCREENSHOT_DIR / f"browser_{ts}.png"
            if isinstance(last_screenshot, bytes):
                path.write_bytes(last_screenshot)
                return str(path)
            elif isinstance(last_screenshot, str):
                # Base64 encoded
                import base64
                data = base64.b64decode(last_screenshot)
                path.write_bytes(data)
                return str(path)
        except Exception as e:
            logger.debug(f"Failed to save browser screenshot: {e}")
        return None
