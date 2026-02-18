"""Headless HTML-to-PNG chart renderer using Playwright + Jinja2 + ECharts.

Usage::

    from getall.charts import render_chart

    path = await render_chart("backtest_dashboard", metrics, save_dir=some_dir)
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from loguru import logger

_TEMPLATES_DIR = Path(__file__).parent / "templates"
_STATIC_DIR = Path(__file__).parent / "static"
_DEFAULT_SAVE_DIR = Path("/tmp/getall_charts")

# Sentinel checked by callers to decide matplotlib fallback
PLAYWRIGHT_AVAILABLE: bool = False

try:
    from playwright.async_api import async_playwright, Browser  # noqa: F401

    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    pass

try:
    import jinja2
except ImportError:
    jinja2 = None  # type: ignore[assignment]


class ChartRenderer:
    """Singleton-ish headless renderer.

    * Lazy-inits a Playwright Chromium browser on first ``render()`` call.
    * Reuses the browser across calls; creates a fresh *context* per render
      for isolation.
    * Renders a Jinja2 template to HTML, opens it in the headless browser,
      waits for ``window.__chartReady`` flag, then takes a screenshot.
    """

    def __init__(self) -> None:
        self._pw: Any | None = None
        self._browser: Browser | None = None  # type: ignore[assignment]
        self._jinja_env: jinja2.Environment | None = None  # type: ignore[union-attr]

    # ── Lifecycle ────────────────────────────────────────

    async def _ensure_browser(self) -> Browser:  # type: ignore[return]
        """Launch Chromium on first call, reuse afterwards."""
        if self._browser is not None and self._browser.is_connected():
            return self._browser

        from playwright.async_api import async_playwright

        self._pw = await async_playwright().start()
        self._browser = await self._pw.chromium.launch(
            headless=True,
            args=[
                "--no-sandbox",
                "--disable-setuid-sandbox",
                "--disable-dev-shm-usage",
                "--disable-gpu",
            ],
        )
        logger.info("ChartRenderer: Chromium browser launched")
        return self._browser

    def _get_jinja_env(self) -> jinja2.Environment:  # type: ignore[name-defined]
        """Return (cached) Jinja2 environment with template loader."""
        if self._jinja_env is not None:
            return self._jinja_env

        if jinja2 is None:
            raise ImportError("jinja2 is required for chart rendering: pip install jinja2")

        self._jinja_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(_TEMPLATES_DIR)),
            autoescape=False,
        )
        return self._jinja_env

    async def close(self) -> None:
        """Shut down browser gracefully."""
        if self._browser is not None:
            try:
                await self._browser.close()
            except Exception:
                pass
            self._browser = None
        if self._pw is not None:
            try:
                await self._pw.stop()
            except Exception:
                pass
            self._pw = None

    # ── ECharts loader ─────────────────────────────────

    _echarts_js_cache: str | None = None

    def _load_echarts_js(self) -> str:
        """Read and cache the bundled echarts.min.js content.

        Inlined into templates because Playwright's ``set_content()``
        renders from ``about:blank`` origin which blocks ``file://`` URIs.
        """
        if self._echarts_js_cache is not None:
            return self._echarts_js_cache

        echarts_path = _STATIC_DIR / "echarts.min.js"
        if not echarts_path.exists():
            logger.warning("echarts.min.js not found in static/, charts will be empty")
            self._echarts_js_cache = ""
            return ""

        self._echarts_js_cache = echarts_path.read_text(encoding="utf-8")
        logger.debug(f"Loaded echarts.min.js ({len(self._echarts_js_cache)} bytes)")
        return self._echarts_js_cache

    # ── Rendering ────────────────────────────────────────

    async def render(
        self,
        template: str,
        data: dict[str, Any],
        *,
        width: int = 1600,
        height: int = 1000,
        scale: float = 2.0,
        save_dir: Path | None = None,
        filename: str | None = None,
        timeout_ms: int = 15_000,
    ) -> Path:
        """Render *template* with *data* and return path to the PNG.

        Parameters
        ----------
        template:
            Template name without extension (e.g. ``"backtest_dashboard"``).
        data:
            Dict passed to the Jinja2 template as ``{{ data }}``.
        width, height:
            Viewport size in CSS pixels.
        scale:
            Device scale factor (2.0 = Retina).
        save_dir:
            Directory for the output PNG.  Defaults to ``/tmp/getall_charts``.
        filename:
            Optional explicit filename.  Auto-generated if *None*.
        timeout_ms:
            Max time (ms) to wait for ``window.__chartReady``.
        """
        browser = await self._ensure_browser()
        env = self._get_jinja_env()

        # Render HTML from template
        tpl = env.get_template(f"{template}.html")

        # Inline ECharts JS — file:// URIs are blocked by set_content() origin
        echarts_js = self._load_echarts_js()

        html = tpl.render(
            data=data,
            data_json=json.dumps(data, default=str),
            echarts_inline_js=echarts_js,
        )

        # Create isolated context + page
        context = await browser.new_context(
            viewport={"width": width, "height": height},
            device_scale_factor=scale,
        )
        page = await context.new_page()

        try:
            # Capture JS errors for debugging
            js_errors: list[str] = []
            page.on(
                "pageerror",
                lambda exc: (
                    js_errors.append(str(exc)),
                    logger.warning(f"ChartRenderer JS error: {exc}"),
                )[0],
            )
            page.on(
                "console",
                lambda msg: logger.debug(f"ChartRenderer console [{msg.type}]: {msg.text}")
                if msg.type in ("error", "warning")
                else None,
            )

            await page.set_content(html, wait_until="networkidle")

            # Wait for the template JS to signal readiness
            try:
                await page.wait_for_function(
                    "() => window.__chartReady === true",
                    timeout=timeout_ms,
                )
            except Exception:
                err_detail = f" | JS errors: {js_errors}" if js_errors else ""
                logger.warning(
                    f"ChartRenderer: __chartReady not set within {timeout_ms}ms "
                    f"for template '{template}', proceeding with screenshot anyway{err_detail}"
                )

            # Screenshot
            if save_dir is None:
                save_dir = _DEFAULT_SAVE_DIR
            save_dir.mkdir(parents=True, exist_ok=True)

            if filename is None:
                filename = f"{template}_{uuid.uuid4().hex[:8]}.png"
            out_path = save_dir / filename

            await page.screenshot(path=str(out_path), full_page=True)
            logger.info(f"ChartRenderer: saved {out_path} ({width}x{height} @{scale}x)")
            return out_path
        finally:
            await context.close()


# ── Module-level singleton ──────────────────────────────

_renderer: ChartRenderer | None = None


def _get_renderer() -> ChartRenderer:
    """Return the module-level singleton, creating it lazily."""
    global _renderer
    if _renderer is None:
        _renderer = ChartRenderer()
    return _renderer


async def render_chart(
    template: str,
    data: dict[str, Any],
    *,
    width: int = 1600,
    height: int = 1000,
    scale: float = 2.0,
    save_dir: Path | None = None,
    filename: str | None = None,
    timeout_ms: int = 15_000,
) -> Path:
    """Module-level convenience wrapper around :pyclass:`ChartRenderer.render`.

    This is the primary public API.
    """
    renderer = _get_renderer()
    return await renderer.render(
        template,
        data,
        width=width,
        height=height,
        scale=scale,
        save_dir=save_dir,
        filename=filename,
        timeout_ms=timeout_ms,
    )
