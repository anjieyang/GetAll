"""Chart rendering engine — HTML/ECharts/Playwright.

Public API
----------
- ``render_chart``   – render a named template to PNG (async).
- ``PLAYWRIGHT_AVAILABLE`` – whether the Playwright backend is usable.

Example::

    from getall.charts import render_chart, PLAYWRIGHT_AVAILABLE

    if PLAYWRIGHT_AVAILABLE:
        path = await render_chart("backtest_dashboard", metrics_dict)
"""

from getall.charts.renderer import PLAYWRIGHT_AVAILABLE, render_chart

__all__ = ["render_chart", "PLAYWRIGHT_AVAILABLE"]
