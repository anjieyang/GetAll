"""回测与 Paper Trading 工具.

Actions:
  - run_backtest:      历史回测 (JSON 策略配置)
  - run_custom:        自定义策略文件回测
  - start_paper:       启动 Paper Trading (Binance Testnet)
  - stop_paper:        停止 Paper Trading 并获取结果
  - paper_status:      查看 Paper Trading 状态
  - list_indicators:   列出所有可用的内置指标
"""

import json
from typing import Any

import pandas as pd
from loguru import logger

from getall.agent.tools.base import Tool
from getall.trading.backtest.data_loader import OHLCVLoader
from getall.trading.backtest.engine import BacktestRunner
from getall.trading.backtest.strategies import StrategySpec
from getall.trading.data.hub import DataHub


class BacktestTool(Tool):
    """历史回测与 Paper Trading 工具.

    支持:
      - 基于 JSON 策略配置的自动回测 (TemplateStrategy)
      - 用户自定义 .py 策略文件回测
      - Binance Testnet Paper Trading (实时模拟)
      - 内置指标查询
    """

    def __init__(self, hub: DataHub, runner: BacktestRunner):
        self.hub = hub
        self.runner = runner

    @property
    def name(self) -> str:
        return "backtest"

    @property
    def description(self) -> str:
        return (
            "Run historical backtests and paper trading. Actions: "
            "run_backtest (JSON strategy config), run_custom (user .py file), "
            "start_paper (Binance Testnet live simulation), stop_paper, "
            "paper_status, list_indicators."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "The backtest action to perform",
                    "enum": [
                        "run_backtest",
                        "run_custom",
                        "start_paper",
                        "stop_paper",
                        "paper_status",
                        "list_indicators",
                    ],
                },
                "strategy_config": {
                    "type": "string",
                    "description": (
                        "JSON string of strategy configuration for run_backtest / start_paper. "
                        "Format: {\"name\": \"my_strat\", \"symbols\": [\"BTC/USDT\"], "
                        "\"timeframe\": \"4h\", \"indicators\": [{\"name\": \"rsi\", \"params\": {\"period\": 14}}], "
                        "\"entry_conditions\": [{\"indicator\": \"rsi\", \"field\": \"value\", \"operator\": \"lt\", \"value\": 30}], "
                        "\"exit_conditions\": [{\"indicator\": \"rsi\", \"field\": \"value\", \"operator\": \"gt\", \"value\": 70}], "
                        "\"direction\": \"long\", \"stop_loss_pct\": 5, \"take_profit_pct\": 15}"
                    ),
                },
                "strategy_file": {
                    "type": "string",
                    "description": "Path to custom .py strategy file (for run_custom)",
                },
                "strategy_code": {
                    "type": "string",
                    "description": (
                        "Python strategy code string (for run_custom, alternative to strategy_file). "
                        "Must contain a class that inherits from nautilus_trader.trading.strategy.Strategy. "
                        "Example: 'from nautilus_trader.trading.strategy import Strategy\\n"
                        "class MyStrategy(Strategy):\\n    def on_bar(self, bar): ...'"
                    ),
                },
                "period": {
                    "type": "string",
                    "description": "Backtest period, e.g. '6m', '1y', '30d'. Default: '6m'",
                },
                "starting_balance": {
                    "type": "number",
                    "description": "Starting balance for backtest. Default: 100000",
                },
                "session_id": {
                    "type": "string",
                    "description": "Paper trading session ID (for stop_paper / paper_status)",
                },
                "api_key": {
                    "type": "string",
                    "description": "Binance testnet API key (for start_paper)",
                },
                "api_secret": {
                    "type": "string",
                    "description": "Binance testnet API secret (for start_paper)",
                },
                "generate_chart": {
                    "type": "boolean",
                    "description": "Generate equity curve chart (default: false)",
                },
            },
            "required": ["action"],
        }

    async def execute(self, **kwargs: Any) -> str:
        action: str = kwargs["action"]

        try:
            handlers = {
                "run_backtest": self._run_backtest,
                "run_custom": self._run_custom,
                "start_paper": self._start_paper,
                "stop_paper": self._stop_paper,
                "paper_status": self._paper_status,
                "list_indicators": self._list_indicators,
            }
            handler = handlers.get(action)
            if handler is None:
                return f"Error: unknown action '{action}'"
            return await handler(**kwargs)
        except Exception as e:
            logger.error(f"Backtest tool error ({action}): {e}")
            return f"Error in backtest/{action}: {e}"

    # ═══════════════════════════════════════════════════════
    # run_backtest: 历史回测
    # ═══════════════════════════════════════════════════════

    async def _run_backtest(self, **kwargs: Any) -> str:
        """运行历史回测."""
        logger.info("📊 Backtest tool called - starting historical backtest")
        config_str = kwargs.get("strategy_config")
        if not config_str:
            return "Error: 'strategy_config' (JSON string) is required for run_backtest"

        try:
            config_data = json.loads(config_str)
            logger.info(f"📊 Backtest config: {json.dumps(config_data, indent=2)}")
        except json.JSONDecodeError as e:
            return f"Error: invalid JSON in strategy_config: {e}"

        spec = StrategySpec.from_json(config_data)

        if not spec.symbols:
            return "Error: 'symbols' is required in strategy_config"
        if not spec.entry_conditions:
            return "Error: 'entry_conditions' is required in strategy_config"

        period = kwargs.get("period", "6m")
        starting_balance = kwargs.get("starting_balance", 100_000)

        logger.info(f"📊 Backtest params - Period: {period}, Symbols: {spec.symbols}, Timeframe: {spec.timeframe}")

        # 拉取历史数据
        ohlcv_data: dict[str, pd.DataFrame] = {}
        for symbol in spec.symbols:
            candle_count = OHLCVLoader.estimate_candle_count(period, spec.timeframe)
            candle_count = min(candle_count, 1000)  # ccxt 限制

            ex = self.hub.exchange
            if ex is None:
                return "Error: no exchange configured. Check exchanges.yaml."

            klines = await ex.get_klines(symbol, spec.timeframe, limit=candle_count)
            if isinstance(klines, dict) and "error" in klines:
                return f"Error fetching data for {symbol}: {klines['error']}"

            df = OHLCVLoader.ccxt_to_dataframe(klines, symbol)
            if df.empty:
                return f"Error: no kline data for {symbol} ({spec.timeframe})"
            logger.info(f"📊 Fetched {len(df)} candles for {symbol} from {df.index[0]} to {df.index[-1]}")
            ohlcv_data[symbol] = df

        # 运行回测
        logger.info(f"📊 Starting backtest execution with {len(ohlcv_data)} symbols...")
        result = await self.runner.run_backtest(
            ohlcv_data=ohlcv_data,
            strategy_spec=spec,
            starting_balance=starting_balance,
        )

        logger.info(f"📊 Backtest completed - Trades: {result.total_trades}, Win Rate: {result.win_rate:.2f}%")

        # 生成图表 (如果请求)
        chart_path = None
        generate_chart = kwargs.get("generate_chart", False)
        if generate_chart:
            logger.info("📊 Generating equity curve chart...")
            chart_path = result.generate_equity_chart()
            if chart_path:
                logger.info(f"📊 Chart saved to: {chart_path}")

        # 构建报告
        report = result.to_report()

        # 如果生成了图表，添加图表路径信息
        if chart_path:
            report += f"\n\n📈 Equity curve chart saved to:\n  {chart_path}\n"
            report += f"\n💡 Use send_image() or channel-specific methods to send this chart."

        return report

    # ═══════════════════════════════════════════════════════
    # run_custom: 自定义策略文件回测
    # ═══════════════════════════════════════════════════════

    async def _run_custom(self, **kwargs: Any) -> str:
        """运行自定义 .py 策略文件回测."""
        strategy_file = kwargs.get("strategy_file")
        strategy_code = kwargs.get("strategy_code")

        # 优先使用 strategy_code，创建临时文件
        if strategy_code:
            import tempfile
            import uuid
            from pathlib import Path

            temp_dir = Path(tempfile.gettempdir()) / "getall_strategies"
            temp_dir.mkdir(parents=True, exist_ok=True)
            strategy_file = str(temp_dir / f"strategy_{uuid.uuid4().hex[:8]}.py")

            Path(strategy_file).write_text(strategy_code)
            logger.info(f"📊 Created temporary strategy file: {strategy_file}")

        if not strategy_file:
            return "Error: either 'strategy_file' (path to .py file) or 'strategy_code' (Python code string) is required for run_custom"

        config_str = kwargs.get("strategy_config", "{}")
        try:
            config_data = json.loads(config_str)
        except json.JSONDecodeError:
            config_data = {}

        spec = StrategySpec.from_json(config_data)
        period = kwargs.get("period", "6m")

        # 拉取数据
        ohlcv_data: dict[str, pd.DataFrame] = {}
        for symbol in (spec.symbols or ["BTC/USDT"]):
            candle_count = OHLCVLoader.estimate_candle_count(period, spec.timeframe)
            candle_count = min(candle_count, 1000)

            ex = self.hub.exchange
            if ex is None:
                return "Error: no exchange configured"

            klines = await ex.get_klines(symbol, spec.timeframe, limit=candle_count)
            if isinstance(klines, dict) and "error" in klines:
                continue
            df = OHLCVLoader.ccxt_to_dataframe(klines, symbol)
            if not df.empty:
                ohlcv_data[symbol] = df

        if not ohlcv_data:
            return "Error: could not fetch any data for backtesting"

        result = await self.runner.run_custom(
            strategy_file=strategy_file,
            ohlcv_data=ohlcv_data,
            starting_balance=kwargs.get("starting_balance", 100_000),
        )

        return result.to_report()

    # ═══════════════════════════════════════════════════════
    # start_paper: 启动 Paper Trading
    # ═══════════════════════════════════════════════════════

    async def _start_paper(self, **kwargs: Any) -> str:
        """启动 Paper Trading 会话."""
        config_str = kwargs.get("strategy_config")
        if not config_str:
            return "Error: 'strategy_config' (JSON string) is required for start_paper"

        try:
            config_data = json.loads(config_str)
        except json.JSONDecodeError as e:
            return f"Error: invalid JSON in strategy_config: {e}"

        spec = StrategySpec.from_json(config_data)
        api_key = kwargs.get("api_key", "")
        api_secret = kwargs.get("api_secret", "")

        session = await self.runner.start_paper(
            strategy_spec=spec,
            api_key=api_key,
            api_secret=api_secret,
        )

        status_note = ""
        if session.status == "simulated":
            status_note = (
                "\n  Note: Running in simulated mode (nautilus_trader Binance adapter "
                "not available). Install nautilus_trader for live testnet connection."
            )
        elif session.status == "error":
            status_note = "\n  Warning: Failed to connect to Binance testnet."

        return (
            f"Paper Trading Session Started\n"
            f"{'═' * 50}\n"
            f"  Session ID: {session.session_id}\n"
            f"  Strategy: {session.strategy_name}\n"
            f"  Symbols: {', '.join(session.symbols)}\n"
            f"  Status: {session.status}\n"
            f"  Started: {session.started_at}\n"
            f"{status_note}\n"
            f"{'═' * 50}\n"
            f"Use backtest(action='stop_paper', session_id='{session.session_id}') to stop.\n"
            f"Use backtest(action='paper_status', session_id='{session.session_id}') to check status."
        )

    # ═══════════════════════════════════════════════════════
    # stop_paper: 停止 Paper Trading
    # ═══════════════════════════════════════════════════════

    async def _stop_paper(self, **kwargs: Any) -> str:
        """停止 Paper Trading 并输出结果."""
        session_id = kwargs.get("session_id")
        if not session_id:
            return "Error: 'session_id' is required for stop_paper"

        result = await self.runner.stop_paper(session_id)

        if not result.success and result.error:
            return f"Error stopping paper trading: {result.error}"

        return (
            f"Paper Trading Stopped\n"
            f"{'═' * 50}\n"
            f"{result.to_report()}"
        )

    # ═══════════════════════════════════════════════════════
    # paper_status: 查看状态
    # ═══════════════════════════════════════════════════════

    async def _paper_status(self, **kwargs: Any) -> str:
        """查看 Paper Trading 会话状态."""
        session_id = kwargs.get("session_id")

        if session_id:
            session = self.runner.get_paper_session(session_id)
            if session is None:
                return f"Session not found: {session_id}"
            return (
                f"Paper Trading Status\n"
                f"{'─' * 40}\n"
                f"  Session: {session.session_id}\n"
                f"  Strategy: {session.strategy_name}\n"
                f"  Symbols: {', '.join(session.symbols)}\n"
                f"  Status: {session.status}\n"
                f"  Started: {session.started_at}\n"
                f"  Stopped: {session.stopped_at or 'still running'}"
            )

        # 列出所有会话
        sessions = self.runner.list_paper_sessions()
        if not sessions:
            return "No paper trading sessions."

        lines = ["Paper Trading Sessions", "─" * 40]
        for s in sessions:
            lines.append(
                f"  {s.session_id} | {s.strategy_name} | {s.status} | "
                f"started: {s.started_at[:16]}"
            )
        return "\n".join(lines)

    # ═══════════════════════════════════════════════════════
    # list_indicators: 列出可用指标
    # ═══════════════════════════════════════════════════════

    async def _list_indicators(self, **kwargs: Any) -> str:
        """列出所有可用的内置指标."""
        indicators = BacktestRunner.available_indicators()

        lines = [
            "Available Indicators for Backtesting",
            "═" * 55,
            "",
            "Nautilus Trader Built-in (use in strategy directly):",
            "─" * 55,
        ]

        categories = {
            "Moving Averages": ["sma", "ema", "dema", "hma", "wma", "vwap"],
            "Momentum": ["rsi", "macd", "bollinger", "cci", "stoch", "roc", "aroon"],
            "Volatility": ["atr", "donchian", "keltner"],
            "Ratio": ["efficiency_ratio"],
        }

        for cat_name, keys in categories.items():
            lines.append(f"\n  {cat_name}:")
            for key in keys:
                info = indicators.get(key, {})
                params = info.get("default_params", {})
                param_str = ", ".join(f"{k}={v}" for k, v in params.items())
                lines.append(f"    {key:20s} defaults: ({param_str})")

        lines.extend([
            "",
            "Additional via pandas-ta (pre-compute, feed as custom data):",
            "─" * 55,
            "  130+ indicators including: Ichimoku, SuperTrend, VWMA,",
            "  OBV, MFI, ADX, Williams %R, Chaikin, Fisher Transform,",
            "  Squeeze, Kama, TRIX, PPO, Vortex, and more.",
            "",
            "Decision rule:",
            "  - Indicator in Nautilus  -> use native (faster, event-driven)",
            "  - Not in Nautilus        -> pre-compute with pandas-ta",
            "  - Complex composite      -> Agent writes custom .py strategy",
        ])

        return "\n".join(lines)
