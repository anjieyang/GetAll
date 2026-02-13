"""回测引擎: 封装 Nautilus Trader 的 BacktestEngine / TradingNode.

提供:
  - run_backtest(): 历史回测 (BacktestEngine low-level API)
  - start_paper() / stop_paper(): Paper Trading (TradingNode + testnet)
"""

from __future__ import annotations

import asyncio
import json
import importlib.util
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from getall.trading.backtest.data_loader import OHLCVLoader
from getall.trading.backtest.strategies import StrategySpec, NAUTILUS_INDICATORS

# Nautilus 按需导入
_NT_AVAILABLE = False
try:
    from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
    from nautilus_trader.backtest.models import FillModel
    from nautilus_trader.config import LoggingConfig
    from nautilus_trader.model import (
        Bar,
        BarType,
        InstrumentId,
        Quantity,
        Venue,
        Money,
        Price,
    )
    from nautilus_trader.model.enums import AccountType, OmsType
    from nautilus_trader.model.objects import Currency
    from nautilus_trader.model.instruments import CryptoPerpetual
    from nautilus_trader.model.identifiers import Symbol
    from nautilus_trader.test_kit.providers import TestInstrumentProvider

    _NT_AVAILABLE = True
except ImportError:
    logger.debug("nautilus_trader not installed — BacktestRunner limited to dry mode")


# ═══════════════════════════════════════════════════════
# 回测结果
# ═══════════════════════════════════════════════════════


@dataclass
class BacktestResult:
    """回测结果摘要."""
    strategy_name: str = ""
    symbols: list[str] = field(default_factory=list)
    timeframe: str = ""
    period_start: str = ""
    period_end: str = ""
    total_days: int = 0

    # 绩效指标
    total_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    total_pnl: float = 0.0
    annualized_return_pct: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    best_trade: float = 0.0
    worst_trade: float = 0.0
    avg_hold_time: str = ""
    expectancy: float = 0.0

    # 原始数据 (可选, 用于详细分析)
    positions_report: str = ""
    orders_report: str = ""

    # 权益曲线数据 (用于生成图表)
    equity_curve: list[dict[str, Any]] = field(default_factory=list)  # [{"timestamp": ..., "equity": ..., "drawdown": ...}]

    # 状态
    success: bool = True
    error: str = ""

    def to_report(self) -> str:
        """格式化为人类可读的报告."""
        if not self.success:
            return f"Backtest Failed: {self.error}"

        return (
            f"Backtest Results: {self.strategy_name}\n"
            f"{'═' * 55}\n"
            f"Period: {self.period_start} -> {self.period_end} ({self.total_days} days)\n"
            f"Symbols: {', '.join(self.symbols)}\n"
            f"Timeframe: {self.timeframe}\n"
            f"\n"
            f"Performance\n"
            f"{'─' * 55}\n"
            f"  Total Trades:      {self.total_trades}\n"
            f"  Win Rate:          {self.win_rate:.1f}%\n"
            f"  Profit Factor:     {self.profit_factor:.2f}\n"
            f"  Max Drawdown:      {self.max_drawdown_pct:.2f}%\n"
            f"  Sharpe Ratio:      {self.sharpe_ratio:.2f}\n"
            f"  Total P&L:         {self.total_pnl:.2f}\n"
            f"  Annualized Return: {self.annualized_return_pct:.2f}%\n"
            f"  Expectancy:        {self.expectancy:.4f}/trade\n"
            f"\n"
            f"Trade Distribution\n"
            f"{'─' * 55}\n"
            f"  Best Trade:  +{self.best_trade:.2f}\n"
            f"  Worst Trade: {self.worst_trade:.2f}\n"
            f"  Avg Winner:  +{self.avg_win:.2f}\n"
            f"  Avg Loser:   {self.avg_loss:.2f}\n"
            f"  Avg Hold:    {self.avg_hold_time}\n"
            f"\n"
            f"Quality Assessment\n"
            f"{'─' * 55}\n"
            f"{self._quality_assessment()}\n"
            f"{'═' * 55}"
        )

    def _quality_assessment(self) -> str:
        """生成质量评估."""
        lines: list[str] = []

        if self.total_trades < 20:
            lines.append("  [!] Low sample size (<20 trades) — results may not be statistically significant")
        if self.win_rate > 80:
            lines.append("  [!] Very high win rate (>80%) — potential overfitting risk")
        if self.max_drawdown_pct > 20:
            lines.append("  [!] High drawdown (>20%) — consider tighter stop-losses")

        if self.win_rate > 55 and self.profit_factor > 1.5:
            lines.append("  [+] Strategy looks promising (WR>55%, PF>1.5)")
        elif self.win_rate < 40 or self.profit_factor < 1.0:
            lines.append("  [-] Strategy needs improvement (low WR or PF<1)")
        else:
            lines.append("  [~] Moderate performance — consider parameter optimization")

        if self.sharpe_ratio > 1.5:
            lines.append("  [+] Good risk-adjusted return (Sharpe>1.5)")
        elif self.sharpe_ratio < 0.5:
            lines.append("  [-] Poor risk-adjusted return (Sharpe<0.5)")

        return "\n".join(lines) if lines else "  No specific concerns."

    def to_strategy_yaml(self) -> str:
        """生成 STRATEGY.md 中的 backtest: YAML 区块."""
        return (
            f"backtest:\n"
            f'  engine: "nautilus_trader"\n'
            f'  period: "{self.period_start} ~ {self.period_end}"\n'
            f"  win_rate: {self.win_rate:.1f}\n"
            f"  profit_factor: {self.profit_factor:.2f}\n"
            f'  max_drawdown: "{self.max_drawdown_pct:.2f}%"\n'
            f"  sharpe_ratio: {self.sharpe_ratio:.2f}\n"
            f"  total_trades: {self.total_trades}"
        )

    def generate_equity_chart(self, save_path: str | None = None) -> str | None:
        """生成收益曲线图.

        Args:
            save_path: 图片保存路径. 如果为 None, 则自动生成临时文件路径.

        Returns:
            图片文件路径, 如果没有数据则返回 None.
        """
        if not self.equity_curve:
            logger.warning("No equity curve data available for chart generation")
            return None

        try:
            import matplotlib
            matplotlib.use('Agg')  # 无头模式
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            from datetime import datetime as dt
        except ImportError:
            logger.error("matplotlib not installed. Run: pip install matplotlib")
            return None

        # 准备数据
        timestamps = []
        equities = []
        drawdowns = []

        for point in self.equity_curve:
            ts = point.get("timestamp")
            if isinstance(ts, str):
                try:
                    timestamps.append(dt.fromisoformat(ts.replace("Z", "+00:00")))
                except ValueError:
                    timestamps.append(dt.now())
            else:
                timestamps.append(ts)
            equities.append(point.get("equity", 0))
            drawdowns.append(point.get("drawdown", 0))

        if not timestamps:
            return None

        # 创建图表
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
        fig.suptitle(f'Backtest Results: {self.strategy_name}', fontsize=14, fontweight='bold')

        # 上图: 权益曲线
        ax1.plot(timestamps, equities, 'b-', linewidth=1.5, label='Equity')
        ax1.fill_between(timestamps, equities[0] if equities else 0, equities, alpha=0.3)
        ax1.axhline(y=equities[0] if equities else 100000, color='gray', linestyle='--', alpha=0.5, label='Initial')
        ax1.set_ylabel('Equity', fontsize=10)
        ax1.set_title(f'Period: {self.period_start} → {self.period_end} | '
                      f'Trades: {self.total_trades} | Win Rate: {self.win_rate:.1f}%', fontsize=10)
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)

        # 格式化 x 轴日期
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        ax1.xaxis.set_major_locator(mdates.AutoDateLocator())

        # 下图: 回撤曲线
        ax2.fill_between(timestamps, 0, drawdowns, color='red', alpha=0.5)
        ax2.plot(timestamps, drawdowns, 'r-', linewidth=1)
        ax2.set_ylabel('Drawdown %', fontsize=10)
        ax2.set_xlabel('Date', fontsize=10)
        ax2.set_ylim(min(drawdowns) * 1.1 if drawdowns and min(drawdowns) < 0 else -10, 0)
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        ax2.xaxis.set_major_locator(mdates.AutoDateLocator())

        # 添加关键指标注释
        stats_text = (
            f"P&L: {self.total_pnl:+.2f}% | "
            f"Max DD: {self.max_drawdown_pct:.2f}% | "
            f"Sharpe: {self.sharpe_ratio:.2f} | "
            f"PF: {self.profit_factor:.2f}"
        )
        fig.text(0.5, 0.02, stats_text, ha='center', fontsize=9, style='italic')

        plt.tight_layout(rect=[0, 0.03, 1, 0.97])

        # 保存图片
        if save_path is None:
            import tempfile
            temp_dir = Path(tempfile.gettempdir()) / "getall_charts"
            temp_dir.mkdir(parents=True, exist_ok=True)
            save_path = str(temp_dir / f"backtest_{self.strategy_name}_{uuid.uuid4().hex[:8]}.png")

        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)

        logger.info(f"Equity chart saved to: {save_path}")
        return save_path


# ═══════════════════════════════════════════════════════
# Paper Trading Session 管理
# ═══════════════════════════════════════════════════════


@dataclass
class PaperSession:
    """Paper Trading 会话信息."""
    session_id: str
    strategy_name: str
    symbols: list[str]
    status: str = "running"       # running / stopped
    started_at: str = ""
    stopped_at: str = ""
    config_json: str = ""
    # TradingNode 实例 (运行时)
    _node: Any = field(default=None, repr=False)
    _task: Any = field(default=None, repr=False)


# ═══════════════════════════════════════════════════════
# 回测引擎
# ═══════════════════════════════════════════════════════


class BacktestRunner:
    """封装 Nautilus Trader 的回测与 Paper Trading 能力.

    使用方式:
      runner = BacktestRunner(workspace_path)
      result = await runner.run_backtest(ohlcv_data, strategy_spec)
      session = await runner.start_paper(strategy_spec, api_key, api_secret)
      result = await runner.stop_paper(session_id)
    """

    def __init__(self, workspace_path: Path):
        self.workspace = workspace_path
        self._catalog_path = workspace_path / ".cache" / "backtest_catalog"
        self._catalog_path.mkdir(parents=True, exist_ok=True)
        self._paper_sessions: dict[str, PaperSession] = {}

    # ─────────────── 历史回测 ───────────────

    async def run_backtest(
        self,
        ohlcv_data: dict[str, pd.DataFrame],
        strategy_spec: StrategySpec,
        starting_balance: float = 100_000.0,
    ) -> BacktestResult:
        """运行历史回测.

        Args:
            ohlcv_data: {symbol: DataFrame} 格式的 OHLCV 数据.
            strategy_spec: 策略配置.
            starting_balance: 初始资金.

        Returns:
            BacktestResult 回测结果.
        """
        if not _NT_AVAILABLE:
            return await self._run_backtest_pandas(ohlcv_data, strategy_spec)

        try:
            return await asyncio.to_thread(
                self._run_backtest_nautilus,
                ohlcv_data,
                strategy_spec,
                starting_balance,
            )
        except Exception as e:
            logger.error(f"Nautilus backtest failed: {e}, falling back to pandas engine")
            return await self._run_backtest_pandas(ohlcv_data, strategy_spec)

    def _run_backtest_nautilus(
        self,
        ohlcv_data: dict[str, pd.DataFrame],
        strategy_spec: StrategySpec,
        starting_balance: float,
    ) -> BacktestResult:
        """使用 Nautilus Trader BacktestEngine 执行回测 (同步, 在线程中运行)."""
        from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
        from nautilus_trader.model import (
            BarType, InstrumentId, Venue, Money, Currency,
        )
        from nautilus_trader.model.enums import AccountType, OmsType
        from nautilus_trader.config import LoggingConfig
        from nautilus_trader.test_kit.providers import TestInstrumentProvider
        from nautilus_trader.trading.strategy import StrategyConfig
        from nautilus_trader.backtest.models import FillModel

        from getall.trading.backtest.strategies import (
            TemplateStrategy,
            TemplateStrategyConfig,
        )

        result = BacktestResult(
            strategy_name=strategy_spec.name,
            symbols=strategy_spec.symbols,
            timeframe=strategy_spec.timeframe,
        )

        # 1. 配置引擎
        engine = BacktestEngine(
            config=BacktestEngineConfig(
                logging=LoggingConfig(log_level="ERROR"),  # Reduce verbosity
            )
        )

        # 2. 添加模拟交易所 (使用 BINANCE 以匹配 TestInstrumentProvider 的 instruments)
        venue = Venue("BINANCE")
        engine.add_venue(
            venue=venue,
            oms_type=OmsType.NETTING,
            account_type=AccountType.MARGIN,
            base_currency=Currency.from_str("USDT"),
            starting_balances=[Money(starting_balance, Currency.from_str("USDT"))],
            fill_model=FillModel(),  # Use default fill model
        )

        # 3. 对每个 symbol 加载数据
        instrument = None
        for symbol, df in ohlcv_data.items():
            if df.empty:
                continue

            # 动态计算价格精度（根据实际数据）
            min_price = df[['open', 'high', 'low', 'close']].min().min()
            if min_price > 0:
                # 计算需要多少小数位才能精确表示价格
                import math
                price_precision = max(0, -int(math.floor(math.log10(min_price))) + 4)
                price_precision = min(price_precision, 10)  # Cap at 10 decimals
            else:
                price_precision = 8  # Default for crypto

            logger.debug(f"Auto-detected price precision for {symbol}: {price_precision} (min_price={min_price})")

            # 创建自定义 instrument 匹配价格精度
            from decimal import Decimal

            instrument_id = InstrumentId(
                symbol=Symbol("GENERIC"),
                venue=Venue("BINANCE")
            )

            # 提取基础货币和报价货币
            parts = symbol.split("/")
            base_ccy = parts[0] if len(parts) > 0 else "BTC"
            quote_ccy = parts[1].split(":")[0] if len(parts) > 1 else "USDT"

            # CryptoPerpetual 用于永续合约（不需要过期时间）
            instrument = CryptoPerpetual(
                instrument_id=instrument_id,
                raw_symbol=Symbol(symbol.replace("/", "").replace(":", "")),
                base_currency=Currency.from_str(base_ccy),
                quote_currency=Currency.from_str(quote_ccy),
                settlement_currency=Currency.from_str(quote_ccy),
                is_inverse=False,
                price_precision=price_precision,
                size_precision=3,
                price_increment=Price(Decimal(f"1e-{price_precision}"), price_precision),
                size_increment=Quantity(Decimal("0.001"), 3),
                max_quantity=Quantity(Decimal("1000000"), 3),
                min_quantity=Quantity(Decimal("0.001"), 3),
                max_price=Price(Decimal("1000000"), price_precision),
                min_price=Price(Decimal(f"1e-{price_precision}"), price_precision),
                margin_init=Decimal("1.00"),
                margin_maint=Decimal("0.35"),
                maker_fee=Decimal("0.0002"),
                taker_fee=Decimal("0.0004"),
                ts_event=0,
                ts_init=0,
            )

            engine.add_instrument(instrument)

            # 转换 bar 数据
            bar_type = BarType.from_str(f"{instrument.id}-1-MINUTE-LAST-EXTERNAL")

            # 记录时间范围
            if not df.empty:
                result.period_start = str(df.index[0])[:10]
                result.period_end = str(df.index[-1])[:10]
                delta = df.index[-1] - df.index[0]
                result.total_days = max(delta.days, 1)

                # 添加 bar 数据到引擎
                from nautilus_trader.core.datetime import dt_to_unix_nanos

                bars = []
                # Use dynamically calculated price precision instead of instrument precision
                # Size precision stays at 3 for crypto perps
                size_precision = 3

                for ts, row in df.iterrows():
                    # Round prices and volume to match precision
                    bar = Bar(
                        bar_type=bar_type,
                        open=Price(round(float(row["open"]), price_precision), price_precision),
                        high=Price(round(float(row["high"]), price_precision), price_precision),
                        low=Price(round(float(row["low"]), price_precision), price_precision),
                        close=Price(round(float(row["close"]), price_precision), price_precision),
                        volume=Quantity(round(float(row["volume"]), size_precision), size_precision),
                        ts_event=dt_to_unix_nanos(ts),
                        ts_init=dt_to_unix_nanos(ts),
                    )
                    bars.append(bar)

                engine.add_data(bars)
                logger.info(f"Added {len(bars)} bars to engine for {symbol}")

        if instrument is None:
            result.success = False
            result.error = "No valid data provided"
            return result

        # 4. 创建策略
        # For backtesting with ETH-USDT perpetual:
        # - Min size: 0.001 ETH
        # - Size precision: 3 decimals
        # - Use a reasonable fixed size for testing
        strategy = TemplateStrategy(
            config=TemplateStrategyConfig(
                instrument_id=instrument.id,
                strategy_spec_json=json.dumps(strategy_spec.to_json()),
                trade_size=1_000,  # 1000 = 1.000 ETH (precision 3)
            )
        )
        engine.add_strategy(strategy)

        # 5. 运行回测
        engine.run()

        # 6. 提取结果
        try:
            positions = engine.trader.generate_positions_report()
            orders = engine.trader.generate_order_fills_report()

            if len(positions) > 0:
                positions["pnl_numeric"] = positions["realized_pnl"].apply(
                    lambda x: float(str(x).replace(" USDT", "").replace(",", ""))
                    if isinstance(x, str) else float(x)
                )
                winners = positions[positions["pnl_numeric"] > 0]
                losers = positions[positions["pnl_numeric"] < 0]

                result.total_trades = len(positions)
                result.win_rate = len(winners) / len(positions) * 100 if len(positions) > 0 else 0
                result.total_pnl = positions["pnl_numeric"].sum()
                result.best_trade = positions["pnl_numeric"].max()
                result.worst_trade = positions["pnl_numeric"].min()

                if len(winners) > 0:
                    result.avg_win = winners["pnl_numeric"].mean()
                if len(losers) > 0:
                    result.avg_loss = losers["pnl_numeric"].mean()

                gross_profit = winners["pnl_numeric"].sum() if len(winners) > 0 else 0
                gross_loss = abs(losers["pnl_numeric"].sum()) if len(losers) > 0 else 0
                result.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

                # Expectancy
                if result.total_trades > 0:
                    result.expectancy = result.total_pnl / result.total_trades

                # Annualized return
                if result.total_days > 0:
                    result.annualized_return_pct = (result.total_pnl / starting_balance) * (365 / result.total_days) * 100

                # Sharpe (简化: 用每笔交易收益)
                if len(positions) > 1:
                    returns = positions["pnl_numeric"] / starting_balance
                    if returns.std() > 0:
                        result.sharpe_ratio = (returns.mean() / returns.std()) * (252 ** 0.5)

                result.positions_report = positions.to_string()
                result.orders_report = orders.to_string() if len(orders) > 0 else ""

                # 构建 equity_curve（类似 pandas 路径）
                equity_curve = self._build_equity_curve_from_positions(
                    positions=positions,
                    starting_balance=starting_balance,
                    period_start=result.period_start,
                )
                result.equity_curve = equity_curve

                # 更新最大回撤
                if equity_curve:
                    result.max_drawdown_pct = abs(min(p["drawdown"] for p in equity_curve))

        except Exception as e:
            logger.warning(f"Error extracting results: {e}")

        engine.dispose()
        return result

    def _build_equity_curve_from_positions(
        self,
        positions: pd.DataFrame,
        starting_balance: float,
        period_start: str,
    ) -> list[dict[str, Any]]:
        """从 Nautilus positions 构建 equity_curve.

        Args:
            positions: Nautilus positions report DataFrame (must have 'pnl_numeric' and 'ts_closed' columns)
            starting_balance: 初始资金
            period_start: 回测起始日期

        Returns:
            equity_curve 数据点列表
        """
        if len(positions) == 0:
            return []

        # 初始化
        equity = starting_balance
        peak_equity = starting_balance
        equity_points = [{"timestamp": period_start, "equity": equity, "drawdown": 0.0}]

        # 按时间排序 positions
        if "ts_closed" in positions.columns:
            positions = positions.sort_values("ts_closed")
        elif "ts_opened" in positions.columns:
            positions = positions.sort_values("ts_opened")

        # 遍历每个交易
        for _, pos in positions.iterrows():
            # 更新权益
            pnl = pos["pnl_numeric"]
            equity += pnl

            # 计算回撤
            if equity > peak_equity:
                peak_equity = equity
            drawdown = ((equity - peak_equity) / peak_equity * 100) if peak_equity > 0 else 0

            # 添加数据点
            timestamp = pos.get("ts_closed") or pos.get("ts_opened")
            if timestamp:
                equity_points.append({
                    "timestamp": str(timestamp),
                    "equity": equity,
                    "drawdown": drawdown,
                })

        return equity_points

    async def _run_backtest_pandas(
        self,
        ohlcv_data: dict[str, pd.DataFrame],
        strategy_spec: StrategySpec,
    ) -> BacktestResult:
        """纯 pandas 回测引擎 (Nautilus 不可用时的 fallback).

        使用 pandas-ta 计算指标, 逐行遍历模拟交易.
        """
        import pandas_ta as ta

        result = BacktestResult(
            strategy_name=strategy_spec.name,
            symbols=strategy_spec.symbols,
            timeframe=strategy_spec.timeframe,
        )

        all_trades: list[dict[str, Any]] = []

        for symbol, df in ohlcv_data.items():
            if df.empty:
                continue

            result.period_start = str(df.index[0])[:10]
            result.period_end = str(df.index[-1])[:10]
            delta = df.index[-1] - df.index[0]
            result.total_days = max(delta.days, 1)

            # 计算指标
            indicator_values: dict[str, pd.Series] = {}
            for ind_cfg in strategy_spec.indicators:
                name = ind_cfg["name"]
                params = ind_cfg.get("params", {})
                key = ind_cfg.get("key", name)

                try:
                    if name == "rsi":
                        indicator_values[key] = ta.rsi(df["close"], length=params.get("period", 14))
                    elif name == "sma":
                        indicator_values[key] = ta.sma(df["close"], length=params.get("period", 20))
                    elif name == "ema":
                        indicator_values[key] = ta.ema(df["close"], length=params.get("period", 20))
                    elif name == "macd":
                        macd_df = ta.macd(
                            df["close"],
                            fast=params.get("fast_period", 12),
                            slow=params.get("slow_period", 26),
                            signal=params.get("signal_period", 9),
                        )
                        if macd_df is not None:
                            indicator_values[f"{key}_value"] = macd_df.iloc[:, 0]
                            indicator_values[f"{key}_signal"] = macd_df.iloc[:, 1]
                            indicator_values[f"{key}_histogram"] = macd_df.iloc[:, 2]
                    elif name == "bollinger":
                        bb = ta.bbands(df["close"], length=params.get("period", 20), std=params.get("std", 2.0))
                        if bb is not None:
                            indicator_values[f"{key}_upper"] = bb.iloc[:, 0]
                            indicator_values[f"{key}_mid"] = bb.iloc[:, 1]
                            indicator_values[f"{key}_lower"] = bb.iloc[:, 2]
                    elif name == "atr":
                        indicator_values[key] = ta.atr(df["high"], df["low"], df["close"], length=params.get("period", 14))
                    elif name == "cci":
                        indicator_values[key] = ta.cci(df["high"], df["low"], df["close"], length=params.get("period", 20))
                    elif name == "stoch":
                        stoch_df = ta.stoch(df["high"], df["low"], df["close"],
                                            k=params.get("period_k", 14), d=params.get("period_d", 3))
                        if stoch_df is not None:
                            indicator_values[f"{key}_k"] = stoch_df.iloc[:, 0]
                            indicator_values[f"{key}_d"] = stoch_df.iloc[:, 1]
                    elif name == "roc":
                        indicator_values[key] = ta.roc(df["close"], length=params.get("period", 10))
                    else:
                        logger.warning(f"Pandas fallback: indicator {name} not supported, skipping")
                except Exception as e:
                    logger.warning(f"Error computing indicator {name}: {e}")

            # 逐行遍历模拟交易
            position_side: str | None = None  # "long" | "short"
            entry_price = 0.0
            entry_idx = 0

            for i in range(50, len(df)):  # 跳过前 50 根用于指标 warmup
                # 构建当前指标快照
                current_values: dict[str, float] = {}
                for key, series in indicator_values.items():
                    if series is not None and i < len(series) and pd.notna(series.iloc[i]):
                        current_values[key] = float(series.iloc[i])

                # 也把 OHLCV 放进快照, 方便条件里引用 "close"/"volume" 等
                price = float(df["close"].iloc[i])
                current_values["open"] = float(df["open"].iloc[i]) if "open" in df.columns else price
                current_values["high"] = float(df["high"].iloc[i]) if "high" in df.columns else price
                current_values["low"] = float(df["low"].iloc[i]) if "low" in df.columns else price
                current_values["close"] = price
                if "volume" in df.columns and pd.notna(df["volume"].iloc[i]):
                    current_values["volume"] = float(df["volume"].iloc[i])

                if position_side is not None:
                    # 检查止损止盈
                    exit_reason = ""
                    if strategy_spec.stop_loss_pct:
                        sl_price = entry_price * (1 - strategy_spec.stop_loss_pct / 100)
                        if position_side == "long" and price <= sl_price:
                            exit_reason = "stop_loss"
                        elif position_side == "short" and price >= entry_price * (1 + strategy_spec.stop_loss_pct / 100):
                            exit_reason = "stop_loss"

                    if not exit_reason and strategy_spec.take_profit_pct:
                        tp_price = entry_price * (1 + strategy_spec.take_profit_pct / 100)
                        if position_side == "long" and price >= tp_price:
                            exit_reason = "take_profit"
                        elif position_side == "short" and price <= entry_price * (1 - strategy_spec.take_profit_pct / 100):
                            exit_reason = "take_profit"

                    # 检查信号出场
                    if not exit_reason:
                        # exit_conditions: OR 逻辑
                        if strategy_spec.exit_conditions:
                            def _cond_dir(c: dict[str, Any]) -> str | None:
                                d = c.get("direction")
                                if d is None:
                                    return None
                                d = str(d).lower()
                                return d if d in ("long", "short") else None

                            for cond in strategy_spec.exit_conditions:
                                d = _cond_dir(cond)
                                if d is not None and d != position_side:
                                    continue
                                if self._eval_pandas_condition(cond, current_values):
                                    exit_reason = "signal"
                                    break

                    if exit_reason:
                        if position_side == "long":
                            pnl = (price - entry_price) / entry_price * 100
                        else:
                            pnl = (entry_price - price) / entry_price * 100

                        all_trades.append({
                            "symbol": symbol,
                            "side": position_side,
                            "entry_price": entry_price,
                            "exit_price": price,
                            "pnl_pct": pnl,
                            "exit_reason": exit_reason,
                            "hold_bars": i - entry_idx,
                            "exit_timestamp": df.index[i],  # 保存时间戳用于权益曲线
                        })
                        position_side = None

                else:
                    # 检查入场
                    def _cond_dir(c: dict[str, Any]) -> str | None:
                        d = c.get("direction")
                        if d is None:
                            return None
                        d = str(d).lower()
                        return d if d in ("long", "short") else None

                    def _all_met(conds: list[dict[str, Any]]) -> bool:
                        return bool(conds) and all(self._eval_pandas_condition(c, current_values) for c in conds)

                    entry_conds = strategy_spec.entry_conditions or []

                    # 默认单向: entry_conditions 是 AND
                    if strategy_spec.direction in ("long", "short"):
                        side = strategy_spec.direction
                        relevant = []
                        for c in entry_conds:
                            d = _cond_dir(c)
                            if d is not None and d != side:
                                continue
                            relevant.append(c)

                        if _all_met(relevant):
                            position_side = side
                            entry_price = price
                            entry_idx = i

                    # 双向: 如果条件里带 direction, 则按多/空分组分别 AND, 多空之间 OR
                    else:
                        long_specific = [c for c in entry_conds if _cond_dir(c) == "long"]
                        short_specific = [c for c in entry_conds if _cond_dir(c) == "short"]
                        neutral = [c for c in entry_conds if _cond_dir(c) is None]

                        # 保守策略: 没有显式 short 条件就不做空; 没有显式 long 条件则用 neutral 作为做多
                        long_enabled = bool(long_specific) or (not long_specific and not short_specific)
                        short_enabled = bool(short_specific)

                        long_conds = (neutral + long_specific) if long_specific else neutral
                        short_conds = (neutral + short_specific) if short_specific else []

                        long_signal = long_enabled and _all_met(long_conds)
                        short_signal = short_enabled and _all_met(short_conds)

                        if long_signal and not short_signal:
                            position_side = "long"
                            entry_price = price
                            entry_idx = i
                        elif short_signal and not long_signal:
                            position_side = "short"
                            entry_price = price
                            entry_idx = i
                        else:
                            # 两边同时触发 (罕见/冲突) -> 跳过
                            pass

        # 汇总结果
        if all_trades:
            pnls = [t["pnl_pct"] for t in all_trades]
            winners = [p for p in pnls if p > 0]
            losers = [p for p in pnls if p < 0]

            result.total_trades = len(pnls)
            result.win_rate = len(winners) / len(pnls) * 100
            result.total_pnl = sum(pnls)
            result.best_trade = max(pnls)
            result.worst_trade = min(pnls)

            if winners:
                result.avg_win = sum(winners) / len(winners)
            if losers:
                result.avg_loss = sum(losers) / len(losers)

            gross_profit = sum(winners) if winners else 0
            gross_loss = abs(sum(losers)) if losers else 0
            result.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

            if result.total_trades > 0:
                result.expectancy = result.total_pnl / result.total_trades

            if result.total_days > 0:
                result.annualized_return_pct = result.total_pnl * (365 / result.total_days)

            # Sharpe (简化)
            if len(pnls) > 1:
                mean_r = sum(pnls) / len(pnls)
                std_r = (sum((p - mean_r) ** 2 for p in pnls) / (len(pnls) - 1)) ** 0.5
                if std_r > 0:
                    result.sharpe_ratio = (mean_r / std_r) * (252 ** 0.5)

            # 平均持仓时间
            hold_bars = [t["hold_bars"] for t in all_trades]
            avg_bars = sum(hold_bars) / len(hold_bars) if hold_bars else 0
            tf_seconds = OHLCVLoader.timeframe_to_seconds(strategy_spec.timeframe)
            avg_hours = avg_bars * tf_seconds / 3600
            if avg_hours >= 24:
                result.avg_hold_time = f"{avg_hours / 24:.1f} days"
            else:
                result.avg_hold_time = f"{avg_hours:.1f} hours"

            # 构建权益曲线数据
            equity = 100.0  # 初始权益 100%
            peak_equity = 100.0
            equity_points = [{"timestamp": result.period_start, "equity": equity, "drawdown": 0.0}]

            for trade in all_trades:
                # 更新权益
                equity += trade["pnl_pct"]

                # 计算回撤
                if equity > peak_equity:
                    peak_equity = equity
                drawdown = ((equity - peak_equity) / peak_equity * 100) if peak_equity > 0 else 0

                # 添加数据点
                timestamp = trade.get("exit_timestamp")
                if timestamp:
                    equity_points.append({
                        "timestamp": str(timestamp),
                        "equity": equity,
                        "drawdown": drawdown,
                    })

            result.equity_curve = equity_points

            # 更新最大回撤
            if equity_points:
                result.max_drawdown_pct = abs(min(p["drawdown"] for p in equity_points))

        return result

    @staticmethod
    def _eval_pandas_condition(cond: dict[str, Any], values: dict[str, float]) -> bool:
        """评估单个条件 (pandas 引擎用)."""
        try:
            ind_key = cond["indicator"]
            ind_field = cond.get("field", "value")
            op = cond["operator"]
            threshold = cond["value"]

            # 获取当前值
            lookup_key = ind_key if ind_field == "value" else f"{ind_key}_{ind_field}"
            current = values.get(lookup_key)
            # 兼容多输出指标 (如 macd_value) 仍然用 field="value" 访问
            if current is None and ind_field == "value":
                current = values.get(f"{ind_key}_value")
            if current is None:
                return False

            # 阈值可能是另一个指标
            if isinstance(threshold, str) and "." in threshold:
                ref_key, ref_field = threshold.split(".", 1)
                ref_lookup = ref_key if ref_field == "value" else f"{ref_key}_{ref_field}"
                threshold = values.get(ref_lookup)
                if threshold is None and ref_field == "value":
                    threshold = values.get(f"{ref_key}_value")
                if threshold is None:
                    return False
            # 阈值也可能直接引用 OHLCV, 例如 "close"
            elif isinstance(threshold, str):
                if threshold in values:
                    threshold = values[threshold]

            threshold = float(threshold)

            if op == "lt":
                return current < threshold
            elif op == "gt":
                return current > threshold
            elif op == "lte":
                return current <= threshold
            elif op == "gte":
                return current >= threshold
            elif op == "eq":
                return abs(current - threshold) < 1e-8
            else:
                return False
        except Exception:
            return False

    # ─────────────── 自定义策略文件回测 ───────────────

    async def run_custom(
        self,
        strategy_file: str,
        ohlcv_data: dict[str, pd.DataFrame],
        starting_balance: float = 100_000.0,
    ) -> BacktestResult:
        """加载用户 .py 文件中的策略并运行回测.

        Args:
            strategy_file: 策略 .py 文件路径.
            ohlcv_data: OHLCV 数据.
            starting_balance: 初始资金.

        用户策略要求:
            - 必须继承 nautilus_trader.trading.strategy.Strategy
            - 如果有自定义 StrategyConfig，必须包含 instrument_id 字段
            - on_bar() 方法接收 bar 数据
        """
        if not _NT_AVAILABLE:
            return BacktestResult(
                success=False,
                error="nautilus_trader not installed. Custom strategy requires Nautilus Trader.",
            )

        strategy_path = Path(strategy_file)
        if not strategy_path.exists():
            return BacktestResult(success=False, error=f"Strategy file not found: {strategy_file}")

        try:
            # 动态导入策略模块
            module_name = f"user_strategy_{strategy_path.stem}_{uuid.uuid4().hex[:4]}"
            spec = importlib.util.spec_from_file_location(module_name, strategy_path)
            if spec is None or spec.loader is None:
                return BacktestResult(success=False, error=f"Cannot load module from {strategy_file}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

            # 查找 Strategy 子类和 StrategyConfig 子类
            from nautilus_trader.trading.strategy import Strategy as NTStrategy, StrategyConfig as NTConfig

            strategy_cls = None
            config_cls = None
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if isinstance(attr, type):
                    if issubclass(attr, NTStrategy) and attr is not NTStrategy:
                        strategy_cls = attr
                    if issubclass(attr, NTConfig) and attr is not NTConfig:
                        config_cls = attr

            if strategy_cls is None:
                return BacktestResult(success=False, error="No Strategy subclass found in file")

            logger.info(f"Loaded custom strategy: {strategy_cls.__name__} from {strategy_file}")

            # 在线程中运行同步回测引擎
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self._run_custom_nautilus,
                ohlcv_data,
                strategy_cls,
                config_cls,
                starting_balance,
            )
            return result

        except Exception as e:
            logger.exception(f"Failed to run custom strategy: {e}")
            return BacktestResult(success=False, error=f"Failed to run custom strategy: {e}")

    def _run_custom_nautilus(
        self,
        ohlcv_data: dict[str, pd.DataFrame],
        strategy_cls: type,
        config_cls: type | None,
        starting_balance: float,
    ) -> BacktestResult:
        """使用 Nautilus Trader BacktestEngine 执行自定义策略回测."""
        from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
        from nautilus_trader.model import BarType, Venue, Money, Currency
        from nautilus_trader.model.enums import AccountType, OmsType
        from nautilus_trader.config import LoggingConfig
        from nautilus_trader.test_kit.providers import TestInstrumentProvider
        from nautilus_trader.backtest.models import FillModel
        from nautilus_trader.model.data import Bar
        from nautilus_trader.model import Price, Quantity
        from nautilus_trader.core.datetime import dt_to_unix_nanos
        from nautilus_trader.trading.strategy import StrategyConfig as NTConfig

        result = BacktestResult(
            strategy_name=strategy_cls.__name__,
            symbols=list(ohlcv_data.keys()),
            timeframe="custom",
        )

        try:
            # 1. 配置引擎
            engine = BacktestEngine(
                config=BacktestEngineConfig(
                    logging=LoggingConfig(log_level="ERROR"),
                )
            )

            # 2. 添加模拟交易所
            venue = Venue("BINANCE")
            engine.add_venue(
                venue=venue,
                oms_type=OmsType.NETTING,
                account_type=AccountType.MARGIN,
                base_currency=Currency.from_str("USDT"),
                starting_balances=[Money(starting_balance, Currency.from_str("USDT"))],
                fill_model=FillModel(),
            )

            # 3. 添加 instrument 和数据
            instrument = TestInstrumentProvider.ethusdt_perp_binance()
            engine.add_instrument(instrument)

            for symbol, df in ohlcv_data.items():
                if df.empty:
                    continue

                bar_type = BarType.from_str(f"{instrument.id}-1-MINUTE-LAST-EXTERNAL")

                result.period_start = str(df.index[0])[:10]
                result.period_end = str(df.index[-1])[:10]
                delta = df.index[-1] - df.index[0]
                result.total_days = max(delta.days, 1)

                bars = []
                price_precision = instrument.price_precision
                size_precision = instrument.size_precision

                for ts, row in df.iterrows():
                    bar = Bar(
                        bar_type=bar_type,
                        open=Price(round(float(row["open"]), price_precision), price_precision),
                        high=Price(round(float(row["high"]), price_precision), price_precision),
                        low=Price(round(float(row["low"]), price_precision), price_precision),
                        close=Price(round(float(row["close"]), price_precision), price_precision),
                        volume=Quantity(round(float(row["volume"]), size_precision), size_precision),
                        ts_event=dt_to_unix_nanos(ts),
                        ts_init=dt_to_unix_nanos(ts),
                    )
                    bars.append(bar)

                engine.add_data(bars)
                logger.info(f"Added {len(bars)} bars to custom strategy engine for {symbol}")

            # 4. 创建策略实例
            if config_cls is not None:
                # 用户定义了 StrategyConfig 子类
                try:
                    config = config_cls(instrument_id=instrument.id)
                except TypeError:
                    # 可能 config_cls 有其他必需参数
                    config = config_cls()
                strategy = strategy_cls(config=config)
            else:
                # 使用默认的空 config
                strategy = strategy_cls()

            engine.add_strategy(strategy)
            logger.info(f"Running custom strategy: {strategy_cls.__name__}")

            # 5. 运行回测
            engine.run()

            # 6. 提取结果（复用 _run_backtest_nautilus 的逻辑）
            positions = engine.trader.generate_positions_report()
            orders = engine.trader.generate_order_fills_report()

            if len(positions) > 0:
                positions["pnl_numeric"] = positions["realized_pnl"].apply(
                    lambda x: float(str(x).replace(" USDT", "").replace(",", ""))
                    if isinstance(x, str) else float(x)
                )
                winners = positions[positions["pnl_numeric"] > 0]
                losers = positions[positions["pnl_numeric"] < 0]

                result.total_trades = len(positions)
                result.win_rate = len(winners) / len(positions) * 100 if len(positions) > 0 else 0
                result.total_pnl = positions["pnl_numeric"].sum()
                result.best_trade = positions["pnl_numeric"].max()
                result.worst_trade = positions["pnl_numeric"].min()

                if len(winners) > 0:
                    result.avg_win = winners["pnl_numeric"].mean()
                if len(losers) > 0:
                    result.avg_loss = losers["pnl_numeric"].mean()

                gross_profit = winners["pnl_numeric"].sum() if len(winners) > 0 else 0
                gross_loss = abs(losers["pnl_numeric"].sum()) if len(losers) > 0 else 0
                result.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

                if result.total_trades > 0:
                    result.expectancy = result.total_pnl / result.total_trades

                if result.total_days > 0:
                    result.annualized_return_pct = (result.total_pnl / starting_balance) * (365 / result.total_days) * 100

                if len(positions) > 1:
                    returns = positions["pnl_numeric"] / starting_balance
                    if returns.std() > 0:
                        result.sharpe_ratio = (returns.mean() / returns.std()) * (252 ** 0.5)

                result.positions_report = positions.to_string()
                result.orders_report = orders.to_string() if len(orders) > 0 else ""

                # 构建 equity_curve
                equity_curve = self._build_equity_curve_from_positions(
                    positions=positions,
                    starting_balance=starting_balance,
                    period_start=result.period_start,
                )
                result.equity_curve = equity_curve

                if equity_curve:
                    result.max_drawdown_pct = abs(min(p["drawdown"] for p in equity_curve))

            engine.dispose()
            result.success = True
            return result

        except Exception as e:
            logger.exception(f"Custom strategy backtest error: {e}")
            result.success = False
            result.error = str(e)
            return result

    # ─────────────── Paper Trading ───────────────

    async def start_paper(
        self,
        strategy_spec: StrategySpec,
        api_key: str = "",
        api_secret: str = "",
        exchange: str = "binance",
    ) -> PaperSession:
        """启动 Paper Trading 会话 (Binance Testnet).

        Args:
            strategy_spec: 策略配置.
            api_key: Binance testnet API key.
            api_secret: Binance testnet API secret.
            exchange: 交易所名称.

        Returns:
            PaperSession 对象.
        """
        session_id = str(uuid.uuid4())[:8]
        session = PaperSession(
            session_id=session_id,
            strategy_name=strategy_spec.name,
            symbols=strategy_spec.symbols,
            started_at=datetime.now(timezone.utc).isoformat(),
            config_json=json.dumps(strategy_spec.to_json()),
        )

        if not _NT_AVAILABLE:
            session.status = "simulated"
            self._paper_sessions[session_id] = session
            logger.info(f"Paper trading session created (simulated mode): {session_id}")
            return session

        try:
            from nautilus_trader.adapters.binance import BINANCE
            from nautilus_trader.adapters.binance import BinanceLiveDataClientFactory
            from nautilus_trader.adapters.binance import BinanceLiveExecClientFactory
            from nautilus_trader.live.node import TradingNode, TradingNodeConfig
            from nautilus_trader.config import LoggingConfig

            from getall.trading.backtest.strategies import (
                TemplateStrategy,
                TemplateStrategyConfig,
            )

            config = TradingNodeConfig(
                logging=LoggingConfig(log_level="INFO"),
                data_clients={
                    BINANCE: {
                        "api_key": api_key,
                        "api_secret": api_secret,
                        "account_type": "usdt_future",
                        "testnet": True,
                    },
                },
                exec_clients={
                    BINANCE: {
                        "api_key": api_key,
                        "api_secret": api_secret,
                        "account_type": "usdt_future",
                        "testnet": True,
                    },
                },
            )

            node = TradingNode(config=config)
            node.add_data_client_factory(BINANCE, BinanceLiveDataClientFactory)
            node.add_exec_client_factory(BINANCE, BinanceLiveExecClientFactory)
            node.build()

            # 注册策略
            inst_id_str = OHLCVLoader.symbol_to_nautilus_id(
                strategy_spec.symbols[0] if strategy_spec.symbols else "BTC/USDT",
                "BINANCE",
            )
            from nautilus_trader.model import InstrumentId
            strategy = TemplateStrategy(
                config=TemplateStrategyConfig(
                    instrument_id=InstrumentId.from_str(inst_id_str),
                    strategy_spec_json=json.dumps(strategy_spec.to_json()),
                )
            )
            node.trader.add_strategy(strategy)

            # 启动 (后台运行)
            task = asyncio.create_task(asyncio.to_thread(node.run))
            session._node = node
            session._task = task
            session.status = "running"

            logger.info(f"Paper trading started on Binance testnet: {session_id}")

        except ImportError as e:
            session.status = "simulated"
            logger.warning(f"Nautilus Binance adapter not available: {e}. Running in simulated mode.")
        except Exception as e:
            session.status = "error"
            session.stopped_at = datetime.now(timezone.utc).isoformat()
            logger.error(f"Failed to start paper trading: {e}")

        self._paper_sessions[session_id] = session
        return session

    async def stop_paper(self, session_id: str) -> BacktestResult:
        """停止 Paper Trading 会话并返回结果.

        Args:
            session_id: 会话 ID.

        Returns:
            BacktestResult 绩效结果.
        """
        session = self._paper_sessions.get(session_id)
        if session is None:
            return BacktestResult(success=False, error=f"Session not found: {session_id}")

        session.stopped_at = datetime.now(timezone.utc).isoformat()
        session.status = "stopped"

        result = BacktestResult(
            strategy_name=session.strategy_name,
            symbols=session.symbols,
            period_start=session.started_at[:10],
            period_end=session.stopped_at[:10],
        )

        # 如果有 TradingNode, 停止并提取结果
        if session._node is not None:
            try:
                node = session._node
                node.stop()

                # 提取交易结果
                positions = node.trader.generate_positions_report()
                orders = node.trader.generate_order_fills_report()

                if len(positions) > 0:
                    positions["pnl_numeric"] = positions["realized_pnl"].apply(
                        lambda x: float(str(x).replace(" USDT", "").replace(",", ""))
                        if isinstance(x, str) else float(x)
                    )
                    winners = positions[positions["pnl_numeric"] > 0]
                    losers = positions[positions["pnl_numeric"] < 0]

                    result.total_trades = len(positions)
                    result.win_rate = len(winners) / len(positions) * 100 if len(positions) > 0 else 0
                    result.total_pnl = positions["pnl_numeric"].sum()
                    result.positions_report = positions.to_string()

                node.dispose()
            except Exception as e:
                logger.error(f"Error stopping paper trading: {e}")
                result.error = str(e)

        # 取消后台 task
        if session._task is not None and not session._task.done():
            session._task.cancel()

        logger.info(f"Paper trading stopped: {session_id}")
        return result

    def get_paper_session(self, session_id: str) -> PaperSession | None:
        """获取 Paper Trading 会话信息."""
        return self._paper_sessions.get(session_id)

    def list_paper_sessions(self) -> list[PaperSession]:
        """列出所有 Paper Trading 会话."""
        return list(self._paper_sessions.values())

    @staticmethod
    def available_indicators() -> dict[str, dict[str, Any]]:
        """返回所有可用的 Nautilus 内置指标列表."""
        return NAUTILUS_INDICATORS
