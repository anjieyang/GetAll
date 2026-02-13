"""技术分析工具 - RSI、MACD、布林带、均线、ATR、支撑阻力等指标计算 + 批量分析"""

import asyncio
import json
from typing import Any

import pandas as pd
try:
    import pandas_ta as ta  # type: ignore
except ImportError:  # pragma: no cover
    ta = None

from getall.agent.tools.base import Tool
from getall.trading.data.hub import DataHub


class TechnicalAnalysisTool(Tool):
    """基于 pandas-ta 计算各类技术指标，辅助 LLM 做交易分析决策。
    支持 batch_analysis 批量操作，一次 tool call 完成多币种综合技术面分析。
    """

    def __init__(self, hub: DataHub):
        # 通过 DataHub 获取 K 线数据
        self.hub = hub

    @property
    def name(self) -> str:
        return "technical_analysis"

    @property
    def description(self) -> str:
        return (
            "Calculate technical indicators (RSI, MACD, Bollinger Bands, "
            "MA, EMA, ATR, support/resistance) on crypto price data.\n"
            "Supports batch_analysis: run full_analysis on multiple symbols "
            "in one call (ideal for morning briefing and position review)."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "The technical indicator to calculate",
                    "enum": [
                        "batch_analysis",
                        "rsi", "macd", "bollinger", "ma", "ema",
                        "atr", "support_resistance", "full_analysis",
                    ],
                },
                "symbol": {
                    "type": "string",
                    "description": "Trading pair symbol, e.g. 'BTC/USDT'. Required for single-coin actions.",
                },
                "symbols": {
                    "type": "string",
                    "description": (
                        "Comma-separated symbols for batch_analysis, e.g. 'BTC/USDT,ETH/USDT,SOL/USDT'. "
                        "Required for batch_analysis."
                    ),
                },
                "timeframe": {
                    "type": "string",
                    "description": "Kline timeframe, e.g. '1h', '4h', '1d'. Default: '4h'",
                },
                "period": {
                    "type": "integer",
                    "description": "Indicator period / lookback length. Default varies by indicator.",
                    "minimum": 2,
                    "maximum": 500,
                },
            },
            "required": ["action"],
        }

    async def execute(self, **kwargs: Any) -> str:
        if ta is None:
            return (
                "Error: missing dependency 'pandas_ta'.\n"
                "Install it to use the technical_analysis tool."
            )

        action: str = kwargs["action"]
        symbol: str | None = kwargs.get("symbol")
        symbols_str: str | None = kwargs.get("symbols")
        timeframe: str = kwargs.get("timeframe", "4h")
        period: int | None = kwargs.get("period")

        try:
            # ── 批量操作 ──
            if action == "batch_analysis":
                return await self._batch_analysis(
                    symbols_str=symbols_str or "",
                    timeframe=timeframe,
                )

            # ── 单币操作 ──
            if not symbol:
                return "Error: 'symbol' is required for this action."

            # 拉取 K 线数据并转为 DataFrame
            df = await self._fetch_ohlcv(symbol, timeframe)
            if df.empty:
                return f"No kline data available for {symbol} ({timeframe})"

            # 分发到对应的指标计算函数
            handlers = {
                "rsi": self._rsi,
                "macd": self._macd,
                "bollinger": self._bollinger,
                "ma": self._ma,
                "ema": self._ema,
                "atr": self._atr,
                "support_resistance": self._support_resistance,
                "full_analysis": self._full_analysis,
            }
            handler = handlers.get(action)
            if handler is None:
                return f"Error: unknown action '{action}'"
            return handler(df, symbol, timeframe, period)
        except Exception as e:
            return f"Error calculating {action} for {symbol}: {e}"

    # ══════════════════════════════════════════════
    # 批量技术分析
    # ══════════════════════════════════════════════

    async def _batch_analysis(self, symbols_str: str, timeframe: str) -> str:
        """并发对多个币种执行 full_analysis, 一次 tool call 返回全部结果.

        适用于 morning-briefing、持仓技术面复查等场景。
        """
        symbols = [s.strip() for s in symbols_str.split(",") if s.strip()]
        if not symbols:
            return "Error: 'symbols' is required for batch_analysis, e.g. 'BTC/USDT,ETH/USDT,SOL/USDT'"

        sem = asyncio.Semaphore(4)  # 限制并发, 避免交易所限流
        results: dict[str, str] = {}

        async def _analyze(sym: str) -> None:
            async with sem:
                try:
                    df = await self._fetch_ohlcv(sym, timeframe)
                    if df.empty:
                        results[sym] = f"  ❌ No kline data available"
                    else:
                        results[sym] = self._full_analysis(df, sym, timeframe, None)
                except Exception as e:
                    results[sym] = f"  ❌ Error: {e}"

        await asyncio.gather(*[_analyze(s) for s in symbols])

        # 组装输出
        lines = [f"📊 Batch Technical Analysis ({len(symbols)} symbols, {timeframe})"]
        lines.append("═" * 60)
        for sym in symbols:
            lines.append(f"\n{results.get(sym, f'  ❌ {sym}: no result')}")
        lines.append(f"\n{'═' * 60}")
        lines.append(f"Symbols: {', '.join(symbols)} | Timeframe: {timeframe}")
        return "\n".join(lines)

    # ──────────────────────────────────────────────
    # K 线数据获取
    # ──────────────────────────────────────────────

    async def _fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 200) -> pd.DataFrame:
        """从交易所拉取 K 线并转换为 pandas DataFrame"""
        ohlcv = await self.hub.exchange.get_klines(symbol, timeframe, limit=limit)
        if isinstance(ohlcv, dict) and "error" in ohlcv:
            return pd.DataFrame()
        if not ohlcv:
            return pd.DataFrame()
        # get_klines 返回 list[dict], 每个 dict 有 timestamp/open/high/low/close/volume
        df = pd.DataFrame(ohlcv)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df.set_index("timestamp", inplace=True)
        # 确保数值类型
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    # ──────────────────────────────────────────────
    # 各指标计算
    # ──────────────────────────────────────────────

    def _rsi(self, df: pd.DataFrame, symbol: str, timeframe: str, period: int | None) -> str:
        """计算 RSI 指标"""
        p = period or 14
        rsi = ta.rsi(df["close"], length=p)
        current = rsi.iloc[-1]
        prev = rsi.iloc[-2]

        # 判断超买超卖状态
        if current > 70:
            signal = "OVERBOUGHT - potential reversal down"
        elif current < 30:
            signal = "OVERSOLD - potential reversal up"
        elif current > 60:
            signal = "Bullish momentum"
        elif current < 40:
            signal = "Bearish momentum"
        else:
            signal = "Neutral"

        return (
            f"📊 RSI({p}) for {symbol} ({timeframe})\n"
            f"{'─' * 40}\n"
            f"  Current RSI: {current:.2f}\n"
            f"  Previous RSI: {prev:.2f}\n"
            f"  Trend: {'Rising' if current > prev else 'Falling'}\n"
            f"  Signal: {signal}\n"
            f"  Price: {df['close'].iloc[-1]}"
        )

    def _macd(self, df: pd.DataFrame, symbol: str, timeframe: str, period: int | None) -> str:
        """计算 MACD 指标"""
        # MACD 默认参数: fast=12, slow=26, signal=9
        macd_df = ta.macd(df["close"], fast=12, slow=26, signal=9)
        macd_line = macd_df.iloc[-1, 0]   # MACD 线
        signal_line = macd_df.iloc[-1, 1]  # 信号线
        histogram = macd_df.iloc[-1, 2]    # 柱状图

        prev_hist = macd_df.iloc[-2, 2]

        # 判断金叉死叉
        if macd_line > signal_line and macd_df.iloc[-2, 0] <= macd_df.iloc[-2, 1]:
            cross = "GOLDEN CROSS (bullish)"
        elif macd_line < signal_line and macd_df.iloc[-2, 0] >= macd_df.iloc[-2, 1]:
            cross = "DEATH CROSS (bearish)"
        elif macd_line > signal_line:
            cross = "Above signal (bullish)"
        else:
            cross = "Below signal (bearish)"

        return (
            f"📊 MACD(12,26,9) for {symbol} ({timeframe})\n"
            f"{'─' * 40}\n"
            f"  MACD Line: {macd_line:.4f}\n"
            f"  Signal Line: {signal_line:.4f}\n"
            f"  Histogram: {histogram:.4f} ({'expanding' if abs(histogram) > abs(prev_hist) else 'contracting'})\n"
            f"  Cross: {cross}\n"
            f"  Price: {df['close'].iloc[-1]}"
        )

    def _bollinger(self, df: pd.DataFrame, symbol: str, timeframe: str, period: int | None) -> str:
        """计算布林带指标"""
        p = period or 20
        bbands = ta.bbands(df["close"], length=p, std=2.0)
        upper = bbands.iloc[-1, 0]   # 上轨
        mid = bbands.iloc[-1, 1]     # 中轨
        lower = bbands.iloc[-1, 2]   # 下轨
        bandwidth = bbands.iloc[-1, 3] if bbands.shape[1] > 3 else (upper - lower) / mid * 100
        price = df["close"].iloc[-1]

        # 判断价格相对布林带位置
        pct_b = (price - lower) / (upper - lower) * 100 if upper != lower else 50
        if price > upper:
            position = "ABOVE upper band (overbought / breakout)"
        elif price < lower:
            position = "BELOW lower band (oversold / breakdown)"
        elif price > mid:
            position = "Between middle and upper band (bullish)"
        else:
            position = "Between lower and middle band (bearish)"

        return (
            f"📊 Bollinger Bands({p}, 2σ) for {symbol} ({timeframe})\n"
            f"{'─' * 40}\n"
            f"  Upper Band: {upper:.4f}\n"
            f"  Middle Band: {mid:.4f}\n"
            f"  Lower Band: {lower:.4f}\n"
            f"  Bandwidth: {bandwidth:.2f}%\n"
            f"  %B: {pct_b:.1f}%\n"
            f"  Price: {price} → {position}"
        )

    def _ma(self, df: pd.DataFrame, symbol: str, timeframe: str, period: int | None) -> str:
        """计算简单移动平均线 (SMA)"""
        periods = [period] if period else [7, 25, 99]
        price = df["close"].iloc[-1]
        lines = [
            f"📊 SMA for {symbol} ({timeframe})",
            f"{'─' * 40}",
            f"  Current Price: {price}",
        ]
        for p in periods:
            sma = ta.sma(df["close"], length=p)
            val = sma.iloc[-1]
            diff_pct = (price - val) / val * 100 if val else 0
            direction = "above" if price > val else "below"
            lines.append(f"  SMA({p}): {val:.4f} (price {direction}, {diff_pct:+.2f}%)")

        # 多条均线时判断排列
        if len(periods) >= 3:
            sma_vals = [ta.sma(df["close"], length=p).iloc[-1] for p in sorted(periods)]
            if all(sma_vals[i] >= sma_vals[i + 1] for i in range(len(sma_vals) - 1)):
                lines.append("  Alignment: Bullish (short > medium > long)")
            elif all(sma_vals[i] <= sma_vals[i + 1] for i in range(len(sma_vals) - 1)):
                lines.append("  Alignment: Bearish (short < medium < long)")
            else:
                lines.append("  Alignment: Mixed / Transitioning")

        return "\n".join(lines)

    def _ema(self, df: pd.DataFrame, symbol: str, timeframe: str, period: int | None) -> str:
        """计算指数移动平均线 (EMA)"""
        periods = [period] if period else [9, 21, 55]
        price = df["close"].iloc[-1]
        lines = [
            f"📊 EMA for {symbol} ({timeframe})",
            f"{'─' * 40}",
            f"  Current Price: {price}",
        ]
        for p in periods:
            ema = ta.ema(df["close"], length=p)
            val = ema.iloc[-1]
            diff_pct = (price - val) / val * 100 if val else 0
            direction = "above" if price > val else "below"
            lines.append(f"  EMA({p}): {val:.4f} (price {direction}, {diff_pct:+.2f}%)")

        return "\n".join(lines)

    def _atr(self, df: pd.DataFrame, symbol: str, timeframe: str, period: int | None) -> str:
        """计算 ATR (平均真实波幅)"""
        p = period or 14
        atr = ta.atr(df["high"], df["low"], df["close"], length=p)
        current_atr = atr.iloc[-1]
        prev_atr = atr.iloc[-2]
        price = df["close"].iloc[-1]
        atr_pct = current_atr / price * 100 if price else 0

        return (
            f"📊 ATR({p}) for {symbol} ({timeframe})\n"
            f"{'─' * 40}\n"
            f"  Current ATR: {current_atr:.4f} ({atr_pct:.2f}% of price)\n"
            f"  Previous ATR: {prev_atr:.4f}\n"
            f"  Volatility: {'Increasing' if current_atr > prev_atr else 'Decreasing'}\n"
            f"  Suggested SL distance: {current_atr * 1.5:.4f} (1.5x ATR)\n"
            f"  Price: {price}"
        )

    def _support_resistance(self, df: pd.DataFrame, symbol: str, timeframe: str, period: int | None) -> str:
        """基于近期高低点计算支撑位和阻力位"""
        lookback = period or 50
        recent = df.tail(lookback)
        price = df["close"].iloc[-1]

        # 使用滑动窗口检测局部极值点
        highs = self._find_pivots(recent["high"], is_high=True)
        lows = self._find_pivots(recent["low"], is_high=False)

        # 筛选阻力位（价格上方）和支撑位（价格下方）
        resistance_levels = sorted([h for h in highs if h > price])[:3]
        support_levels = sorted([l for l in lows if l < price], reverse=True)[:3]

        lines = [
            f"📊 Support & Resistance for {symbol} ({timeframe})",
            f"{'─' * 40}",
            f"  Current Price: {price}",
            "",
            "  Resistance levels (above price):",
        ]
        if resistance_levels:
            for i, level in enumerate(resistance_levels, 1):
                dist = (level - price) / price * 100
                lines.append(f"    R{i}: {level:.4f} (+{dist:.2f}%)")
        else:
            lines.append("    No significant resistance found in range")

        lines.append("")
        lines.append("  Support levels (below price):")
        if support_levels:
            for i, level in enumerate(support_levels, 1):
                dist = (price - level) / price * 100
                lines.append(f"    S{i}: {level:.4f} (-{dist:.2f}%)")
        else:
            lines.append("    No significant support found in range")

        return "\n".join(lines)

    def _full_analysis(self, df: pd.DataFrame, symbol: str, timeframe: str, period: int | None) -> str:
        """综合技术分析：一次性计算所有关键指标"""
        price = df["close"].iloc[-1]
        prev_close = df["close"].iloc[-2]

        # RSI
        rsi = ta.rsi(df["close"], length=14)
        rsi_val = rsi.iloc[-1]

        # MACD
        macd_df = ta.macd(df["close"], fast=12, slow=26, signal=9)
        macd_line = macd_df.iloc[-1, 0]
        signal_line = macd_df.iloc[-1, 1]
        histogram = macd_df.iloc[-1, 2]

        # 布林带
        bbands = ta.bbands(df["close"], length=20, std=2.0)
        bb_upper = bbands.iloc[-1, 0]
        bb_mid = bbands.iloc[-1, 1]
        bb_lower = bbands.iloc[-1, 2]

        # ATR
        atr_val = ta.atr(df["high"], df["low"], df["close"], length=14).iloc[-1]

        # 均线
        sma_7 = ta.sma(df["close"], length=7).iloc[-1]
        sma_25 = ta.sma(df["close"], length=25).iloc[-1]
        sma_99 = ta.sma(df["close"], length=99).iloc[-1]
        ema_9 = ta.ema(df["close"], length=9).iloc[-1]
        ema_21 = ta.ema(df["close"], length=21).iloc[-1]

        # 综合信号判定
        bullish_signals = 0
        bearish_signals = 0

        if rsi_val < 30: bullish_signals += 1
        elif rsi_val > 70: bearish_signals += 1

        if macd_line > signal_line: bullish_signals += 1
        else: bearish_signals += 1

        if price > bb_mid: bullish_signals += 1
        else: bearish_signals += 1

        if sma_7 > sma_25 > sma_99: bullish_signals += 1
        elif sma_7 < sma_25 < sma_99: bearish_signals += 1

        if price > ema_21: bullish_signals += 1
        else: bearish_signals += 1

        # 总结信号
        if bullish_signals >= 4:
            overall = "STRONG BULLISH"
        elif bullish_signals >= 3:
            overall = "BULLISH"
        elif bearish_signals >= 4:
            overall = "STRONG BEARISH"
        elif bearish_signals >= 3:
            overall = "BEARISH"
        else:
            overall = "NEUTRAL / MIXED"

        return (
            f"📊 Full Technical Analysis: {symbol} ({timeframe})\n"
            f"{'═' * 50}\n"
            f"  Price: {price}  (prev: {prev_close}, chg: {(price - prev_close) / prev_close * 100:+.2f}%)\n"
            f"\n"
            f"  RSI(14): {rsi_val:.2f}  {'⚠ Overbought' if rsi_val > 70 else '⚠ Oversold' if rsi_val < 30 else ''}\n"
            f"  MACD: {macd_line:.4f} | Signal: {signal_line:.4f} | Hist: {histogram:.4f}\n"
            f"  BB: Upper={bb_upper:.4f} | Mid={bb_mid:.4f} | Lower={bb_lower:.4f}\n"
            f"  ATR(14): {atr_val:.4f} ({atr_val / price * 100:.2f}% of price)\n"
            f"\n"
            f"  SMA: 7={sma_7:.4f} | 25={sma_25:.4f} | 99={sma_99:.4f}\n"
            f"  EMA: 9={ema_9:.4f} | 21={ema_21:.4f}\n"
            f"\n"
            f"  Bullish signals: {bullish_signals}/5\n"
            f"  Bearish signals: {bearish_signals}/5\n"
            f"  ➤ Overall: {overall}"
        )

    # ──────────────────────────────────────────────
    # 工具方法
    # ──────────────────────────────────────────────

    @staticmethod
    def _find_pivots(series: pd.Series, is_high: bool, window: int = 5) -> list[float]:
        """使用滑动窗口检测局部极值点（支撑/阻力候选位）"""
        pivots: list[float] = []
        values = series.values
        for i in range(window, len(values) - window):
            segment = values[i - window: i + window + 1]
            if is_high and values[i] == segment.max():
                pivots.append(float(values[i]))
            elif not is_high and values[i] == segment.min():
                pivots.append(float(values[i]))
        return pivots
