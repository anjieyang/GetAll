"""通用模板策略: 根据 JSON 配置自动组装 Nautilus Trader Strategy.

支持两种模式:
  1. TemplateStrategy: 用 JSON 配置自动生成策略 (覆盖 80% 常见场景)
  2. 加载用户 .py 文件中的自定义策略 (ImportableStrategyConfig)

Nautilus 自带指标 (可在策略中直接使用):
  均线: SMA, EMA, DEMA, HMA, WMA, VWAP
  动量: RSI, MACD, Aroon, Bollinger Bands, CCI, Stochastics, ROC
  波动率: ATR, Donchian Channels, Keltner Channels
  比率: Efficiency Ratio, Spread Analyzer
  订单簿: Book Imbalance Ratio
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

# Nautilus 按需导入, 避免未安装时 import error
_NT_AVAILABLE = False
try:
    from nautilus_trader.indicators import (
        RelativeStrengthIndex as RSI,
        MovingAverageConvergenceDivergence as MACD,
        BollingerBands,
        SimpleMovingAverage as SMA,
        ExponentialMovingAverage as EMA,
        AverageTrueRange as ATR,
    )
    from nautilus_trader.model import (
        Bar,
        BarType,
        InstrumentId,
        Quantity,
        Price,
    )
    from nautilus_trader.model.enums import OrderSide
    from nautilus_trader.model.events import PositionOpened, PositionClosed
    from nautilus_trader.trading.strategy import Strategy, StrategyConfig
    from nautilus_trader.core.message import Event

    _NT_AVAILABLE = True
except ImportError:
    # 提供占位, 让模块可以被导入
    Strategy = object  # type: ignore[assignment, misc]
    StrategyConfig = object  # type: ignore[assignment, misc]
    logger.debug("nautilus_trader not installed — TemplateStrategy unavailable")


# ═══════════════════════════════════════════════════════
# 策略配置 (JSON-friendly)
# ═══════════════════════════════════════════════════════


@dataclass
class IndicatorConfig:
    """单个指标配置."""
    name: str                          # rsi, macd, bollinger, sma, ema, atr
    params: dict[str, Any] = field(default_factory=dict)  # {"period": 14}


@dataclass
class ConditionConfig:
    """单个条件配置."""
    indicator: str       # 指标名 (如 "rsi")
    field: str           # 指标字段 (如 "value")
    operator: str        # lt, gt, lte, gte, eq, cross_above, cross_below
    value: float | str   # 阈值 (数字) 或另一个 indicator.field (如 "macd.signal")


@dataclass
class StrategySpec:
    """完整策略描述 (JSON 可序列化)."""
    name: str = "template_strategy"
    symbols: list[str] = field(default_factory=list)       # ["BTC/USDT"]
    timeframe: str = "4h"
    indicators: list[dict[str, Any]] = field(default_factory=list)
    entry_conditions: list[dict[str, Any]] = field(default_factory=list)  # AND 逻辑
    exit_conditions: list[dict[str, Any]] = field(default_factory=list)   # OR 逻辑
    direction: str = "long"            # long / short / both
    trade_size_pct: float = 3.0        # 账户百分比
    stop_loss_pct: float | None = 5.0  # 止损百分比
    take_profit_pct: float | None = 15.0  # 止盈百分比
    leverage: int = 1

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "StrategySpec":
        """从 JSON dict 创建."""
        # 规范化 indicators 格式: 支持 dict 和 list 两种格式
        indicators_raw = data.get("indicators", [])
        if isinstance(indicators_raw, dict):
            # 转换 dict 格式 {"rsi": {"params": {...}}} 为 list 格式
            indicators = []
            for name, config in indicators_raw.items():
                if isinstance(config, dict):
                    indicators.append({
                        "name": name,
                        "params": config.get("params", {}),
                        "key": config.get("key", name),
                    })
                else:
                    # 简单格式: {"rsi": 14} -> {"name": "rsi", "params": {"period": 14}}
                    indicators.append({
                        "name": name,
                        "params": {"period": config} if isinstance(config, int) else {},
                        "key": name,
                    })
        else:
            indicators = indicators_raw

        return cls(
            name=data.get("name", "template_strategy"),
            symbols=data.get("symbols", []),
            timeframe=data.get("timeframe", "4h"),
            indicators=indicators,
            entry_conditions=data.get("entry_conditions", []),
            exit_conditions=data.get("exit_conditions", []),
            direction=data.get("direction", "long"),
            trade_size_pct=data.get("trade_size_pct", 3.0),
            stop_loss_pct=data.get("stop_loss_pct", 5.0),
            take_profit_pct=data.get("take_profit_pct", 15.0),
            leverage=data.get("leverage", 1),
        )

    def to_json(self) -> dict[str, Any]:
        """序列化为 JSON dict."""
        return {
            "name": self.name,
            "symbols": self.symbols,
            "timeframe": self.timeframe,
            "indicators": self.indicators,
            "entry_conditions": self.entry_conditions,
            "exit_conditions": self.exit_conditions,
            "direction": self.direction,
            "trade_size_pct": self.trade_size_pct,
            "stop_loss_pct": self.stop_loss_pct,
            "take_profit_pct": self.take_profit_pct,
            "leverage": self.leverage,
        }


# ═══════════════════════════════════════════════════════
# 指标工厂: 根据名称创建 Nautilus 指标实例
# ═══════════════════════════════════════════════════════

# 完整的 Nautilus 内置指标注册表
NAUTILUS_INDICATORS = {
    # 均线
    "sma": {"class": "SimpleMovingAverage", "default_params": {"period": 20}},
    "ema": {"class": "ExponentialMovingAverage", "default_params": {"period": 20}},
    "dema": {"class": "DoubleExponentialMovingAverage", "default_params": {"period": 20}},
    "hma": {"class": "HullMovingAverage", "default_params": {"period": 20}},
    "wma": {"class": "WeightedMovingAverage", "default_params": {"period": 20}},
    "vwap": {"class": "VolumeWeightedAveragePrice", "default_params": {}},
    # 动量
    "rsi": {"class": "RelativeStrengthIndex", "default_params": {"period": 14}},
    "macd": {"class": "MovingAverageConvergenceDivergence", "default_params": {"fast_period": 12, "slow_period": 26}},
    "bollinger": {"class": "BollingerBands", "default_params": {"period": 20, "k": 2}, "param_aliases": {"stddev": "k", "std_dev": "k"}},
    "cci": {"class": "CommodityChannelIndex", "default_params": {"period": 20}},
    "stoch": {"class": "Stochastics", "default_params": {"period_k": 14, "period_d": 3}},
    "roc": {"class": "RateOfChange", "default_params": {"period": 10}},
    "aroon": {"class": "AroonOscillator", "default_params": {"period": 25}},
    # 波动率
    "atr": {"class": "AverageTrueRange", "default_params": {"period": 14}},
    "donchian": {"class": "DonchianChannel", "default_params": {"period": 20}},
    "keltner": {"class": "KeltnerChannel", "default_params": {"period": 20}},
    # 比率
    "efficiency_ratio": {"class": "EfficiencyRatio", "default_params": {"period": 10}},
}


def create_indicator(name: str, params: dict[str, Any]) -> Any:
    """根据名称和参数创建 Nautilus 指标实例.

    Args:
        name: 指标名 (如 'rsi', 'macd', 'bollinger').
        params: 指标参数 (如 {'period': 14}).

    Returns:
        Nautilus 指标实例.
    """
    if not _NT_AVAILABLE:
        raise RuntimeError("nautilus_trader not installed")

    import nautilus_trader.indicators as nt_indicators

    info = NAUTILUS_INDICATORS.get(name.lower())
    if not info:
        raise ValueError(f"Unknown indicator: {name}. Available: {list(NAUTILUS_INDICATORS.keys())}")

    cls_name = info["class"]
    merged_params = {**info["default_params"], **params}

    # 处理参数别名映射（如 bollinger 的 stddev -> k）
    param_aliases = info.get("param_aliases", {})
    if param_aliases:
        for alias, real_name in param_aliases.items():
            if alias in merged_params:
                merged_params[real_name] = merged_params.pop(alias)

    cls = getattr(nt_indicators, cls_name, None)
    if cls is None:
        raise ValueError(f"Indicator class {cls_name} not found in nautilus_trader.indicators")

    return cls(**merged_params)


# ═══════════════════════════════════════════════════════
# Nautilus Strategy 配置与实现
# ═══════════════════════════════════════════════════════


if _NT_AVAILABLE:

    class TemplateStrategyConfig(StrategyConfig):
        """TemplateStrategy 的配置 (Nautilus-compatible)."""
        instrument_id: InstrumentId | None = None  # type: ignore[assignment]
        strategy_spec_json: str = "{}"   # JSON 序列化的 StrategySpec
        trade_size: int = 100_000

    class TemplateStrategy(Strategy):
        """通用模板策略: 根据 JSON 配置自动组装信号逻辑.

        支持的条件操作符:
          lt, gt, lte, gte, eq
          cross_above, cross_below (金叉/死叉)
        """

        def __init__(self, config: TemplateStrategyConfig):
            super().__init__(config=config)

            self.spec = StrategySpec.from_json(json.loads(config.strategy_spec_json))
            # Use precision 3 for crypto perpetual contracts (min 0.001)
            self.trade_size = Quantity(config.trade_size / 1000, 3)  # Convert from raw int to proper quantity

            # 动态注册指标
            self._indicators: dict[str, Any] = {}
            self._prev_values: dict[str, float] = {}  # 用于 cross 检测

            for ind_cfg in self.spec.indicators:
                name = ind_cfg["name"]
                params = ind_cfg.get("params", {})
                key = ind_cfg.get("key", name)  # 允许同名不同参数: rsi_14, rsi_7
                try:
                    indicator = create_indicator(name, params)
                    self._indicators[key] = indicator
                    self.register_indicator_for_bars(
                        BarType.from_str(
                            f"{config.instrument_id}-1-MINUTE-LAST-EXTERNAL"
                        ),
                        indicator,
                    )
                    logger.info(f"Registered indicator: {key} ({name}, {params})")
                except Exception as e:
                    logger.error(f"Failed to create indicator {name}: {e}")

            self._has_position = False

        def on_start(self) -> None:
            """策略启动: 订阅 bar 数据."""
            bar_type = BarType.from_str(
                f"{self.config.instrument_id}-1-MINUTE-LAST-EXTERNAL"
            )
            self.subscribe_bars(bar_type)
            logger.info(f"TemplateStrategy started: {self.spec.name}")

        def on_stop(self) -> None:
            """策略停止: 平仓."""
            self.close_all_positions(self.config.instrument_id)

        def on_bar(self, bar: Bar) -> None:
            """每根 K 线更新: 检查入场/出场条件."""
            # WORKAROUND: Manually update indicators since register_indicator_for_bars doesn't work properly
            # Update all indicators with the current bar
            for key, ind in self._indicators.items():
                if hasattr(ind, 'handle_bar'):
                    ind.handle_bar(bar)

            # 等待所有指标初始化完成
            if not all(ind.initialized for ind in self._indicators.values()):
                return


            # 检查出场 (OR 逻辑: 任一满足则平仓)
            if self._has_position and self._check_exit():
                self.close_all_positions(self.config.instrument_id)
                self._has_position = False
                return

            # 检查入场 (AND 逻辑: 全部满足才开仓)
            if not self._has_position and self._check_entry():
                side = OrderSide.BUY if self.spec.direction == "long" else OrderSide.SELL
                order = self.order_factory.market(
                    instrument_id=self.config.instrument_id,
                    order_side=side,
                    quantity=self.trade_size,
                )
                self.submit_order(order)
                self._has_position = True

            # 更新 prev 值 (用于 cross 检测)
            self._update_prev_values()

        def on_event(self, event: Event) -> None:
            """处理仓位事件."""
            from nautilus_trader.model.events import OrderRejected
            if isinstance(event, OrderRejected):
                logger.warning(f"Order rejected: {event.reason}")
                self._has_position = False
            elif isinstance(event, PositionOpened):
                self._has_position = True
            elif isinstance(event, PositionClosed):
                self._has_position = False

        # ── 条件判断引擎 ──

        def _check_entry(self) -> bool:
            """检查所有入场条件 (AND 逻辑)."""
            for i, cond in enumerate(self.spec.entry_conditions):
                result = self._evaluate_condition(cond)
                if not result:
                    return False
            return len(self.spec.entry_conditions) > 0

        def _check_exit(self) -> bool:
            """检查出场条件 (OR 逻辑)."""
            for cond in self.spec.exit_conditions:
                if self._evaluate_condition(cond):
                    return True
            return False

        def _evaluate_condition(self, cond: dict[str, Any]) -> bool:
            """评估单个条件.

            Args:
                cond: 条件 dict, 如:
                    {"indicator": "rsi", "field": "value", "operator": "lt", "value": 30}
                    {"indicator": "close", "field": "value", "operator": "lt", "value": "bollinger.lower"}
                    {"indicator": "bollinger", "field": "price", "operator": "lt", "value": "lower"}  # 简化写法
                    {"indicator": "macd", "field": "value", "operator": "cross_above", "value": "macd.signal"}
            """
            try:
                ind_key = cond["indicator"]
                ind_field = cond.get("field", "value")
                op = cond["operator"]
                threshold = cond["value"]

                # 特殊处理1：bollinger/donchian/keltner + "price" field -> 自动转换为 close 价格
                if ind_key in ["bollinger", "donchian", "keltner"] and ind_field == "price":
                    ind_key = "close"
                    ind_field = "value"

                # 特殊处理2：close/open/high/low/volume 作为 indicator，从 bar 数据获取
                if ind_key in ["close", "open", "high", "low", "volume"]:
                    # 获取最新 bar
                    bars = self.cache.bars(
                        BarType.from_str(f"{self.config.instrument_id}-1-MINUTE-LAST-EXTERNAL")
                    )
                    if not bars:
                        return False
                    last_bar = bars[-1]
                    current = float(getattr(last_bar, ind_key).as_double())
                else:
                    # 获取指标当前值
                    indicator = self._indicators.get(ind_key)
                    if indicator is None:
                        return False

                    current = getattr(indicator, ind_field, None)
                    if current is None:
                        current = indicator.value if hasattr(indicator, "value") else 0

                    # Nautilus RSI 返回 0-1 范围，需要转换为传统的 0-100 范围
                    if ind_key.lower() == "rsi" or "rsi" in ind_key.lower():
                        current = float(current) * 100

                # 处理阈值
                # 1. indicator.field 格式（如 "bollinger.lower"）
                if isinstance(threshold, str) and "." in threshold:
                    ref_key, ref_field = threshold.split(".", 1)
                    ref_ind = self._indicators.get(ref_key)
                    if ref_ind is None:
                        return False
                    threshold = getattr(ref_ind, ref_field, 0)
                # 2. 简化字段引用（如 "lower"，需要从原始条件中推断所属指标）
                elif isinstance(threshold, str) and threshold in ["lower", "upper", "middle", "signal"]:
                    # 从原始条件的 indicator 中查找（在 bollinger 条件简化写法中使用）
                    original_ind_key = cond["indicator"]
                    parent_ind = self._indicators.get(original_ind_key)
                    if parent_ind and hasattr(parent_ind, threshold):
                        threshold = getattr(parent_ind, threshold, 0)
                    else:
                        return False

                threshold = float(threshold)
                current = float(current)

                # 比较运算
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
                elif op == "cross_above":
                    prev_key = f"{ind_key}.{ind_field}"
                    prev = self._prev_values.get(prev_key, current)
                    return prev <= threshold < current
                elif op == "cross_below":
                    prev_key = f"{ind_key}.{ind_field}"
                    prev = self._prev_values.get(prev_key, current)
                    return prev >= threshold > current
                else:
                    logger.warning(f"Unknown operator: {op}")
                    return False

            except Exception as e:
                logger.warning(f"Condition evaluation error: {e}")
                return False

        def _update_prev_values(self) -> None:
            """保存当前值用于下一根 bar 的 cross 检测."""
            for key, ind in self._indicators.items():
                if hasattr(ind, "value"):
                    self._prev_values[f"{key}.value"] = float(ind.value)
                # MACD 特殊: 保存 signal line
                if hasattr(ind, "signal"):
                    self._prev_values[f"{key}.signal"] = float(ind.signal)

else:
    # Nautilus 未安装时的占位类
    class TemplateStrategyConfig:  # type: ignore[no-redef]
        pass

    class TemplateStrategy:  # type: ignore[no-redef]
        pass
