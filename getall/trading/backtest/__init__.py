# 回测模块: Nautilus Trader 集成
# 提供历史回测 + Paper Trading 能力

from getall.trading.backtest.engine import BacktestRunner, BacktestResult
from getall.trading.backtest.strategies import TemplateStrategy, TemplateStrategyConfig
from getall.trading.backtest.data_loader import OHLCVLoader

__all__ = [
    "BacktestRunner",
    "BacktestResult",
    "TemplateStrategy",
    "TemplateStrategyConfig",
    "OHLCVLoader",
]
