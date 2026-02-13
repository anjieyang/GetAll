---
name: backtest-runner
description: "Backtesting with Nautilus Trader. Run historical simulations of trading strategies via the `backtest` tool, generate performance reports with win rate, Sharpe ratio, drawdown, and profit factor."
metadata: '{"getall":{"always":false}}'
---

# Backtest Runner — Strategy Validation via `backtest` Tool

Execute rigorous historical backtests for any trading strategy using the `backtest` tool. Supports both JSON-configured template strategies (80% of cases) and custom `.py` strategy files (complex scenarios).

## When to Use

- User says "backtest this strategy", "validate this strategy historically"
- After creating a new strategy (suggested by strategy-builder)
- User wants to compare strategy variants
- Periodic strategy health check (actual vs backtest performance)

## Quick Start

### Simple Strategy (JSON config → `run_backtest`)

For most strategies, use the `backtest` tool directly with a JSON configuration:

```
backtest(
  action="run_backtest",
  strategy_config='{
    "name": "rsi_oversold_bounce",
    "symbols": ["BTC/USDT"],
    "timeframe": "4h",
    "indicators": [
      {"name": "rsi", "params": {"period": 14}},
      {"name": "ema", "key": "ema_21", "params": {"period": 21}}
    ],
    "entry_conditions": [
      {"indicator": "rsi", "field": "value", "operator": "lt", "value": 30},
      {"indicator": "ema_21", "field": "value", "operator": "lt", "value": "close"}
    ],
    "exit_conditions": [
      {"indicator": "rsi", "field": "value", "operator": "gt", "value": 70}
    ],
    "direction": "long",
    "stop_loss_pct": 5,
    "take_profit_pct": 15,
    "leverage": 1
  }',
  period="6m"
)
```

### Complex Strategy (custom `.py` file → `run_custom`)

When the strategy requires custom factor computation, multi-asset correlation, or on-chain data:

1. **Write the strategy file** using `write_file` tool:

```python
# workspace/strategies/my_strat/strategy.py
from nautilus_trader.trading.strategy import Strategy, StrategyConfig
from nautilus_trader.indicators import RelativeStrengthIndex as RSI
from nautilus_trader.model import Bar, OrderSide, Quantity

class MyConfig(StrategyConfig):
    instrument_id: str = ""
    trade_size: int = 100_000

class MyStrategy(Strategy):
    def __init__(self, config: MyConfig):
        super().__init__(config=config)
        self.rsi = RSI(14)

    def on_start(self):
        self.subscribe_bars(...)

    def on_bar(self, bar: Bar):
        self.rsi.handle_bar(bar)
        if self.rsi.initialized and self.rsi.value < 30:
            order = self.order_factory.market(...)
            self.submit_order(order)
```

2. **Run backtest**:

```
backtest(
  action="run_custom",
  strategy_file="workspace/strategies/my_strat/strategy.py",
  strategy_config='{"symbols": ["BTC/USDT"], "timeframe": "4h"}',
  period="6m"
)
```

## Workflow

### Step 1: Parse the Strategy

Read the strategy source (STRATEGY.md file or user description) and extract:

| Parameter | Source | Example |
|-----------|--------|---------|
| Entry conditions | Strategy entry rules | RSI(14) < 30 AND volume > 2x avg |
| Exit conditions | Strategy exit rules | RSI > 60 OR price +8% OR SL -5% |
| Symbols | Strategy `symbols` field | [BTC/USDT, ETH/USDT] |
| Timeframe | Strategy `timeframe` field | 4h |
| Position sizing | Strategy `risk` section | 3% per trade, max 3x leverage |
| Stop-loss | Strategy exit rules | -5% from entry |
| Take-profit | Strategy exit rules | +15% from entry or RSI > 60 |

If any critical parameter is missing, ask the user before proceeding.

### Step 2: Build the JSON Strategy Config

Map the strategy to a JSON configuration for the `backtest` tool:

**Condition operators:**
- `lt` (less than), `gt` (greater than), `lte`, `gte`, `eq`
- `cross_above` (golden cross), `cross_below` (death cross)

**Indicator field references:**
- Simple: `{"indicator": "rsi", "field": "value", "operator": "lt", "value": 30}`
- Cross-reference: `{"indicator": "macd", "field": "value", "operator": "cross_above", "value": "macd.signal"}`

### Step 3: Call the `backtest` Tool

```
backtest(action="run_backtest", strategy_config='{...}', period="6m")
```

### Step 4: Interpret Results and Report

The tool returns a formatted report. Provide your assessment:
- Win rate > 55% and profit factor > 1.5 → "Looks promising"
- Win rate < 40% or profit factor < 1.0 → "Needs improvement"
- Max drawdown > 20% → "High drawdown risk, consider tighter stops"
- Total trades < 20 → "Low sample size, results may not be significant"

### Step 5: Update Strategy File

If the strategy has a STRATEGY.md file, update the `backtest:` section:

```yaml
backtest:
  engine: "nautilus_trader"
  period: "{start} ~ {end}"
  win_rate: {rate}
  profit_factor: {pf}
  max_drawdown: "{dd}%"
  sharpe_ratio: {sharpe}
  total_trades: {n}
```

## Available Indicators Reference

### Nautilus Trader Built-in (use directly in strategy)

| Category | Indicators | Default Params |
|----------|-----------|----------------|
| **Moving Averages** | sma, ema, dema, hma, wma, vwap | period=20 |
| **Momentum** | rsi, macd, bollinger, cci, stoch, roc, aroon | varies |
| **Volatility** | atr, donchian, keltner | period=14-20 |
| **Ratio** | efficiency_ratio | period=10 |
| **Order Book** | book_imbalance_ratio | — |

### pandas-ta Extended (pre-compute, feed as custom data)

130+ additional indicators including: Ichimoku Cloud, SuperTrend, VWMA, OBV, MFI, ADX, Williams %R, Chaikin, Fisher Transform, Squeeze, KAMA, TRIX, PPO, Vortex, etc.

### Decision Rule

- **Indicator exists in Nautilus** → use native (faster, event-driven, use in strategy JSON config)
- **Not in Nautilus** → pre-compute with `technical_analysis` tool or pandas-ta, then write custom `.py`
- **Complex composite factors** → Agent writes Python code to pre-compute, saves to workspace, then uses in custom strategy

Use `backtest(action="list_indicators")` to see the full list with default parameters.

## Quality Guidelines

- **Minimum sample size**: Warn if total trades < 20 — results may not be statistically significant
- **Overfitting risk**: If win rate > 80%, warn that the strategy may be overfit to historical data
- **Market regime**: Note if the backtest period was predominantly bull/bear/range — results may differ in other regimes
- **Suggest out-of-sample testing**: backtest on 70% of data, validate on remaining 30%
- **Multiple timeframes**: Consider running the same strategy on different periods to check robustness
