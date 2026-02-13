---
name: strategy-builder
description: "Strategy creation lifecycle. Create, validate, backtest, activate, pause, and review trading strategies stored as structured STRATEGY.md files."
metadata: '{"getall":{"always":false}}'
---

# Strategy Builder — Full Lifecycle Strategy Management

Create, manage, and operate trading strategies as structured, persistent skill files. Each strategy is a living document that evolves with backtesting results and real-world performance.

## When to Use

- User says "create a strategy for...", "build me a strategy"
- User wants to formalize a trading idea into a repeatable system
- User asks to activate, pause, or review an existing strategy
- KOL tracker produces a strategy that needs formalization

## Strategy File Format

All strategies are stored in `workspace/strategies/{strategy_name}/STRATEGY.md`:

```yaml
---
name: "strategy-name"
author: "user|agent|kol:{name}"
version: "1.0"
tags: [trend-following, low-risk, range-market]
symbols: [BTC, ETH, SOL]
timeframe: "4h"
status: "draft|active|paused|retired"
created: "YYYY-MM-DD"
updated: "YYYY-MM-DD"
cron_expr: "*/15 * * * *"
backtest:
  engine: "nautilus_trader"
  period: "YYYY-MM-DD ~ YYYY-MM-DD"
  win_rate: 0.00
  profit_factor: 0.0
  max_drawdown: "-0.0%"
  sharpe_ratio: 0.00
  total_trades: 0
risk:
  max_position_pct: 3
  max_leverage: 3
  stop_loss_pct: 5
  take_profit_pct: 15
---

## Strategy Logic
{Natural language description of the core idea}

## Entry Conditions
{All conditions that must be met — be specific with indicator values}

## Exit Conditions
{Stop-loss, take-profit, trailing stop, time-based exit rules}

## Position Sizing
{How much to allocate, scaling in/out rules}

## Suitable Market Conditions
{When this strategy works best — trending, ranging, high-vol, low-vol}

## Unsuitable Market Conditions
{When to avoid using this strategy}

## Monitoring Rules
{What to check when strategy is active — frequency, indicators to watch}
```

## Strategy Lifecycle

### 1. CREATE — Build the Strategy

When user describes a trading idea:

1. **Extract the core logic**: what triggers entry, what triggers exit
2. **Ask clarifying questions** if needed:
   - "What timeframe are you thinking?" (if not specified)
   - "Which coins should this apply to?"
   - "What's your max acceptable loss per trade?"
3. **Generate STRATEGY.md** following the template above
4. **Set status to `draft`**
5. **Suggest backtesting**: "Strategy created! Want me to backtest it before going live?"

### 2. BACKTEST — Validate with History

When user says "backtest this strategy":

1. Invoke the `backtest-runner` skill
2. Await results
3. **Update the `backtest:` section** in STRATEGY.md with actual results
4. **Provide assessment**:
   - Win rate > 55% and profit factor > 1.5 → "Looks promising"
   - Win rate < 40% or profit factor < 1.0 → "Needs improvement, consider adjusting parameters"
   - Max drawdown > 20% → "High drawdown risk, consider tighter stops"

### 3. ACTIVATE — Go Live

When user says "activate this strategy" or "start monitoring":

1. **Update status to `active`**
2. **Register a Cron task** using the `cron` tool:

```
cron(
  action="add",
  name="strategy:{strategy_name}",
  every_seconds={interval based on timeframe},
  message="Execute strategy '{strategy_name}': Read workspace/strategies/{name}/STRATEGY.md. Check entry conditions using market_data and technical_analysis tools for symbols {symbols}. If conditions are met, notify the user with the signal and suggested parameters. If not, stay silent."
)
```

3. **Confirm to user**: "Strategy '{name}' is now active. Checking every {interval} for {symbols}."

### 4. SIGNAL — Conditions Met

When the Cron task detects conditions are met:

1. Send notification:

```
📡 Strategy Signal: {strategy_name}

{SYMBOL} meets entry conditions:
{list each condition and its current value}

Suggested trade:
• Direction: {LONG/SHORT}
• Entry: ~${current_price}
• Stop-loss: ${sl} ({sl_pct}%)
• Take-profit: ${tp} ({tp_pct}%)
• Position size: {size}% of account
• Leverage: {lev}x

Want me to place this order? (先预演：模拟下单（Paper Trade）)
```

2. If user confirms → execute via `trade` tool with `模拟下单（Paper Trade）` first
3. Record in `trades.jsonl` with `source: "strategy:{name}"`

### 5. PAUSE — Temporarily Stop

When user says "pause this strategy":

1. **Update status to `paused`**
2. **Remove or disable the Cron task**: `cron(action="remove", job_id="{id}")`
3. Confirm: "Strategy '{name}' paused. Say 'resume {name}' to restart."

### 6. REVIEW — Performance Analysis

When user says "how is this strategy doing?" or "review strategy performance":

1. **Read `trades.jsonl`** filtering by `source: "strategy:{name}"`
2. **Compare actual vs backtest**:
   - Actual win rate vs backtest win rate
   - Actual profit factor vs backtest
   - Any divergence? Why?
3. **Suggest adjustments** if actual performance diverges significantly from backtest

### 7. RETIRE — Archive

When user says "retire this strategy" or it's been paused for 30+ days:

1. **Update status to `retired`**
2. **Ensure Cron is removed**
3. Keep the file for historical reference

## Best Practices for Strategy Creation

- **Be specific**: "RSI < 30" not "RSI is low"
- **Define all exit conditions**: never create a strategy without stop-loss rules
- **Include unsuitable conditions**: knowing when NOT to trade is as important as when to trade
- **Start conservative**: low leverage, small position sizes for new strategies
- **Backtest before activating**: always validate with historical data first
