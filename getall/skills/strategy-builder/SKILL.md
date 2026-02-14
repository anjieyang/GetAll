---
name: strategy-builder
description: "Create, manage, and lifecycle trading strategies as STRATEGY.md files. Use when user says 'create a strategy', 'build me a strategy', 'activate/pause/review strategy'."
metadata: '{"getall":{"always":false}}'
---

# Strategy Builder — Full Lifecycle Management

Create, validate, and operate trading strategies as persistent STRATEGY.md files. Each strategy is a living document that evolves with backtest results and real-world performance.

## Strategy File Format

All strategies live in `workspace/strategies/{name}/STRATEGY.md`:

```yaml
---
name: "strategy-name"
author: "user|agent|kol:{name}"
version: "1.0"
symbols: [BTC/USDT, ETH/USDT]
timeframe: "4h"
status: "draft|active|paused|retired"
created: "YYYY-MM-DD"
updated: "YYYY-MM-DD"
cron_expr: "*/15 * * * *"
backtest:
  engine: "vectorbt"
  period: "YYYY-MM-DD ~ YYYY-MM-DD"
  total_return_pct: 0.0
  win_rate_pct: 0.0
  profit_factor: 0.0
  max_drawdown_pct: 0.0
  sharpe_ratio: 0.0
  total_trades: 0
risk:
  max_position_pct: 3
  max_leverage: 3
  stop_loss_pct: 5
  take_profit_pct: 15
---

## Strategy Logic
{Core idea in natural language}

## Entry Conditions
{Specific indicator values and thresholds}

## Exit Conditions
{Stop-loss, take-profit, signal-based exits}

## Suitable Market Conditions
{When this works — trending, ranging, high-vol}

## Unsuitable Market Conditions
{When to avoid}
```

## Lifecycle

### 1. CREATE (draft)

When user describes a trading idea:
1. Extract entry/exit logic, risk params, suitable assets
2. Ask ONE clarifying question if critical info is missing
3. Generate STRATEGY.md with `status: draft`
4. Suggest: "Strategy created. Want me to backtest it?"

### 2. BACKTEST (validate)

Invoke the `backtest-runner` skill (it handles tool calls and interpretation).
Update the `backtest:` section with results.

### 3. ACTIVATE (live monitoring)

1. Set `status: active`
2. Register a cron task:
```
reminders(action="add", every_seconds={interval}, message="Execute strategy '{name}': check entry conditions for {symbols} using market_data and technical_analysis. If conditions met, notify user with signal details.")
```

### 4. PAUSE / RETIRE

- Pause: set `status: paused`, remove cron
- Retire: set `status: retired`, remove cron, keep file for reference

### 5. REVIEW

Compare actual trade results (from `trades.jsonl`) vs backtest metrics.
Flag significant divergences.

## Available Strategy Templates

Pre-built strategy templates as starting points. User can say "show me strategy templates" or "start from the RSI template":

| Template | Style | Direction | Key Signal |
|----------|-------|-----------|------------|
| `strategy-rsi-oversold` | Mean-reversion | Long | RSI < 30 + price below Bollinger lower |
| `strategy-volume-breakout` | Momentum | Long | Price breaks resistance + volume 3x avg + OI rising |
| `strategy-bollinger-squeeze` | Volatility expansion | Long/Short | Band width at minimum + directional breakout |
| `strategy-panic-bottom` | Contrarian | Long | Fear & Greed < 20 + whale accumulation + liquidation cascade |
| `strategy-funding-reversal` | Contrarian | Short | Funding > 0.06% sustained + RSI > 75 + crowded longs |

To use a template: load the corresponding skill, extract its backtest config, and customize parameters.

## Best Practices

- Be specific: "RSI(14) < 30" not "RSI is low"
- Always define stop-loss — no strategy without risk management
- Start conservative: low leverage, small position sizes
- Backtest before activating — always
