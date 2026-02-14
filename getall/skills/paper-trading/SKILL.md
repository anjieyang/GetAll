---
name: paper-trading
description: "Simulated live trading via cron-based monitoring. Use when user says 'paper trade this', 'test this strategy live', 'simulate this for a few days'."
metadata: '{"getall":{"always":false}}'
---

# Paper Trading — Cron-Based Live Simulation

Simulate live strategy execution by combining periodic market checks with the `backtest` tool on rolling recent data. No real money at risk.

## How It Works

Paper trading in GetAll uses **cron-scheduled market monitoring** rather than a dedicated paper trading engine. The agent checks live conditions on a schedule and records simulated entries/exits.

## Workflow

### 1. START

1. Parse the strategy (from STRATEGY.md or user description)
2. Set up a **rolling backtest** cron that:
   - Fetches recent OHLCV data
   - Runs the backtest tool on a short rolling window (e.g. 7d)
   - Compares with previous run to detect new trade signals
3. Set up monitoring cron:

```
reminders(action="add", every_seconds=3600, message="Paper trading '{name}': fetch latest data for {symbols}, run backtest(action='run', strategy_config='{...}', period='7d'). Compare with previous results. If new entry/exit signals detected, log to memory/trading/paper_trades.jsonl and notify user.")
```

### 2. MONITOR

When user asks "how is paper trading going?":
1. Read `memory/trading/paper_trades.jsonl`
2. Summarize: trades taken, running P&L, win/loss ratio
3. Compare with original backtest expectations

### 3. STOP

1. Remove cron tasks
2. Run a final summary of all paper trades
3. Compare with the historical backtest that validated the strategy
4. Provide recommendation: go live, adjust, or abandon

## Trade Logging Format

Append to `memory/trading/paper_trades.jsonl`:
```json
{"timestamp": "...", "strategy": "rsi_bounce", "symbol": "BTC/USDT", "action": "entry", "side": "long", "price": 65000, "reason": "RSI(14) = 28 < 30"}
```

## Limitations

- Not tick-level simulation — uses bar-level OHLCV data
- Execution assumes fills at close price — real slippage may differ
- Suitable for validating signal quality, not execution mechanics
