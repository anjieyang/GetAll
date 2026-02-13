---
name: paper-trading
description: "Paper trading management. Start, monitor, and stop simulated live trading sessions on Binance Testnet using the `backtest` tool's paper trading capabilities."
metadata: '{"getall":{"always":false}}'
---

# Paper Trading — Live Simulation on Binance Testnet

Run trading strategies in real-time simulation using Binance Testnet. Same strategy code as backtesting, but connected to live market data — no real money at risk.

## When to Use

- User says "paper trade this strategy", "test this strategy live", "try this for a few days"
- After a backtest shows promising results and user wants real-time validation
- User wants to compare backtest performance with live execution
- Strategy has been idle and user wants to re-validate before going live

## Paper Trading Lifecycle

### 1. START — Launch Paper Trading

When user wants to paper trade a strategy:

1. **Parse the strategy** (from STRATEGY.md or user description)
2. **Build the strategy JSON config** (same format as backtest-runner)
3. **Start the paper session**:

```
backtest(
  action="start_paper",
  strategy_config='{
    "name": "rsi_bounce_paper",
    "symbols": ["BTC/USDT"],
    "timeframe": "4h",
    "indicators": [
      {"name": "rsi", "params": {"period": 14}}
    ],
    "entry_conditions": [
      {"indicator": "rsi", "field": "value", "operator": "lt", "value": 30}
    ],
    "exit_conditions": [
      {"indicator": "rsi", "field": "value", "operator": "gt", "value": 70}
    ],
    "direction": "long",
    "stop_loss_pct": 5,
    "take_profit_pct": 15
  }',
  api_key="BINANCE_TESTNET_API_KEY",
  api_secret="BINANCE_TESTNET_API_SECRET"
)
```

4. **Record the session ID** returned by the tool
5. **Confirm to user**: "Paper trading started! Session ID: {id}. Monitoring {symbols} on Binance Testnet."

### 2. SCHEDULE MONITORING — Set Up Cron Tasks

After starting paper trading, **always** set up monitoring:

**Periodic status check** (every few hours):
```
cron(
  action="add",
  every_seconds=14400,
  message="Check paper trading session {session_id}: call backtest(action='paper_status', session_id='{session_id}'). If there are new trades, briefly notify the user with entry/exit prices and P&L."
)
```

**End-of-period report** (when user specifies a duration):
```
cron(
  action="add",
  cron_expr="0 10 11 2 *",
  message="Paper trading session {session_id} has reached its scheduled end. Stop it: call backtest(action='stop_paper', session_id='{session_id}'). Generate a full performance report and compare with the historical backtest results. Notify the user with the comparison."
)
```

Tell the user: "I've set up automatic monitoring. You'll get updates every {interval} and a full report on {date}."

### 3. MONITOR — Check Status

When user asks "how is the paper trading going?" or periodically via cron:

```
backtest(action="paper_status", session_id="{session_id}")
```

Report to user:
- Current session status (running / stopped)
- Number of trades executed
- Current open positions
- Running P&L

### 4. STOP — End Session and Report

When user says "stop paper trading" or the scheduled end time arrives:

```
backtest(action="stop_paper", session_id="{session_id}")
```

The tool returns a full performance report. Additionally:

1. **Compare with backtest**: If the strategy was backtested before, compare:
   - Paper WR vs Backtest WR
   - Paper PF vs Backtest PF
   - Paper drawdown vs Backtest drawdown
   - Any significant divergence? Explain why.

2. **Provide recommendation**:
   - Results match backtest → "Strategy validated. Consider going live with small position."
   - Results significantly worse → "Live performance diverges from backtest. Possible causes: market regime change, execution differences. Consider adjusting parameters."
   - Results significantly better → "Caution: short sample period. Continue paper trading or go live with conservative sizing."

3. **Clean up cron tasks**: Remove the monitoring cron jobs.

```
cron(action="remove", job_id="{monitoring_job_id}")
cron(action="remove", job_id="{report_job_id}")
```

### 5. COMPARE — Backtest vs Paper vs Live

When user asks "compare my strategy results":

1. **Read backtest results** from STRATEGY.md `backtest:` section
2. **Read paper trading results** from the stopped session
3. **Read live trades** from `trades.jsonl` (if any, filtered by `source: "strategy:{name}"`)
4. **Generate comparison table**:

```
Strategy Validation: {strategy_name}
═══════════════════════════════════════
           | Backtest | Paper  | Live
───────────|----------|--------|──────
Win Rate   | 62%      | 58%    | 55%
PF         | 1.8      | 1.5    | 1.3
Max DD     | 8%       | 11%    | 9%
Sharpe     | 1.6      | 1.2    | 1.0
Trades     | 45       | 12     | 8
───────────|----------|--------|──────
Assessment: Performance degrades slightly from
backtest → paper → live (normal). Within
acceptable range. Strategy is viable.
═══════════════════════════════════════
```

## Setup Requirements

### Binance Testnet API Keys

Paper trading requires Binance Testnet credentials:

1. Go to https://testnet.binancefuture.com/
2. Create an account / login with GitHub
3. Generate API Key + Secret
4. Store in `workspace/exchanges.yaml` under `binance_testnet` section, or pass directly

If user doesn't have testnet keys:
- Inform them how to get them (link above)
- Offer to run in **simulated mode** (uses the backtest engine with recent data instead of live feed)

### Simulated Mode Fallback

If Nautilus Trader's Binance adapter is not available, paper trading runs in **simulated mode**:
- Uses the pandas-based backtest engine
- Fetches recent data via ccxt at regular intervals
- Provides a similar (but less precise) paper trading experience
- Clearly marked as "simulated" in status reports

## Best Practices

- **Always backtest first**: Don't skip to paper trading without historical validation
- **Run for sufficient duration**: At least 1 week for 4h strategies, 3+ days for 1h strategies
- **Monitor actively**: Paper trading can reveal issues that backtests miss (e.g., slippage, timing)
- **Compare consistently**: Use the same metrics and time periods for fair comparison
- **Document findings**: Update STRATEGY.md with paper trading observations
- **Don't over-optimize**: If paper results differ from backtest, understand why before changing parameters
