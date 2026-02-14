---
name: strategy-rsi-oversold
description: "RSI oversold bounce strategy template. Use when user asks about RSI bounce, oversold entry, or mean-reversion strategies."
metadata: '{"getall":{"always":false}}'
---

# Strategy Template: RSI Oversold Bounce

Mean-reversion long strategy: enter when RSI signals extreme oversold conditions, exit when price bounces to neutral territory.

## Backtest Config (ready to use)

```json
{
  "name": "rsi_oversold_bounce",
  "symbols": ["BTC/USDT"],
  "timeframe": "4h",
  "indicators": [
    {"name": "rsi", "params": {"period": 14}},
    {"name": "bollinger", "params": {"period": 20, "std": 2.0}}
  ],
  "entry_conditions": [
    {"indicator": "rsi", "field": "value", "operator": "lt", "value": 30},
    {"indicator": "close", "field": "value", "operator": "lt", "value": "bollinger.lower"}
  ],
  "exit_conditions": [
    {"indicator": "rsi", "field": "value", "operator": "gt", "value": 60}
  ],
  "direction": "long",
  "stop_loss_pct": 5,
  "take_profit_pct": 8,
  "trade_size_pct": 100
}
```

## Logic

When a coin is beaten down (RSI < 30 AND price below Bollinger lower band), enter long. Exit when RSI recovers above 60, or via stop-loss / take-profit.

## Risk

| Parameter | Value | Rationale |
|---|---|---|
| Max leverage | 3x | Mean-reversion is risky — keep low |
| Stop-loss | -5% | Tight enough to limit damage |
| Take-profit | +8% | Realistic bounce target |

## Suitable Conditions

- Range-bound / choppy markets
- Post-correction pullbacks in uptrends
- Individual coin oversold while BTC stable

## Unsuitable Conditions

- Strong bear trends (RSI can stay oversold for weeks)
- Market-wide panic (cascading liquidations)
- Before major macro events (CPI, Fed)

## Usage

1. **Quick backtest**: "Backtest RSI oversold bounce on BTC, 6 months"
2. **Customize**: "Change RSI threshold to 25" → agent adjusts config and re-runs
3. **Activate**: "Monitor this strategy for ETH and SOL" → agent sets up cron
