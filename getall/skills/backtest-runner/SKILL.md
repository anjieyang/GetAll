---
name: backtest-runner
description: "Run and interpret strategy backtests. Use when user says 'backtest', 'validate strategy', 'test this strategy historically', 'run backtest', or after creating a strategy via strategy-builder."
metadata: '{"getall":{"always":false}}'
---

# Backtest Runner — Strategy Validation Workflow

You have a `backtest` tool that returns **structured JSON metrics + a professional dashboard chart**. Your job is to interpret the data into actionable insight. Never dump raw JSON. Never draw your own charts — the tool generates a professional dashboard automatically.

## Hard Rules

1. **ALWAYS use the `backtest` tool** — never `exec` with matplotlib, never install external frameworks
2. **ALWAYS send the chart** — if `chart_path` exists, include it via `message(media=[chart_path])`. Never ask "want to see the chart?"
3. **ALWAYS lead with a verdict** — first sentence must be a clear pass/fail judgment
4. **ALWAYS compare vs benchmark** — use `benchmark_return_pct` and `excess_return_pct`
5. **ALWAYS end with one actionable suggestion** — what to try next

## Workflow

### Step 1: Parse Strategy → Build Config

Extract from user description or STRATEGY.md and build a JSON config:

```json
{
  "name": "strategy_name",
  "symbols": ["BTC/USDT"],
  "timeframe": "4h",
  "indicators": [{"name": "rsi", "params": {"period": 14}}],
  "entry_conditions": [{"indicator": "rsi", "field": "value", "operator": "lt", "value": 30}],
  "exit_conditions": [{"indicator": "rsi", "field": "value", "operator": "gt", "value": 70}],
  "direction": "long",
  "stop_loss_pct": 5,
  "take_profit_pct": 15,
  "trade_size_pct": 100
}
```

If critical info is missing, ask **one** clarifying question. Otherwise, fill reasonable defaults.

### Step 2: Call the Tool

```
backtest(action="run", strategy_config='{ ... }', period="6m", exchange="binance")
```

### Step 3: Interpret — Follow This Report Structure

Your text response MUST follow this structure (adapt wording to context):

```
📊 [Strategy Name] 回测结果 ([period] / [timeframe] / [symbols])

[VERDICT EMOJI] 结论: [one-sentence pass/fail judgment with key number]

┌─────────────┬──────────┬──────────┐
│ 指标        │ 策略     │ 基准     │
├─────────────┼──────────┼──────────┤
│ 总收益      │ X%       │ Y%       │
│ 年化收益    │ X%       │ -        │
│ 最大回撤    │ X%       │ -        │
│ Sharpe      │ X        │ -        │
│ 胜率        │ X%       │ -        │
│ 盈亏比      │ X        │ -        │
│ Profit Factor│ X       │ -        │
│ 交易次数    │ X        │ -        │
└─────────────┴──────────┴──────────┘

[WARNINGS if any — see quality flags below]

💡 建议: [one concrete, actionable next step]
```

Verdict emojis:
- 🟢 Strategy looks promising (PF > 1.5, excess return > 0, Sharpe > 1.0)
- 🟡 Mixed results, needs refinement (PF 1.0-1.5, or weak Sharpe)
- 🔴 Strategy fails (PF < 1.0, or negative excess return, or DD > 25%)

### Step 4: Send Chart + Text Together

```
message(content="[your analysis text above]", media=["[chart_path from metrics]"])
```

The dashboard chart has 4 panels: equity+benchmark, drawdown, monthly heatmap, metrics box. It speaks for itself — your text adds the *judgment* and *suggestion*.

## Quality Flags (always check, mention if triggered)

| Condition | Flag |
|---|---|
| total_trades < 30 | ⚠️ 样本量不足 (<30笔), 结论参考性有限 |
| win_rate_pct > 80 | ⚠️ 胜率异常高, 可能过拟合 |
| max_drawdown_pct > 25 | ⚠️ 回撤过大, 多数交易者无法承受 |
| excess_return_pct < 0 | ⚠️ 跑输持币不动, 策略不创造 alpha |
| sharpe_ratio < 0.5 | ⚠️ 风险调整收益差 |
| profit_factor < 1.0 | ⚠️ 策略在亏钱 (每赚1块要亏>1块) |
| max_consecutive_losses > 8 | ⚠️ 连续亏损过长, 心理承受力考验 |

## Iteration Suggestions (pick the most relevant ONE)

| Problem | Suggestion |
|---|---|
| Low win rate | 放宽入场阈值 or 加确认指标 |
| High drawdown | 收紧止损 or 减小仓位 |
| Few trades | 放宽条件 or 缩短 timeframe |
| Negative excess | 换方向 or 换策略类型 (趋势→均值回归) |
| Good results | 跑不同时段做 out-of-sample 验证 |

## Available Indicators

| Category | Names | Default |
|---|---|---|
| MA | sma, ema, dema, hma, wma | period=20 |
| Momentum | rsi, macd, roc, cci | period=14 |
| Bands | bollinger, stoch | period=20 |
| Volatility | atr | period=14 |

Multi-output field references: `macd.signal`, `bollinger.lower`, `stoch.k`

## Condition Operators

`lt`, `gt`, `lte`, `gte`, `eq`, `cross_above`, `cross_below`

Threshold: number (30) or indicator ref ("macd.signal", "bollinger.lower", "close")
