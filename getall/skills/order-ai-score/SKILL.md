---
name: order-ai-score
description: "AI-powered position scoring. Evaluate a user's specific position with a comprehensive 0-10 AI reasoning score based on all available market data and position management quality."
metadata: '{"getall":{"always":false}}'
---

# Order AI Score — AI Reasoning-Based Position Evaluation

Score any position from 0 to 10 using pure LLM reasoning over all available data. No fixed weights — the AI considers context, market conditions, and position management holistically.

## When to Use

- User says "evaluate my {SYMBOL} position", "how's my trade?", "AI score"
- After user places an order (optional follow-up from position-monitor)
- User asks "rate this setup" with trade parameters
- Periodic position quality review

## Why No Fixed Weights

- Not every trade has complete data across all dimensions (technical / on-chain / sentiment)
- Different coins and market regimes make each factor's importance vary
- Mechanical weighting is less intelligent than LLM contextual reasoning
- A trade with perfect technicals but 20x leverage deserves a low score — weights can't capture this nuance

## Step 1: Collect All Available Information

Attempt to gather data using these tools. Mark each dimension as available or unavailable:

### 1A. Technical Analysis (`technical_analysis` tool)

```
action: full_analysis
symbol: {SYMBOL}
timeframe: "4h" (primary), also check "1h" and "1d"
```

Extract: RSI, MACD, Bollinger Bands, moving average alignment, candlestick patterns, support/resistance levels.

→ If successful: mark `"technical"` as available

### 1B. Derivatives / Contract Indicators (`market_data` tool)

Use batch_scan for efficiency — gets all derivative dimensions in one call:

```
market_data(action="batch_scan", symbols="{SYMBOL}", dimensions="oi,funding,long_short,net_long_short,taker,cvd,liquidations")
```

Or use individual calls if you need more detail on a specific dimension:
```
action: funding_rate → current funding rate and trend
action: long_short_ratio → crowd positioning
action: open_interest → OI changes (new money flow)
action: liquidations → recent liquidation events
action: cvd → cumulative volume delta (buyer/seller aggression)
action: net_long_short → net long vs net short positioning
action: taker_buy_sell → active buy/sell ratio
```

→ If successful: mark `"derivatives"` as available

### 1C. On-Chain / Whale Activity (`market_data` tool)

```
action: whale_transfers → large transfers in/out of exchanges
action: coin_flow → exchange net inflow/outflow
```

→ If successful: mark `"whale_transfer"` as available

### 1D. Sentiment / News (`news_sentiment` tool + `market_data` tool)

```
news_sentiment: action=kol_opinions, symbol={SYMBOL}
news_sentiment: action=trending_news
market_data: action=fear_greed
```

→ If successful: mark `"sentiment"` as available

**Note: Some dimensions may fail due to API unavailability or the coin not being supported. This is normal. Score based on what IS available.**

## Step 2: Read Position Information

From `positions.json` or user's description, extract:

- **Entry price** and **current price** (unrealized P&L)
- **Stop-loss** price and percentage
- **Take-profit** price and percentage (if set)
- **Leverage** multiplier
- **Position size** as % of total account
- **Liquidation price** and distance
- **Hold duration** (how long the position has been open)
- **Direction** (long / short)

## Step 3: AI Comprehensive Reasoning Score

With all collected information, reason through the following aspects:

### Market Context Assessment
- Is the current market environment favorable for this trade direction?
- Are technicals, derivatives, and sentiment aligned or conflicting?
- Are there any red flags (extreme funding, whale dumping, bearish divergence)?

### Position Management Assessment
- Is the leverage appropriate for the current volatility?
- Is the stop-loss set and at a reasonable level?
- Is the risk-reward ratio (SL distance vs TP distance) favorable?
- Is the position size prudent relative to account size?
- Is the liquidation distance safe?

### Scoring Guidelines

| Score | Meaning |
|-------|---------|
| **9-10** | Multiple dimensions strongly aligned + excellent position management (reasonable SL, conservative leverage, proper sizing). Rare — only give this for truly exceptional setups. |
| **7-8** | Overall positive signals + good position management. A solid trade. |
| **5-6** | Mixed signals or unclear direction. Or good signals but flawed position management. |
| **3-4** | Majority of signals unfavorable, OR serious position management issues (no SL, high leverage, oversized). |
| **1-2** | Strongly unfavorable signals AND dangerous position management. This trade is likely to lose money. |

## Step 4: Output Format

```
🎯 Order AI Score: {score}/10

📊 Data Sources Used:
• Technical: {status — key findings or "unavailable"}
• Derivatives: {status — key findings or "unavailable"}
• On-chain/Whale: {status — key findings or "unavailable"}
• Sentiment: {status — key findings or "unavailable"}

📋 Your Position:
• Direction: {LONG/SHORT}
• Leverage: {lev}x — {assessment}
• Stop-loss: ${sl} ({sl_pct}%) — {assessment}
• Take-profit: ${tp} (+{tp_pct}%) — {assessment}
• Position size: {size_pct}% of account — {assessment}
• Liquidation distance: {liq_dist}% — {assessment}
• Risk-reward ratio: {rr_ratio} — {assessment}

💡 Score Reasoning:
{Detailed paragraph explaining the reasoning — what supports the trade,
what goes against it, and how position management quality affects the score.
Be specific, reference actual data points.}

⚠️ Suggestions:
{Specific, actionable improvements — e.g., "Reduce leverage from 5x to 3x",
"Set stop-loss at $X", "Consider taking partial profit at $Y"}
```

## Step 5: Record to Memory

Append the score to `memory/trading/ai_scores.jsonl`:

```json
{
  "ts": "{ISO timestamp}",
  "symbol": "{SYMBOL}",
  "trade_id": "{matching trade ID from trades.jsonl if exists}",
  "score": {score},
  "inputs_available": ["technical", "derivatives", ...],
  "position": {
    "leverage": {lev},
    "stop_loss": {sl},
    "take_profit": {tp},
    "position_pct": {size_pct}
  },
  "reason": "{one-paragraph summary of reasoning}",
  "actual_1h": null,
  "actual_4h": null,
  "actual_24h": null
}
```

The `actual_*` fields are backfilled by Cron tasks later to evaluate AI Score accuracy over time.

## Quality Standards

- Never give a score above 7 if there's no stop-loss and leverage > 3x
- Never give a score above 8 if you only have 1 data dimension available
- Always explain what information was NOT available and how that affects confidence
- Be honest — a harsh but accurate score is more valuable than a feel-good number
