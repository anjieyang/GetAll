---
name: morning-briefing
description: "Daily morning briefing / market briefing（晨报/早报/市场简报/盘前简报）. Generate a comprehensive market overview including price action, positions, anomalies, news/KOL highlights, and today's key events. Designed for Cron-based daily delivery (also works on-demand)."
metadata: '{"getall":{"always":false}}'
---

# Morning Briefing — Your Daily Market Summary

Generate a concise, information-dense daily briefing that replaces opening 10+ apps. Designed to be read in 30 seconds on Telegram.

## Activation

User says "send me a daily briefing at {time}" → Register Cron task:

```
cron(
  action="add",
  name="cron:morning-briefing",
  cron_expr="0 {hour} * * *",
  message="Generate and send the daily morning briefing: Follow the morning-briefing skill instructions. Include market overview, user positions, yesterday's anomalies, and today's events."
)
```

Can also be invoked on-demand: "give me a market briefing", "morning report"

## Data Collection (Minimize Tool Calls)

Use batch operations to gather all data efficiently. The entire briefing should require only **3-4 tool calls** total:

### Call 1: Multi-Price + Fear & Greed

```
market_data(action="multi_price", symbols="BTC/USDT,ETH/USDT,SOL/USDT,...")
```

Pass all coins from positions + watchlist. Returns prices + 24h changes for all in one call.

Also get: `market_data(action="fear_greed", symbol="BTC")`

### Call 2: Batch Technical Analysis (for watchlist/positions)

```
technical_analysis(action="batch_analysis", symbols="BTC/USDT,ETH/USDT,SOL/USDT", timeframe="4h")
```

Returns RSI, MACD, Bollinger, MA alignment for all coins. Use this to flag concerning technicals.

**Fallback**: If `technical_analysis` returns an error (e.g. missing dependency), **skip the technical section** and proceed with the rest of the briefing (do not block on TA).

### Call 3: Batch Sentiment (news + KOL for watched coins)

```
news_sentiment(action="batch_coin_sentiment", symbols="BTC,ETH,SOL", count=3)
```

Returns trending news + trending topics + per-coin KOL opinions in one call.

**Fallback**: If results are empty, retry once with:

`news_sentiment(action="recent", symbols="<all>", window_minutes=1440, mode="auto", prefer_non_empty=true)`

### Call 4 (if needed): Read memory files

Read `memory/trading/positions.json`, `memory/trading/anomalies.jsonl`, `PROFILE.md` for position data and yesterday's anomaly review.

---

## Briefing Sections

### Section 1: Market Overview

From Call 1 (multi_price) + fear_greed:

```
📈 Market: BTC ${price} ({24h_change}%) | ETH ${price} ({24h_change}%)
   Fear & Greed: {index} ({label})
   24h Liquidations: ${total} ({long_pct}% long / {short_pct}% short)
```

Include top 3 gainers and top 3 losers if notable (>10% move).

### Section 2: Your Positions

From Call 4 (positions.json) + Call 1 (current prices):

```
📋 Positions:
   Contract:
   • {SYMBOL} {SIDE} {amount} {lev}x | Entry ${entry} → Now ${current} | P&L {sign}${pnl} ({pnl_pct}%)
   Spot:
   • {SYMBOL}: {amount} | ${value} | 24h {change}%
   
   Total unrealized: {sign}${total_pnl}
   Contract exposure: {exposure_pct}% (limit: {limit}%)
```

Flag any positions with concerning metrics (liquidation close, high funding cost, large unrealized loss).

### Section 3: Watchlist Technical Snapshot

From Call 2 (batch_analysis) — for each watched coin NOT in positions:

```
👀 Watchlist:
   • {SYMBOL}: ${price} ({24h_change}%) | RSI {rsi} | MACD {signal} | {overall_signal}
```

Only include if there's something notable to report (big move, extreme indicator, signal change).

### Section 4: Yesterday's Anomalies Review

From Call 4 (anomalies.jsonl), filter to yesterday:

```
🚨 Yesterday's Anomalies:
   • {time} {SYMBOL}: {anomaly_type} — {brief description}
     → Outcome: price {moved direction} {pct}% since detection
```

This helps user see whether anomalies were predictive — builds trust in the monitoring system.

### Section 5: News & KOL Highlights

From Call 3 (batch_coin_sentiment):

```
📰 Key News:
   • {headline_1} ({source})
   • {headline_2} ({source})
   • KOL take: {notable KOL opinion on user's coins}
```

Focus on news relevant to user's positions and watchlist.

### Section 6: Today's Calendar

**Source**: `web_search` for crypto calendar events, or maintained in memory

```
📅 Today:
   • {time} US CPI data release — expect volatility
   • {SYMBOL} token unlock: {amount} tokens ({pct}% of supply)
   • Fed speaker {name} at {time}
```

If no major events: "No major scheduled events today."

### Section 7: Active Strategies Status

**Source**: Read `workspace/strategies/` for `status: active` strategies

```
🎯 Strategies:
   • {strategy_name}: monitoring {symbols} — {status: "no signal" or "signal triggered at {time}"}
```

## Output Format

Keep the entire briefing compact — it should fit in one or two Telegram screens:

```
☀️ GetAll Daily | {YYYY-MM-DD}

📈 BTC ${price} ({change}%) | ETH ${price} ({change}%) | F&G: {index} ({label})

📋 Your Positions:
   {position_lines}
   Unrealized: {sign}${total} | Exposure: {pct}%

🚨 Yesterday: {anomaly_count} anomalies detected
   {top anomaly if notable}

📰 {top_headline}
   KOL: {notable_opinion}

📅 {today_event or "No major events"}

🎯 Strategies: {active_count} active, {signal_status}
```

## Customization

- User can say "add {SYMBOL} to my briefing" → update watchlist
- User can say "skip the news section" → customize briefing format
- User can say "send briefing at 9am instead" → update Cron schedule
- Briefing language follows user preference from `PROFILE.md`
