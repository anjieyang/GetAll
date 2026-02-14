---
name: kol-tracker
description: "Learn from KOL trading patterns. Analyze crypto KOLs' trading styles, historical call accuracy, and extract their strategies into reusable templates via Followin data. Use when user says 'learn from @{name}', 'study this KOL', '学习/分析这个KOL', 'track KOL calls', or wants to replicate a KOL's approach."
metadata: '{"getall":{"always":false}}'
---

# KOL Tracker — Learn & Extract Trading Strategies from Key Opinion Leaders

Study specific crypto KOLs' trading patterns, verify their historical accuracy, and distill their approach into actionable, backtestable strategies.

## When to Use

- User says "learn from @{kol_name}", "study {kol}'s trading style"
- User says "what does {kol} think about {SYMBOL}?"
- User says "track {kol}'s calls and check accuracy"
- User wants to replicate a KOL's approach systematically

## Data Sources

### Primary: Followin API (`news_sentiment` tool)

```
news_sentiment(action="kol_opinions", symbol={SYMBOL})
```

- KOL opinions on specific coins
- Identifies which KOLs are talking about what, with timestamps
- Available for all coins covered by Followin

### Secondary: Web Search (`web_search` tool)

- Search for the KOL's Twitter/X posts, blog articles, interviews
- Look for Hyperliquid leaderboard profiles if applicable
- Find any published trading records or performance claims

### Tertiary: Web Fetch (`web_fetch` tool)

- Fetch specific URLs discovered during search
- Read detailed articles or threads about the KOL's approach

## Analysis Workflow

### Step 1: Data Collection

1. **Followin KOL data**: Search for their opinions across major coins

```
For each major coin (BTC, ETH, SOL, and user's watchlist):
  news_sentiment(action="kol_opinions", symbol={COIN})
  → Filter for the target KOL's posts
```

2. **Web search**: `web_search("{kol_name} crypto trading strategy site:twitter.com OR site:x.com")`
3. **Web search**: `web_search("{kol_name} crypto performance track record")`
4. **Fetch content** from relevant URLs found

### Step 2: Trading Style Analysis

From collected data, extract and analyze:

| Dimension | What to Look For |
|-----------|-----------------|
| **Preferred coins** | Which coins do they trade most? Major only or altcoins too? |
| **Timeframe** | Scalping (minutes), swing (hours-days), or position (weeks-months)? |
| **Entry style** | Technical analysis driven? News/catalyst driven? Contrarian? |
| **Position management** | Heavy conviction bets or diversified small positions? |
| **Risk management** | Do they mention stop-losses? How tight? Do they cut losses quickly? |
| **Market conditions** | Do they trade in all conditions or selectively? |
| **Edge** | What seems to be their unique insight or advantage? |

### Step 3: Call Verification (If Possible)

If historical calls can be identified (from Followin timestamps or Twitter posts):

1. **List identifiable calls** with timestamps
2. **Use `market_data` tool** to check what actually happened after each call:

```
market_data(action="klines", symbol={SYMBOL}, timeframe="4h")
→ Check price movement after the call timestamp
```

3. **Calculate accuracy**:
   - Calls that moved in the predicted direction within 24h/48h/1week
   - Average return if followed
   - Worst case scenario if followed

4. **Present results honestly** — even if the KOL's accuracy is low

### Step 4: Strategy Extraction

Transform the KOL's approach into a structured strategy:

1. **Identify the core pattern**: What conditions tend to precede their calls?
2. **Formalize entry conditions**: Translate from narrative to indicator-based rules
3. **Define risk management**: Based on their stated or observed SL/TP patterns
4. **Write STRATEGY.md** using the `strategy-builder` format

```
workspace/strategies/{kol_name}_style/STRATEGY.md
```

With metadata:
```yaml
author: "kol:{kol_name}"
tags: [{kol_name}, extracted, {style_tags}]
```

### Step 5: Output Report

```
📝 KOL Analysis: @{kol_name}

## Profile
• Main coins: {coins}
• Trading style: {swing/scalp/position}
• Entry approach: {technical/catalyst/contrarian}
• Risk style: {aggressive/moderate/conservative}

## Historical Accuracy (if verifiable)
• Identifiable calls: {n}
• Directionally correct (24h): {pct}%
• Average return if followed: {pct}%
• Best call: {description} ({return}%)
• Worst call: {description} ({return}%)

## Extracted Strategy
Name: "{kol_name}_style"
Core logic: {one-sentence summary}
Entry: {conditions}
Exit: {conditions}
Risk: {parameters}

Saved to: workspace/strategies/{kol_name}_style/STRATEGY.md

## Next Steps
• [Backtest] Run historical simulation
• [Activate] Start monitoring for signals
• [Track] Set up periodic check for new posts from this KOL
```

### Step 6: Optional — Ongoing Monitoring

If user wants to keep tracking this KOL:

```
cron(
  action="add",
  name="kol:{kol_name}",
  every_seconds=3600,
  message="Check for new opinions from @{kol_name} via news_sentiment tool (kol_opinions for user's watchlist coins). If new noteworthy opinions found, notify user."
)
```

## Quality Guidelines

- **Be balanced**: Present both positive and negative findings about the KOL
- **Don't overfit**: A KOL who was right 3 times might just be lucky — mention sample size
- **Separate opinion from data**: Clearly distinguish between what the data shows and what you're interpreting
- **Respect that KOLs can be wrong**: A strategy extracted from a KOL still needs backtesting
- **Update over time**: If user tracks a KOL long-term, periodically update the accuracy assessment
