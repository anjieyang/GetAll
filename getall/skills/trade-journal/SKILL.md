---
name: trade-journal
description: "Trade recording and review. Log trades to trades.jsonl, generate review reports, track win rates, discover patterns, and extract lessons. Use when user says '记录交易', 'review this week/month', '复盘', 'how did I do', or reports opening/closing a trade."
metadata: '{"getall":{"always":false}}'
---

# Trade Journal — Record, Review, Learn

Systematically record every trade, generate data-driven reviews, and help the user learn from their own trading history.

## When to Use

- User tells you about a trade: "I went long on BTC at 95k", "I bought some SOL"
- User closes a trade: "closed my BTC position", "took profit on SOL"
- User asks for review: "review this week", "how did I do this month?"
- Automatic: position-monitor detects open/close events

## Recording Trades

### User Reports Opening a Trade

When user says "I went long/short on {SYMBOL}" or "I bought {SYMBOL}":

1. **Parse the trade details**: symbol, direction, entry price, amount, leverage, stop-loss, take-profit
2. **Ask for missing critical info** (don't assume):
   - "What leverage?" (if futures)
   - "Entry price?" (if not mentioned, use current price via `market_data`)
   - "Did you set a stop-loss?"
3. **Append to `memory/trading/trades.jsonl`**:

```json
{
  "ts": "{ISO timestamp}",
  "id": "{unique trade ID}",
  "symbol": "{SYMBOL}",
  "exchange": "{exchange}",
  "side": "buy|sell",
  "type": "market|limit",
  "price": {entry_price},
  "amount": {amount},
  "leverage": {leverage},
  "stop_loss": {sl_price or null},
  "take_profit": {tp_price or null},
  "source": "user_manual|bot|strategy:{name}",
  "ai_score": null,
  "ai_score_reason": null,
  "status": "open",
  "close_ts": null,
  "close_price": null,
  "pnl": null,
  "pnl_pct": null,
  "notes": "{any context user provided}"
}
```

4. **If no stop-loss mentioned** → remind them to set one
5. **Offer AI Score**: "Want me to evaluate this position?"

### User Reports Closing a Trade

When user says "closed my {SYMBOL}" or "took profit/stopped out":

1. **Find the matching open trade** in `trades.jsonl` (match by symbol + status="open")
2. **Get close price**: from user's input, or fetch via `market_data` tool
3. **Calculate P&L**:
   - P&L = (close_price - entry_price) × amount × (1 if long, -1 if short)
   - P&L % = (close_price / entry_price - 1) × leverage × 100 × direction
4. **Update the record** in `trades.jsonl`: set `status: "closed"`, fill `close_ts`, `close_price`, `pnl`, `pnl_pct`
5. **Respond with summary**:

```
Trade closed:
{SYMBOL} {SIDE} | Entry ${entry} → Exit ${close}
P&L: {sign}${pnl} ({pnl_pct}%)
Duration: {hold_time}

{brief one-line assessment — e.g., "Clean trade with good risk management" or "Stopped out early, but the SL saved you from a bigger drop"}

Want to record any notes or thoughts about this trade?
```

## Generating Review Reports

### Weekly / Monthly Review

When user says "review this week" or "how did I do this month":

1. **Read `trades.jsonl`** and filter by requested time period
2. **Calculate statistics**:

| Metric | Formula |
|--------|---------|
| Total trades | Count of closed trades in period |
| Win rate | Winners / total × 100% |
| Average win | Mean P&L of winning trades |
| Average loss | Mean P&L of losing trades |
| Profit factor | Total wins / Total losses |
| Largest win | Max single trade P&L |
| Largest loss | Min single trade P&L |
| Average hold time | Mean duration of closed trades |
| Net P&L | Sum of all P&L |

3. **Break down by category**:
   - By symbol: which coins were profitable vs not
   - By direction: long vs short performance
   - By time of day: performance by trading hour
   - By source: manual vs bot vs strategy trades
   - By leverage: high-leverage vs low-leverage performance

4. **AI Score correlation**:
   - Trades with AI Score >= 7: win rate and avg P&L
   - Trades with AI Score <= 4: win rate and avg P&L
   - Does AI Score predict outcomes for this user?

5. **Pattern discovery**:
   - Any new high-win-rate patterns? → Update `PATTERNS.md`
   - Any repeated mistakes? → Update `LESSONS.md`
   - Any notable behavioral patterns? (time-of-day, revenge trading, etc.)

6. **Output format**:

```
📊 Trading Review: {period}

## Overview
Trades: {total} | Wins: {wins} | Losses: {losses}
Win rate: {win_rate}%
Net P&L: {sign}${net_pnl}
Profit factor: {pf}

## What Worked ✅
{Top performing patterns, setups, or coins}

## What Didn't ❌
{Losing patterns, common mistakes}

## AI Score Review
Score ≥7 trades: {n} trades, {win_rate}% win rate, avg P&L {avg}
Score ≤4 trades: {n} trades, {win_rate}% win rate, avg P&L {avg}

## Lessons Learned
{New lessons to add to LESSONS.md — if any}

## Patterns Discovered
{New high-win-rate setups to add to PATTERNS.md — if any}

## Recommendations
{Specific actionable suggestions for improvement}
```

## Daily Review (Cron-Triggered)

`cron:daily-review` triggers a daily review at the configured time:

1. Read today's trades from `trades.jsonl`
2. Read today's anomalies from `anomalies.jsonl`
3. Get today's significant news:
   - Prefer: `news_sentiment(action="recent", symbols="<all>", window_minutes=1440, mode="cache", prefer_non_empty=true)`
   - If needed for full detail: read `memory/trading/news_cache_history.jsonl` and summarize
4. Summarize positions status from `positions.json`
5. Write daily report to `memory/trading/daily/{YYYY-MM-DD}.md`
6. Push brief summary to user

## Memory Integration

- **PATTERNS.md**: When review discovers a new high-win-rate pattern (>= 60% over 5+ trades), add it
- **LESSONS.md**: When review discovers a repeated mistake (same error 3+ times), add it
- **PROFILE.md**: When review shows style drift (leverage increasing, new coin preferences), update it
