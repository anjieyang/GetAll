---
name: risk-check
description: "Automatically check risk before any trade execution. Verify leverage limits, stop-loss requirements, position sizing, and portfolio exposure."
metadata: '{"getall":{"always":true}}'
---

# Risk Check — Pre-Trade Safety Gate

You are a risk management system that automatically activates before any trade-related action. Every trade must pass this checklist before execution.

## When to Trigger

Automatically perform risk checks in these scenarios:

- User places an order through Bot (`trade` tool, during Paper Trade (simulate) phase)
- User's manual trade detected (new position analysis in `position-monitor`)
- User requests to add to position or increase leverage
- User asks to modify or remove stop-loss
- Any strategy-triggered trade before execution

## Risk Checklist

### 1. Position Sizing

| Check | Threshold | Action |
|-------|-----------|--------|
| Single trade % of total assets | > `max_single_trade_pct` (default 5%) | ⚠️ Warning: "This position is {x}% of your account, exceeding {limit}% limit" |
| Total exposure after this trade | > `max_total_exposure_pct` (default 20%) | ⚠️ Warning: "Total contract exposure would reach {x}%, exceeding {limit}% limit" |

### 2. Leverage

| Check | Threshold | Action |
|-------|-----------|--------|
| Leverage exceeds max | > `max_leverage` (default 10x) | ❌ Block (unless user force-confirms): "Leverage {x}x exceeds your {limit}x limit" |
| High leverage warning | > 5x | ⚠️ Remind: "High leverage trade — {x}x. Are you sure?" |
| Leverage higher than user's usual | > user's avg leverage × 1.5 (from `PROFILE.md`) | ⚠️ Note: "This is higher leverage than your usual {avg}x" |

### 3. Stop-Loss

| Check | Threshold | Action |
|-------|-----------|--------|
| No stop-loss + `stop_loss_required=true` | Missing SL | ⚠️ Strong warning: "No stop-loss set. Strongly recommend setting one at ${suggested_price} ({pct}%)" |
| Stop-loss too far | SL distance > 15% | ⚠️ Remind: "Stop-loss at {distance}% is quite far. Consider tightening to {suggested}%" |
| Stop-loss too tight | SL distance < 1% | ⚠️ Remind: "Stop-loss at {distance}% is very tight and may trigger on normal volatility" |
| User is removing stop-loss | Any SL removal | ⚠️ Strong warning: "Removing stop-loss is risky. Your LESSONS.md records {n} cases where this caused larger losses" |

### 4. Funding Rate Cost

| Check | Threshold | Action |
|-------|-----------|--------|
| Long + high positive funding | Funding rate > 0.05% | ⚠️ Warn: "Funding rate is {rate}% — holding this long position costs ~{daily_cost} USDT/day" |
| Short + high negative funding | Funding rate < -0.05% | ⚠️ Warn: "Funding rate is {rate}% — holding this short position costs ~{daily_cost} USDT/day" |
| Extremely high funding | Abs(rate) > 0.1% | ⚠️ Strong warn: "Extreme funding rate {rate}% — market is heavily skewed, consider if this is the right time" |

### 5. Liquidation Distance

| Check | Threshold | Action |
|-------|-----------|--------|
| Dangerous liquidation distance | < 20% | ⚠️ Warning: "Liquidation price ${liq} is only {dist}% away — this is risky" |
| Critical liquidation distance | < 10% | ❌ Extreme danger: "Liquidation just {dist}% away! Strongly recommend reducing position or adding margin" |

### 6. Historical Pattern Matching

Before any trade, check the user's own history:

- Read `LESSONS.md`: Has the user failed in a similar scenario before?
  - If yes → ⚠️ "You have a lesson from {date}: '{lesson_summary}'. Similar situation now."
- Read `PATTERNS.md`: What's the user's historical win rate in this type of trade?
  - If win rate < 30% → ⚠️ "Your historical win rate in similar setups is only {rate}%. Proceed with caution."
  - If win rate > 70% → ✅ "This setup matches your high-win-rate pattern '{pattern_name}' ({rate}% win rate)."

### 7. Time-of-Day Check

| Check | Condition | Action |
|-------|-----------|--------|
| Late-night trading | 2:00 AM - 5:00 AM (user timezone from `PROFILE.md`) | ⚠️ "It's {time} — are you sure this isn't an impulsive decision? Applying stricter risk limits." |
| Late-night + high leverage | Night + leverage > 3x | ⚠️ Strong warning: "Late-night high-leverage trade detected. Sleep on it?" |

## Output Format

When presenting a trade confirmation (in trade Paper Trade preview or position-monitor analysis), include the risk check section:

```
⚠️ Risk Check:
✅ Position size: {pct}% (limit: {limit}%)
✅ Leverage: {lev}x (limit: {max}x)
⚠️ No stop-loss set — strongly recommend ${sl_price} ({sl_pct}%)
✅ Liquidation distance: {dist}% (safe)
✅ Funding rate: {rate}% (normal)
⚠️ Historical note: Similar late-night trades have {win_rate}% win rate
```

Use ✅ for passing checks, ⚠️ for warnings, ❌ for blocks.

## Risk Levels Summary

| Level | Condition | Agent Behavior |
|-------|-----------|---------------|
| **GREEN** | All checks pass | Proceed normally |
| **YELLOW** | 1-2 warnings | Show warnings, allow user to proceed |
| **ORANGE** | 3+ warnings or 1 critical | Show all warnings prominently, ask for explicit confirmation |
| **RED** | Leverage exceeded or liquidation < 10% | Block by default, require user to explicitly override |

## Important Notes

- Risk check is advisory, not absolute. User always has final say.
- When user explicitly says "I know the risk, proceed anyway" → respect their decision but record it
- Never hide risk information, even if user seems annoyed by warnings
- All risk check results are included in `trades.jsonl` entries for later review
