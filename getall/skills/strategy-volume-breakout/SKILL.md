---
name: strategy-volume-breakout
description: "Volume breakout tracking strategy template. Enter when price breaks key resistance with significant volume expansion and OI confirmation."
metadata: '{"getall":{"always":false}}'
---

# Strategy Template: Volume Breakout

A momentum strategy that enters when price breaks through key resistance with strong volume confirmation and new capital entering (OI increase). Rides the breakout momentum until exhaustion.

## Strategy Logic

Genuine breakouts are accompanied by volume. When price pushes above a key resistance level on volume 3x+ the average AND open interest is rising simultaneously, it signals new capital is flowing in — not just short-covering. This creates a high-probability trend continuation setup.

## Entry Conditions (ALL must be true)

| # | Condition | Tool / Indicator | Rationale |
|---|-----------|-----------------|-----------|
| 1 | Price breaks above 20-day high (or identified resistance) | `technical_analysis(action="support_resistance")` + `market_data(action="klines")` | Key level breakout |
| 2 | Breakout candle volume > 3x 20-day average volume | `market_data(action="klines")` → compare current vs rolling avg | Volume validates the breakout |
| 3 | OI increasing > 3% during the breakout period | `market_data(action="open_interest")` | New money entering — not just short squeeze |
| 4 | Price closes above resistance (not just a wick) | Confirmed on candle close | Eliminate fakeouts |

### Optional Confirmation (Strengthens Signal)

- MACD histogram turning positive / expanding (momentum building)
- CVD trending upward (active buyers dominating)
- Taker buy/sell ratio > 1.2 (aggressive buying)
- Breakout aligns with higher-timeframe trend (daily trend is up)
- No major resistance immediately above (clear air above breakout level)
- Large orderbook ask wall was absorbed (sellers overcome)

## Exit Conditions

| Condition | Action | Rationale |
|-----------|--------|-----------|
| Price reaches 1.5x breakout range | Take partial profit (50%) | Secure gains at first target |
| Price reaches 2x breakout range | Take remaining profit | Full target achieved |
| Price drops back below breakout level -2% | Stop-loss (full close) | Failed breakout — exit immediately |
| Volume dries up (< 1x average) for 3+ candles | Consider exiting | Breakout losing steam |
| OI starts declining while price flat | Tighten stop | Participants leaving — exhaustion |

### Breakout Range Calculation
- Breakout range = distance from the support of the consolidation range to the breakout level
- Example: if price consolidated between $95k-$100k and broke above $100k, breakout range = $5k
- First target: $100k + ($5k × 1.5) = $107.5k
- Second target: $100k + ($5k × 2) = $110k

## Risk Management

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Max leverage | 3x | Momentum trades can be volatile — moderate leverage |
| Position size | ≤ 3% of account | Standard sizing |
| Stop-loss | Breakout level -2% | If price falls back below, the breakout failed |
| Entry method | Market order on confirmed close | Don't front-run — wait for candle close |

## Suitable Market Conditions

- **Consolidation following an uptrend** — coiled spring ready to release
- **High-timeframe trend is up** (weekly/daily) — breakout WITH the trend
- **Market sentiment is positive** — Fear & Greed > 40
- **After a squeeze** of the Bollinger Band (band width at low) — volatility expansion expected

## Unsuitable Market Conditions

- **Bear market / downtrend** — breakouts above resistance often fail and reverse
- **Low liquidity** — fake breakouts are more common
- **Major resistance overhead** (e.g., all-time high with massive sell wall) — breakout may stall
- **News-driven spike** without organic volume — often reverses quickly

## Suitable Assets

- BTC, ETH — cleanest breakout patterns, deepest liquidity
- SOL, BNB — can work but more prone to fakeouts
- Mid-cap alts — only if volume is genuinely 3x+ average (not just one large order)

## How to Use

1. **Identify setup**: "Is BTC setting up for a volume breakout?" → Agent checks consolidation + resistance + volume patterns
2. **Set alert**: "Alert me if BTC breaks $100k with volume" → Price + volume condition alert
3. **Activate monitoring**: "Watch for volume breakout on BTC and ETH" → Cron monitoring
4. **Backtest**: "Backtest volume breakout on SOL, last 6 months"

## Common Mistakes to Avoid

- **Entering during the wick** before candle close — many breakouts are fakeouts intrabar
- **Ignoring OI** — a breakout without OI increase is often just short-covering (weaker signal)
- **Chasing extended breakouts** — if price is already 5%+ above the breakout level, the entry is late
- **Not cutting failed breakouts** — if price drops back below, exit immediately; don't hope for re-break
