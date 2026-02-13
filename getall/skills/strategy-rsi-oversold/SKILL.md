---
name: strategy-rsi-oversold
description: "RSI oversold bounce strategy template. Buy when RSI enters oversold territory with volume confirmation and Bollinger Band support."
metadata: '{"getall":{"always":false}}'
---

# Strategy Template: RSI Oversold Bounce

A mean-reversion strategy that enters long when RSI signals extreme oversold conditions, confirmed by volume expansion and Bollinger Band proximity. Targets a bounce back to neutral territory.

## Strategy Logic

When a coin is beaten down into oversold territory (RSI < 30), it often experiences a relief bounce. This strategy captures that bounce by entering long when multiple oversold confirmations align, and exits when the bounce reaches neutral territory.

## Entry Conditions (ALL must be true)

| # | Condition | Tool / Indicator | Rationale |
|---|-----------|-----------------|-----------|
| 1 | RSI(14) < 30 | `technical_analysis(action="indicators", indicators=["rsi"])` | Oversold territory — sellers exhausted |
| 2 | 5-min volume > 2x 20-day average volume | `market_data(action="klines")` → compare volume | Volume confirmation — bounce has participation |
| 3 | Price at or below Bollinger Band lower band (20, 2) | `technical_analysis(action="indicators", indicators=["bollinger"])` | Price at statistical extreme |
| 4 | NOT in strong downtrend (daily MA20 not declining for 10+ days) | `technical_analysis(action="indicators", indicators=["ma"], timeframe="1d")` | Avoid catching falling knives in bear trends |

### Optional Confirmation (Strengthens Signal)

- Funding rate negative or near zero (shorts paying longs — supportive)
- Fear & Greed Index < 40 (market fearful — contrarian bullish)
- RSI bullish divergence on lower timeframe (1h RSI making higher lows while price makes lower lows)

## Exit Conditions

| Condition | Action | Rationale |
|-----------|--------|-----------|
| RSI(14) > 60 | Take profit (full close) | Bounce has reached neutral/mildly overbought |
| Price up +8% from entry | Take profit (full close) | Target reached |
| Price down -5% from entry | Stop-loss (full close) | Thesis invalidated |
| Trailing stop: after +5% gain, if price retraces 3% | Exit remaining | Lock in profits on pullback |

### Exit Priority
1. Stop-loss (-5%) — always executes first, non-negotiable
2. RSI > 60 or +8% — whichever hits first
3. Trailing stop — activates after 5% unrealized gain

## Risk Management

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Max leverage | 3x | Mean-reversion is inherently risky — keep leverage low |
| Position size | ≤ 3% of account | Small size since catching bottoms is uncertain |
| Stop-loss | -5% from entry | Tight enough to limit damage, wide enough to survive noise |
| Max concurrent trades | 2 | Don't pile into multiple oversold bounces at once |

## Suitable Market Conditions

- **Range-bound / choppy markets** — RSI bounces work best when there's a floor
- **Post-correction in an uptrend** — pullback within a larger bull trend
- **Individual coin oversold** while BTC is stable or up — sector rotation play

## Unsuitable Market Conditions

- **Strong bear trend** (daily MA20 declining, lower lows) — RSI can stay oversold for weeks
- **Market-wide panic** (BTC -15%+ in a day, cascading liquidations) — wait for stabilization first
- **Right before major events** (CPI, Fed, token unlocks) — volatility may invalidate the setup

## Suitable Assets

- BTC, ETH, SOL, and other large-cap coins with sufficient liquidity
- Avoid low-cap coins (< $100M market cap) — RSI less reliable with thin orderbooks

## How to Use

1. **Manual check**: "Check RSI bounce setup for BTC" → Agent checks all entry conditions
2. **Activate monitoring**: "Activate RSI oversold strategy for BTC and ETH" → Cron task monitors conditions every 15 minutes
3. **Backtest**: "Backtest RSI oversold strategy on SOL, last 6 months" → Nautilus Trader simulation
4. **Customize**: "Change the RSI threshold to 25 instead of 30" → Agent updates the strategy parameters

## Historical Notes

RSI oversold bounces in crypto tend to be:
- More reliable on 4h timeframe than 1h (fewer false signals)
- Stronger when aligned with support levels
- Weaker during funding rate capitulation events (where even oversold keeps going)
