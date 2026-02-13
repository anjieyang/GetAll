---
name: strategy-bollinger-squeeze
description: "Bollinger Band squeeze breakout strategy template. Enter when Bollinger Bands squeeze to historical tightness and price breaks out of the band with volume confirmation."
metadata: '{"getall":{"always":false}}'
---

# Strategy Template: Bollinger Squeeze Breakout

A volatility expansion strategy that identifies periods of extreme low volatility (Bollinger Band squeeze) and enters when the inevitable breakout occurs. Low volatility always leads to high volatility — the question is which direction.

## Strategy Logic

Bollinger Band width reaching a local minimum (squeeze) indicates the market is coiling like a spring. When the bands are at their tightest, a large move is imminent. This strategy waits for the squeeze to resolve with a directional breakout, then enters in the breakout direction.

The key insight: **we don't predict the direction during the squeeze. We WAIT for the breakout and FOLLOW it.**

## Entry Conditions (ALL must be true)

| # | Condition | Tool / Indicator | Rationale |
|---|-----------|-----------------|-----------|
| 1 | Bollinger Band width (20, 2) at 20-period low | `technical_analysis(action="indicators", indicators=["bollinger"])` | Squeeze identified — volatility at minimum |
| 2 | Price closes outside the Bollinger Band | Same as above — track band break | Squeeze is resolving — direction chosen |
| 3 | RSI confirms direction: > 50 for upside break, < 50 for downside break | `technical_analysis(action="indicators", indicators=["rsi"])` | Momentum alignment |
| 4 | Volume on breakout candle > 1.5x average | `market_data(action="klines")` | Breakout has participation |

### Direction Rules
- **Long entry**: Price closes ABOVE upper Bollinger Band + RSI > 50 + volume confirmation
- **Short entry**: Price closes BELOW lower Bollinger Band + RSI < 50 + volume confirmation
- **No entry**: Price breaks a band but RSI doesn't confirm → wait

### Optional Confirmation (Strengthens Signal)

- OI increasing during breakout (new positions being opened — genuine interest)
- MACD crossing in breakout direction (momentum building)
- Squeeze duration > 10 candles (longer squeeze = more powerful breakout)
- Higher-timeframe trend aligns with breakout direction
- CVD confirming buyer/seller aggression in breakout direction

## Exit Conditions

### For Long Breakout (price broke above upper band):

| Condition | Action | Rationale |
|-----------|--------|-----------|
| Price returns to middle band (20 SMA) | Take profit | Mean-reversion target for range plays |
| Price up +5% from entry | Take profit | Conservative target |
| Price drops back inside bands and closes below middle band | Stop-loss | False breakout confirmed |
| Bollinger Bands re-squeeze (width drops again) | Exit | Momentum lost, new setup forming |

### For Short Breakout (price broke below lower band):

| Condition | Action | Rationale |
|-----------|--------|-----------|
| Price returns to middle band (20 SMA) | Take profit | Mean-reversion target |
| Price down -5% from entry | Take profit | Conservative target |
| Price rises back inside bands and closes above middle band | Stop-loss | False breakout confirmed |

### Universal Exit
- Stop-loss: -3% below breakout level (for longs) or +3% above (for shorts)
- This is tighter than other strategies because false Bollinger breakouts are common

## Risk Management

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Max leverage | 3x | Moderate — breakouts can be volatile both ways |
| Position size | ≤ 3% of account | Standard sizing |
| Stop-loss | -3% from entry (tight) | False breakouts need fast exits |
| Entry timing | On candle CLOSE only | Never enter mid-candle — too many fakeouts |
| Max hold time | 20 candles | If the move hasn't happened in 20 candles, it's not happening |

## Suitable Market Conditions

- **Range-bound / consolidation** — this is WHERE squeezes form
- **After a trend pause** — price consolidates before continuing
- **When band width is at multi-week low** — the tighter the squeeze, the more explosive the breakout
- **On higher timeframes (4h, 1d)** — more reliable than lower timeframes

## Unsuitable Market Conditions

- **Trending strongly** — if the market is already moving, there's no squeeze
- **During low liquidity** (weekends, holidays) — breakouts may be fakeouts
- **Very low timeframes (1m, 5m)** — too much noise, too many false signals
- **When Bollinger Bands are already wide** — no squeeze = no setup

## Suitable Assets

- **BTC, ETH** — cleanest band patterns, most reliable squeezes
- **SOL, BNB** — usable but expect more fakeouts
- **Any liquid asset** with > $100M daily volume — bands need sufficient data

## How to Use

1. **Scan for squeezes**: "Any coins in a Bollinger squeeze right now?" → Agent checks watchlist
2. **Monitor specific squeeze**: "Watch BTC Bollinger squeeze" → Alert when bands break
3. **Activate strategy**: "Activate Bollinger squeeze strategy for BTC and ETH" → Cron task
4. **Backtest**: "Backtest Bollinger squeeze on BTC 4h, last 6 months"
5. **Customize**: "Use 2.5 standard deviations instead of 2" → adjust parameters

## Pro Tips

- **Patience is key**: The squeeze can last for days or weeks. Don't force the entry — wait for the breakout.
- **First candle outside the band is NOT confirmation** — wait for the CLOSE outside the band.
- **Volume is the truth detector**: A breakout without volume is likely a fakeout.
- **Combine with trend filter**: Squeezes that resolve in the direction of the higher-timeframe trend have significantly higher success rates.
- **Multiple timeframe alignment**: If the daily AND 4h are both squeezing, the breakout will be larger.
