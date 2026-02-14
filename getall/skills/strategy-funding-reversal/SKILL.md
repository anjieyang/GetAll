---
name: strategy-funding-reversal
description: "Funding rate reversal short strategy template. Use when user asks about funding rate strategy, crowded long short, funding reversal, overbought short, or 资金费率反转做空."
metadata: '{"getall":{"always":false}}'
---

# Strategy Template: Funding Rate Reversal Short

A contrarian strategy that shorts when funding rates indicate extreme long crowding, confirmed by overbought technical signals. Profits from the inevitable mean-reversion when overleveraged longs get squeezed.

## Strategy Logic

When funding rates are persistently high (> 0.06%), it means long traders are paying a premium to maintain positions — a sign of overcrowding. Combined with overbought technicals, this creates a high-probability short setup. The trade profits when longs capitulate, funding normalizes, and price corrects.

## Entry Conditions (ALL must be true)

| # | Condition | Tool / Indicator | Rationale |
|---|-----------|-----------------|-----------|
| 1 | Funding rate > 0.06% for 3+ consecutive periods | `market_data(action="funding_rate")` | Sustained long overcrowding — not just a spike |
| 2 | RSI(14) > 75 | `technical_analysis(action="indicators", indicators=["rsi"])` | Technical overbought confirmation |
| 3 | Long/Short ratio > 1.5 | `market_data(action="long_short_ratio")` | Crowd is heavily positioned long |
| 4 | Price is near or above Bollinger Band upper band | `technical_analysis(action="indicators", indicators=["bollinger"])` | Statistical extreme to the upside |

### Optional Confirmation (Strengthens Signal)

- OI at local high and flattening (new longs not entering anymore)
- CVD showing declining buy aggression despite price at highs (bearish divergence)
- Large whale transfers INTO exchanges (potential sell preparation)
- Taker buy/sell ratio declining (active buying weakening)
- Net long ratio > 65% (extreme one-sided positioning)

## Exit Conditions

| Condition | Action | Rationale |
|-----------|--------|-----------|
| Funding rate drops below 0.01% | Take profit (full close) | Mean-reversion complete |
| Price down -8% from entry | Take profit (full close) | Target reached |
| Price up +4% from entry | Stop-loss (full close) | Thesis invalidated — momentum too strong |
| Funding rate spikes to > 0.15% | Re-evaluate | Market may be in parabolic phase — consider cutting loss |

### Exit Priority
1. Stop-loss (+4%) — countertrend trade needs tight stop
2. Funding rate normalization (< 0.01%) — primary exit signal
3. Price target (-8%) — secondary exit

## Risk Management

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Max leverage | 2x | This is a COUNTERTREND trade — must be conservative |
| Position size | ≤ 2% of account | Smaller than trend-following trades due to higher risk |
| Stop-loss | +4% from entry | Tight stop — if longs keep pushing, exit quickly |
| Max hold time | 7 days | If funding hasn't normalized in a week, close and reassess |

## Suitable Market Conditions

- **Late-stage rallies** — euphoria phase where everyone is long
- **After extended uptrend** (2+ weeks of green) — overheated market
- **When multiple coins** show high funding simultaneously — market-wide overheating

## Unsuitable Market Conditions

- **Early bull trend** — funding can stay high for weeks during genuine breakouts
- **Major bullish catalyst** (ETF approval, halving, regulatory clarity) — fundamentals override technicals
- **Low liquidity periods** (holidays, weekends) — wider spreads make shorting riskier
- **When shorts are also crowded** — indicates genuine uncertainty, not one-sided positioning

## Suitable Assets

- BTC, ETH — most reliable for funding rate signals (highest liquidity)
- SOL, DOGE — can work but funding spikes are more volatile
- Avoid low-cap coins — funding can be manipulated with small capital

## How to Use

1. **Manual check**: "Check funding reversal setup for BTC" → Agent evaluates all conditions
2. **Activate monitoring**: "Watch for funding reversal on BTC and ETH" → Cron task every 15 min
3. **Backtest**: "Backtest funding reversal strategy on BTC, last year" → historical validation
4. **Adjust**: "Make it more aggressive — use 0.05% threshold" → update parameters

## Key Insight

The edge in this strategy comes from patience — you need funding to be persistently high, not just spiking. A single 0.06% reading is noise. Three consecutive periods at 0.06%+ is a genuine signal. The market almost always mean-reverts, but timing matters.
