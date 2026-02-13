---
name: strategy-panic-bottom
description: "Fear & Greed panic buy strategy template. Scale into long positions during extreme market fear with whale accumulation and liquidation cascade confirmation."
metadata: '{"getall":{"always":false}}'
---

# Strategy Template: Panic Bottom (Fear & Greed Contrarian Buy)

A contrarian accumulation strategy that buys in tranches during extreme market fear. Historically, extreme fear in crypto marks some of the best buying opportunities — but the key is scaling in gradually, not going all-in.

## Strategy Logic

When the market enters extreme panic (Fear & Greed < 20), prices have already fallen significantly, overleveraged traders have been liquidated, and weak hands have sold. The remaining sellers are exhausted. By scaling in over multiple tranches, this strategy captures the bottom without needing to time it perfectly.

## Entry Conditions (ALL must be true for first tranche)

| # | Condition | Tool / Indicator | Rationale |
|---|-----------|-----------------|-----------|
| 1 | Fear & Greed Index < 20 (Extreme Fear) | `market_data(action="fear_greed")` | Market at emotional extreme |
| 2 | BTC 24h decline > 8% | `market_data(action="price", symbol="BTC")` | Significant drop has occurred |
| 3 | Large whale transfers OUT of exchanges | `market_data(action="whale_transfers")` | Smart money accumulating, not selling |
| 4 | Recent liquidation cascade > $200M in 24h | `market_data(action="liquidations")` | Forced sellers cleared — selling exhaustion |

### Additional Tranche Triggers

| Tranche | Trigger | Size |
|---------|---------|------|
| 1st (initial) | All 4 conditions met | 40% of planned position |
| 2nd | Price drops another 3-5% from first entry | 30% of planned position |
| 3rd | Price drops another 3-5% from second entry | 30% of planned position |

### Optional Confirmation (Strengthens Signal)

- Exchange net outflow (coins leaving exchanges — supply squeeze)
- Funding rate deeply negative (shorts paying longs — contrarian bullish)
- RSI < 20 on daily timeframe (extreme oversold)
- High-profile capitulation events (major fund liquidation, stablecoin depeg fear)
- CVD flattening after heavy selling (selling pressure exhausting)

## Exit Conditions

| Condition | Action | Rationale |
|-----------|--------|-----------|
| Fear & Greed recovers above 50 | Take profit (close 50%) | Sentiment normalized — first target |
| Price up +15% from average entry | Take profit (close remaining) | Strong bounce captured |
| Price down -10% from average entry | Stop-loss (full close) | Panic is deeper than expected — capital preservation |
| 30 days holding without recovery | Re-evaluate | If no bounce in a month, thesis may be wrong |

### Exit Priority
1. Stop-loss (-10% from average entry) — protect capital even in contrarian trade
2. Fear & Greed > 50 + partial take profit
3. Price target +15%
4. Time stop (30 days)

## Risk Management

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Max leverage | 2x (spot preferred) | Panic buys should use LOW leverage — market could drop further |
| Total position size | ≤ 5% of account (split across 3 tranches) | Enough to be meaningful, not enough to be devastating |
| Stop-loss | -10% from average entry | Wider than normal because panic bottoms are volatile |
| Entry method | 3 tranches over time | NEVER all-in — scale in to average into the bottom |
| Preferred instruments | Spot or very low leverage futures | Spot can't be liquidated — safer for bottom-fishing |

## Suitable Market Conditions

- **Market-wide crash** (not single-coin crash) — broad fear is more reliable than coin-specific fear
- **After cascading liquidations** — forced sellers already cleared
- **Fear & Greed has been < 30 for 2+ days** — sustained fear, not just a flash crash
- **No fundamental breakdown** (exchange hack, major protocol exploit) — fear from price action, not structural failure

## Unsuitable Market Conditions

- **Single coin crashing** while BTC is fine — that's a coin-specific problem, not systemic fear
- **Known upcoming negative catalyst** (major hack, proven fraud, regulatory ban) — fear may be justified
- **Very early in a bear market** — Fear & Greed can stay < 20 for weeks in a real bear
- **Stablecoin crisis** (USDT/USDC depeg risk) — systemic risk is different from sentiment

## Suitable Assets

- **BTC** — most reliable panic bounce candidate (always recovers... so far)
- **ETH** — second most reliable, especially if BTC is leading the bounce
- **SOL, BNB** — can work but higher risk, allocate less
- **Avoid altcoins** during panic buys — many never recover from major crashes

## How to Use

1. **Manual check**: "Is this a good time to panic buy?" → Agent checks all conditions
2. **Set alert**: "Alert me when Fear & Greed drops below 20" → conditional alert
3. **Activate**: "Activate panic bottom strategy" → Cron monitors F&G index daily
4. **Execute**: When conditions met, Agent suggests first tranche with specific parameters

## Historical Context

Crypto markets have rewarded buying extreme fear:
- Every time F&G dropped below 15 since 2020, BTC was higher 30 days later (sample: 8 events)
- Average 30-day return from F&G < 20 entry: +18.5%
- BUT: some drawdowns from entry to actual bottom can be 10-20% (hence the tranche approach)

**The hardest part of this strategy is psychological** — buying when everything looks terrible. This is where the Agent's emotional support skill helps.
