---
name: whale-watcher
description: "Track whale movements with in-depth analysis. Monitor large on-chain transfers, exchange inflows/outflows, and large orderbook walls via Coinglass. Use when user asks about whale activity, 巨鲸动向, large transfers, exchange flows, or '鲸鱼在干嘛'. For detailed whale deep-dives; anomaly-detector handles whale alerts as part of broader scans."
metadata: '{"getall":{"always":false}}'
---

# Whale Watcher — Large Movement Tracking via Coinglass

Monitor and interpret large-scale capital movements including whale transfers, exchange flows, and large orderbook walls. All data sourced from Coinglass API.

## When to Use

- User asks "any whale activity?", "what are whales doing?"
- User asks about a specific coin's on-chain flows
- Part of `cron:anomaly-scan` checks (whale_transfers dimension)
- User says "check large transfers for {SYMBOL}"
- Pre-trade analysis (whale activity as a signal input)

## Relationship with anomaly-detector

The `anomaly-detector` skill includes whale transfers as ONE dimension in its batch scan — it flags whale anomalies but does not provide deep analysis. This skill (`whale-watcher`) provides the **detailed interpretation**: transfer pattern analysis, exchange flow trends, orderbook wall detection, and synthesized narratives. When `anomaly-detector` flags a whale anomaly, use this skill to dig deeper.

## Data Sources (All via `market_data` Tool)

### 1. Whale Transfers

```
market_data(action="whale_transfers", symbol={SYMBOL})
```

Large on-chain transfers detected by Coinglass. Each transfer includes:
- Amount and USD value
- Source → Destination (exchange/unknown wallet/cold wallet)
- Timestamp

**Interpretation Guide**:

| Transfer Pattern | Meaning | Signal |
|-----------------|---------|--------|
| Unknown → Exchange | Whale depositing to sell | 🔴 Bearish — potential sell pressure |
| Exchange → Unknown | Whale withdrawing to hold | 🟢 Bullish — reducing supply on exchange |
| Exchange → Exchange | Whale moving between venues | 🟡 Neutral — could be arbitrage |
| Unknown → Unknown | OTC or cold wallet reorganization | 🟡 Neutral |
| Multiple large → Exchange in short time | Coordinated selling | 🔴 Strongly bearish |
| Multiple large → Cold wallet in short time | Coordinated accumulation | 🟢 Strongly bullish |

### 2. Exchange Net Flow (Coin Flow)

```
market_data(action="coin_flow", symbol={SYMBOL})
```

Aggregate net flow of a coin across all major exchanges:
- Net inflow = more coins deposited than withdrawn (potential selling)
- Net outflow = more coins withdrawn than deposited (accumulation)

**Interpretation**:

| Flow Pattern | Meaning |
|-------------|---------|
| Sustained net outflow (days) | Long-term accumulation — bullish |
| Sudden large net inflow | Preparing to sell — bearish short-term |
| Net outflow during price drop | Smart money buying the dip |
| Net inflow during price rise | Whales distributing into strength |

### 3. Large Orderbook

```
market_data(action="large_orderbook", symbol={SYMBOL})
```

Detect large resting orders (bid/ask walls):
- Large bid walls = strong support level (buyers defending)
- Large ask walls = strong resistance level (sellers blocking)
- Wall removal = signal that the level may be tested

## Analysis Workflow

### Quick Check (On-Demand)

When user asks "what are whales doing with {SYMBOL}?":

1. Fetch whale transfers (last 24h)
2. Fetch coin flow (net in/out)
3. Fetch large orderbook
4. Synthesize into a narrative

### Output Format — Quick Check

```
🐋 Whale Activity: {SYMBOL}

## Large Transfers (24h)
{n} large transfers detected:
1. {amount} {SYMBOL} (${value}) | {source} → {destination} | {interpretation}
2. {amount} {SYMBOL} (${value}) | {source} → {destination} | {interpretation}
...

## Exchange Flow
Net flow (24h): {sign}{amount} {SYMBOL} (${value})
Direction: {net inflow / net outflow}
Interpretation: {what this suggests}

## Large Orders
Buy wall: ${price} — ${size} order
Sell wall: ${price} — ${size} order
Bias: {buy walls stronger / sell walls stronger / balanced}

## Summary
{One-paragraph synthesis — what does the whale data collectively suggest
about {SYMBOL}'s near-term direction? Be balanced and note uncertainty.}
```

### Anomaly Scan Integration

During `cron:anomaly-scan`, the whale watcher checks:

1. **Any whale transfer > $10M** involving user's watched coins → flag as anomaly
2. **Sudden large exchange inflow** (> 2x daily average) → flag
3. **Coordinated transfers** (3+ large transfers same direction within 1h) → high severity
4. Record all findings to `memory/trading/anomalies.jsonl` with `type: "whale_transfer"` or `type: "exchange_flow"`

### Pre-Trade Integration

When used as part of Order AI Score or pre-trade analysis:

- Whale data is one input dimension alongside technicals, derivatives, and sentiment
- Large whale selling into the user's buy → lower the score
- Large whale accumulation aligned with user's direction → boost confidence
- Absence of whale data → note as "whale activity: no significant moves detected"

## Limitations to Communicate

- Whale transfers have a delay (blockchain confirmation + Coinglass processing)
- Not all large transfers are directional bets (could be exchange operations, OTC, fund rebalancing)
- A whale moving coins to exchange doesn't guarantee they'll sell immediately
- Always combine whale data with other indicators — never trade on whale data alone
