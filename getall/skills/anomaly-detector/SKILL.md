---
name: anomaly-detector
description: "Market anomaly detection via batch scanning. Monitor OI spikes, volume surges, funding rate extremes, liquidation cascades, whale transfers, and CVD divergences for watched coins. Use when user says 'scan for anomalies', '有什么异常', 'anything unusual', or 'check the market'. For deep whale analysis, defer to whale-watcher skill."
metadata: '{"getall":{"always":false}}'
---

# Anomaly Detector — Market Anomaly Surveillance

Detect unusual market activity across multiple dimensions for user's watched coins. Primarily driven by `cron:anomaly-scan` (every 15 minutes), but can also be invoked on-demand.

## Trigger Method

- **Automatic**: `cron:anomaly-scan` runs every 15 minutes — this skill defines what to check and thresholds
- **On-demand**: User says "scan for anomalies", "anything unusual happening?", "check the market"

## Relationship with whale-watcher

This skill scans whale transfers as one of many dimensions in `batch_scan`. For **detailed whale analysis** (transfer pattern interpretation, exchange flow trends, large orderbook walls), defer to the `whale-watcher` skill. Use anomaly-detector for broad surveillance; use whale-watcher for deep dives.

## Monitoring Scope

**Only scan coins in the user's monitoring scope:**

```
Monitoring scope = PROFILE.md watchlist coins + all coins in positions.json
```

Anomalies in coins outside this scope → silently record to `anomalies.jsonl`, do NOT notify user.

## Step 1: Batch Data Fetch (Single Tool Call)

**Use batch_scan to get all dimensions for all monitored coins in ONE call:**

```
market_data(
  action="batch_scan",
  symbols="BTC,ETH,SOL,...",  ← all coins from monitoring scope
  dimensions="oi,funding,long_short,net_position,taker,cvd,liquidations,whale,coin_flow,spot_netflow"
)
```

This returns a structured summary: each coin × each dimension with key values.
Parse the output and check against the thresholds below.

> **Why batch_scan?** Without it, scanning 10 coins × 8 dimensions = 80 individual tool calls. With batch_scan, it's 1 call. The tool handles concurrency internally.

## Step 2: Apply Thresholds

### 1. Open Interest (OI) Spike

| Condition | Severity | Meaning |
|-----------|----------|---------|
| OI 15-min change > 8% | 🔴 HIGH | Large new positions opening — big move brewing |
| OI 1-hour change > 15% | 🔴 EXTREME | Massive new money inflow — expect significant volatility |
| OI declining while price rising | 🟡 MEDIUM | Short squeeze or weak rally — caution |
| OI rising while price dropping | 🟡 MEDIUM | New shorts piling in — could accelerate downside or squeeze |

### 2. Funding Rate Extreme

| Condition | Severity | Meaning |
|-----------|----------|---------|
| Funding rate > 0.05% | 🟡 MEDIUM | Longs are crowded — potential reversal risk |
| Funding rate > 0.1% | 🔴 HIGH | Extreme long crowding — short squeeze or dump likely |
| Funding rate < -0.05% | 🟡 MEDIUM | Shorts crowded — potential short squeeze |
| Funding direction sudden reversal | 🟡 MEDIUM | Sentiment shift — trend may be changing |

### 3. Liquidation Cascade

| Condition | Severity | Meaning |
|-----------|----------|---------|
| 1-hour total liquidations > $50M | 🔴 HIGH | Market clearing event — may signal bottom/top |
| Single-side liquidation dominance > 80% | 🟡 MEDIUM | One side getting wrecked — momentum may continue |

### 4. Whale Transfers (On-Chain)

| Condition | Severity | Meaning |
|-----------|----------|---------|
| Large transfer INTO exchange | 🟡 MEDIUM | Potential sell pressure incoming |
| Large transfer OUT of exchange | 🟡 MEDIUM | Accumulation signal — reducing exchange supply |
| Multiple large transfers same direction within 1h | 🔴 HIGH | Coordinated whale activity |

### 5. CVD (Cumulative Volume Delta) Divergence

| Condition | Severity | Meaning |
|-----------|----------|---------|
| Price rising + CVD falling | 🟡 MEDIUM | Bearish divergence — rally may be weak |
| Price falling + CVD rising | 🟡 MEDIUM | Bullish divergence — sellers may be exhausting |
| CVD trend suddenly reverses | 🟡 MEDIUM | Shift in buyer/seller aggression |

### 6. Net Long/Short Ratio Shift

| Condition | Severity | Meaning |
|-----------|----------|---------|
| Net long/short ratio shifts > 20% in 1h | 🔴 HIGH | Major positioning change across market |
| Extreme net long (> 70%) | 🟡 MEDIUM | Crowded trade — reversal risk |

### 7. Taker Buy/Sell Imbalance

| Condition | Severity | Meaning |
|-----------|----------|---------|
| Active sell ratio > 70% | 🟡 MEDIUM | Heavy selling pressure |
| Active buy ratio > 70% | 🟡 MEDIUM | FOMO buying — potential reversal risk |

### 8. Exchange Flow Anomaly

| Condition | Severity | Meaning |
|-----------|----------|---------|
| Large net inflow to exchanges | 🟡 MEDIUM | Potential sell pressure |
| Large net outflow from exchanges | 🟡 MEDIUM | Accumulation / HODL signal |

## Step 3: Alert Rules Check

After anomaly scans, also check `memory/trading/alerts.json` for user-defined alert rules:

- Price alerts (above/below thresholds)
- Funding rate alerts
- OI change alerts
- Custom conditions

## Notification Format

For HIGH severity anomalies in monitored coins:

```
🚨 {SYMBOL} Anomaly Detected

{anomaly_type}: {description}
{detail with numbers}

📊 Context:
• Current price: ${price} ({24h_change})
• Your position: {position_summary or "no position"}
• Funding rate: {rate}%

⚠️ Interpretation: {what this likely means}
```

For MEDIUM severity — batch into a summary if multiple:

```
📋 Market Scan Summary ({time})

{SYMBOL_1}: {anomaly_brief}
{SYMBOL_2}: {anomaly_brief}
...
```

## Recording

All detected anomalies are appended to `memory/trading/anomalies.jsonl`:

```json
{
  "ts": "{ISO timestamp}",
  "type": "{oi_spike|volume_surge|funding_extreme|liquidation_cascade|whale_transfer|cvd_divergence|flow_anomaly}",
  "symbol": "{SYMBOL}",
  "detail": "{human-readable description}",
  "severity": "{high|medium|low}",
  "notified": true/false,
  "user_action": null
}
```

The `user_action` field is updated later if the user acts on the anomaly (for learning purposes).
