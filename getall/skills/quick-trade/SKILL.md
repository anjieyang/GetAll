---
name: quick-trade
description: "Fast trade execution from natural language. Handles spot and futures orders with automatic parameter inference."
metadata: '{"getall":{"always":false,"emoji":"⚡"}}'
---

# Quick Trade — Natural Language Order Execution

Parse user's trading intent and execute orders with minimal friction. **Collect ALL parameters in ONE step.**

## When to Use

User says things like:
- "帮我买2000U的BTC"
- "做多ETH，3000刀，10倍杠杆"
- "卖掉一半BTC持仓"
- "买0.5个ETH"
- "开空SOL，止损5%"

## Core Workflow

### Step 1: Parse Intent (ONE STEP)

From user's message, extract ALL of the following:

| Field | How to Infer | Default |
|-------|--------------|---------|
| `symbol` | Mentioned coin → `{COIN}/USDT` or `{COIN}/USDT:USDT` for futures | — |
| `side` | "买/做多/开多/long" → `buy`; "卖/做空/开空/short" → `sell` | — |
| `type` | Usually `market` unless user mentions specific price | `market` |
| `amount` | See calculation rules below | — |
| `leverage` | Only for futures, user mentions "X倍杠杆" | `None` (spot) |
| `stop_loss` | User mentions "止损X%" or specific price | `None` |
| `take_profit` | User mentions "止盈X%" or specific price | `None` |

### Step 2: Calculate Amount

**If user specifies USDT value (e.g., "2000U"):**
```
1. Get current price: market_data(action="price", symbol="...")
2. For SPOT: amount = usdt_value / price
3. For FUTURES: amount = (usdt_value * leverage) / price
```

**If user specifies percentage (e.g., "一半持仓"):**
```
1. Get current position: portfolio(action="positions")
2. Calculate: amount = position_size * percentage
```

**If user specifies coin amount (e.g., "0.5个ETH"):**
```
amount = 0.5 (directly)
```

### Step 3: Calculate Stop Loss / Take Profit

**If user specifies percentage (e.g., "止损3%"):**
```
For LONG: stop_loss = current_price * (1 - pct)
For SHORT: stop_loss = current_price * (1 + pct)
```

### Step 4: Execute with Confirmation

**Always dry_run first:**
```
trade(
    action="dry_run",
    symbol="BTC/USDT:USDT",
    side="buy",
    type="market",
    amount=0.0308,
    leverage=10,
    stop_loss=63050
)
```

**Show preview to user, then ask for confirmation:**
```
📋 订单预览：
• 做多 BTC/USDT 0.0308 BTC
• 杠杆 10x，保证金 2000 USDT
• 止损价 63050 (距现价 -3%)
• 风险评估：✅ 合理

确认下单？
```

**User confirms → Execute real order:**
```
trade(
    action="place_order",
    symbol="BTC/USDT:USDT",
    side="buy",
    type="market",
    amount=0.0308,
    leverage=10,
    stop_loss=63050,
    paper_trade=False
)
```

## Complete Examples

### Example 1: "帮我做多BTC 2000U 10倍"

```python
# 1. Parse
intent = {
    "symbol": "BTC/USDT:USDT",
    "side": "buy",
    "type": "market",
    "leverage": 10,
    "usdt_value": 2000
}

# 2. Get price
price = market_data(action="price", symbol="BTC/USDT")  # e.g., 65000

# 3. Calculate
notional = 2000 * 10  # 20000 USDT
amount = notional / 65000  # 0.3077 BTC

# 4. Execute
trade(
    action="place_order",
    symbol="BTC/USDT:USDT",
    side="buy",
    type="market",
    amount=0.3077,
    leverage=10,
    paper_trade=True  # Preview first
)
```

### Example 2: "卖掉一半ETH"

```python
# 1. Get current position
positions = portfolio(action="positions", symbol="ETH/USDT")
# e.g., returns: {"ETH": 2.0}

# 2. Calculate half
amount = 2.0 * 0.5  # 1.0 ETH

# 3. Execute
trade(
    action="place_order",
    symbol="ETH/USDT",
    side="sell",
    type="market",
    amount=1.0,
    paper_trade=True
)
```

### Example 3: "3200限价买1个ETH"

```python
trade(
    action="place_order",
    symbol="ETH/USDT",
    side="buy",
    type="limit",
    amount=1.0,
    price=3200,
    paper_trade=True
)
```

### Example 4: "开空SOL，1000U，止损5%，止盈15%"

```python
# 1. Get price
price = market_data(action="price", symbol="SOL/USDT")  # e.g., 150

# 2. Calculate
amount = 1000 / 150  # 6.67 SOL
stop_loss = 150 * 1.05  # 157.5 (SHORT, so SL is above)
take_profit = 150 * 0.85  # 127.5 (SHORT, so TP is below)

# 3. Execute
trade(
    action="place_order",
    symbol="SOL/USDT:USDT",
    side="sell",
    type="market",
    amount=6.67,
    stop_loss=157.5,
    take_profit=127.5,
    paper_trade=True
)
```

## Error Handling

**Missing critical info → Ask ONCE:**
```
如果用户只说"买点BTC"，缺少数量信息，一次性询问：
"请问您想买多少？可以告诉我：
• USDT金额（如：2000U）
• 或者BTC数量（如：0.1个）"
```

**Do NOT ask multiple questions one at a time!**

## Symbol Format Rules

| Market | Format | Example |
|--------|--------|---------|
| Spot | `{BASE}/{QUOTE}` | `BTC/USDT` |
| USDT-M Futures | `{BASE}/{QUOTE}:{SETTLE}` | `BTC/USDT:USDT` |
| Coin-M Futures | `{BASE}/{QUOTE}:{BASE}` | `BTC/USD:BTC` |

**Auto-detect:**
- User mentions "杠杆/合约/做多/做空" → Use futures format
- Otherwise → Use spot format

## Safety Defaults

- `paper_trade=True` for ALL orders by default
- Always show dry_run preview before real execution
- Require explicit user confirmation for real orders
- Flag high-risk orders (leverage > 20x, no stop loss, large position)
