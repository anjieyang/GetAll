---
name: close-position
description: "Smart position closing with analysis. Shows all positions, analyzes P&L, and helps decide what to close. Use when user says '平仓', '平掉', '关仓', '止盈', '止损', '减仓', '全平', 'close position', or wants to exit a trade."
metadata: '{"getall":{"always":false,"emoji":"🚪","triggers":["平仓","close","平掉","关仓","了结","止盈","止损","减仓","全平"]}}'
---

# Close Position — 智能平仓助手

当用户说"平仓"时，**主动**展示所有持仓并分析是否该平。

## 触发词

用户说以下任一词时激活此 skill：
- "平仓"、"平掉"、"关仓"、"了结"
- "止盈"、"止损"（可能想平仓）
- "减仓"、"全平"
- "close position"

## 核心流程

### Step 1: 立即获取所有持仓

**不要问用户任何问题，先获取数据：**

```python
# 1. 获取合约持仓
portfolio(action="positions")

# 2. 获取盈亏汇总
portfolio(action="pnl_summary")

# 3. （可选）获取当前挂单，看看是否有TP/SL
portfolio(action="open_orders")
```

### Step 2: 展示持仓概览

**格式化展示，让用户一目了然：**

```
📊 当前持仓概览（3个仓位）
═══════════════════════════════════════

1️⃣ BTC/USDT:USDT (LONG)
   📈 盈利中 +$127.45 (+3.2%)
   ├─ 开仓价: $65,000 | 现价: $67,080
   ├─ 持仓量: 0.1 BTC ($6,708)
   ├─ 杠杆: 10x | 保证金: $670.8
   ├─ 止损: $63,050 (-3%) ✅
   └─ 持仓时间: 2天3小时
   💡 建议: 盈利超3%，可考虑部分止盈

2️⃣ ETH/USDT:USDT (LONG)
   📉 亏损中 -$45.20 (-1.8%)
   ├─ 开仓价: $3,200 | 现价: $3,142
   ├─ 持仓量: 0.5 ETH ($1,571)
   ├─ 杠杆: 5x | 保证金: $314.2
   ├─ 止损: 未设置 ⚠️
   └─ 持仓时间: 5小时
   💡 建议: 建议设置止损，当前无保护

3️⃣ SOL/USDT:USDT (SHORT)
   📈 盈利中 +$22.30 (+2.1%)
   ├─ 开仓价: $150 | 现价: $146.85
   ├─ 持仓量: 10 SOL ($1,468.5)
   ├─ 杠杆: 3x | 保证金: $489.5
   ├─ 止损: $157.5 (+5%) ✅
   └─ 持仓时间: 12小时
   💡 建议: 可继续持有，止损已设置

═══════════════════════════════════════
📊 总计: +$104.55 | 胜率: 2/3
═══════════════════════════════════════

请问您要：
  [1] 平掉某个仓位（告诉我编号或币种）
  [2] 全部平仓
  [3] 减仓（只平一部分）
  [4] 先设置止损再说
```

### Step 3: 分析建议逻辑

**盈利仓位：**
- 盈利 > 5%：建议考虑止盈或移动止损
- 盈利 > 10%：强烈建议部分止盈锁定利润
- 盈利 > 20%：建议至少平掉50%

**亏损仓位：**
- 亏损 < 2%：可继续持有，但确保有止损
- 亏损 2-5%：评估是否符合预期，考虑减仓
- 亏损 > 5%：建议止损，避免更大损失

**风险检查：**
- 无止损：⚠️ 警告，建议立即设置
- 杠杆 > 10x：⚠️ 高杠杆，注意爆仓风险
- 持仓时间 > 7天：提醒评估持仓理由

### Step 4: 执行平仓

用户选择后，执行平仓：

**平掉单个仓位：**
```python
# 先预览
trade(
    action="dry_run",
    symbol="BTC/USDT:USDT",
    side="sell",  # LONG仓位平仓用sell
    type="market",
    amount=0.1  # 全部持仓量
)

# 用户确认后执行
trade(
    action="place_order",
    symbol="BTC/USDT:USDT",
    side="sell",
    type="market",
    amount=0.1,
    paper_trade=False
)
```

**全部平仓：**
```python
# 依次平掉每个仓位
for position in positions:
    close_side = "sell" if position.side == "long" else "buy"
    trade(
        action="place_order",
        symbol=position.symbol,
        side=close_side,
        type="market",
        amount=position.contracts,
        paper_trade=False
    )
```

**部分减仓（如平掉50%）：**
```python
trade(
    action="place_order",
    symbol="BTC/USDT:USDT",
    side="sell",
    type="market",
    amount=0.05,  # 0.1 * 50%
    paper_trade=False
)
```

## 快捷指令处理

| 用户说 | Agent 动作 |
|-------|-----------|
| "平仓" | 展示所有持仓，等待选择 |
| "平掉BTC" | 直接平BTC仓位（先展示确认） |
| "全平" | 展示所有持仓，确认后全部平仓 |
| "止盈BTC" | 按当前盈利平仓BTC |
| "止损ETH" | 按当前价格止损平仓ETH |
| "BTC减仓一半" | 平掉BTC仓位的50% |
| "平掉所有盈利单" | 只平盈利的仓位 |
| "平掉亏损超过3%的" | 只平亏损>3%的仓位 |

## 关键原则

1. **永远先展示持仓**：不要问"你要平哪个"，先展示有什么
2. **提供分析建议**：不只是数据，要给出是否应该平的建议
3. **一次性收集信息**：用户说"平BTC"后直接执行，不要问side/type/amount
4. **安全确认**：真实平仓前展示预览，等用户确认
5. **批量操作支持**：支持"全平"、"平所有盈利单"等批量操作

## 无持仓时的响应

```
📋 当前没有持仓

您的账户余额：
  USDT: $5,234.56 (可用)

需要帮您开新仓位吗？
```

## 错误处理

**获取持仓失败：**
```
⚠️ 获取持仓信息失败，请稍后重试。
错误信息：{error}

您也可以手动查看：
  portfolio(action="positions")
```
