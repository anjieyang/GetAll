---
name: backtest-runner
description: "Run and interpret strategy backtests. Use when user says 'backtest', 'validate strategy', 'test this strategy historically', 'run backtest', or after creating a strategy via strategy-builder."
metadata: '{"getall":{"always":false}}'
---

# Backtest Runner — Strategy Validation Workflow

You have a `backtest` tool that returns **structured JSON metrics + a professional dashboard chart**. Your job is to interpret the data into actionable insight. Never dump raw JSON. Never draw your own charts — the tool generates a professional dashboard automatically.

## Hard Rules

1. **ALWAYS use the `backtest` tool** — never `exec` with matplotlib, never install external frameworks
2. **ALWAYS send the chart** — if `chart_path` exists, include it via `message(media=[chart_path])`. Never ask "want to see the chart?"
3. **ALWAYS lead with a verdict** — first sentence must be a clear pass/fail judgment
4. **ALWAYS compare vs benchmark** — use `benchmark_return_pct` and `excess_return_pct`
5. **ALWAYS end with one actionable suggestion** — what to try next
6. **ALWAYS follow the Lark-compatible formatting rules below** — your output will be rendered inside a Lark interactive card

## Lark Card Formatting Rules (MUST follow)

Your text response will be rendered as a Lark interactive card. Only a **subset** of markdown works:

**Supported ✅:**
- `## Title` — MUST be the very first line. Becomes the card header with color theme.
- `**bold**`, `*italic*`, `~~strikethrough~~`
- `[link text](url)`
- `` `inline code` `` and fenced code blocks (` ``` `)
- `- item` unordered lists / `1. item` ordered lists (flat only, no nesting)
- `---` horizontal divider (needs a blank line before it)

**NOT supported ❌ (will render as ugly plain text):**
- Markdown table syntax (`| a | b |`) — use the pipe table format below instead so the system can convert it to a native Lark table component
- `>` blockquotes
- `###` or deeper headings (only `#` and `##` work)
- Nested lists / list indentation

**Table format — MUST use this exact pattern** (with separator row):
```
| Column A | Column B | Column C |
|----------|----------|----------|
| value 1  | value 2  | value 3  |
```
The system parses this and renders it as a native Lark table component. If the separator row (`|---|---|---|`) is missing, parsing may fail and the table shows as raw text.

## Workflow

### Step 1: Parse Strategy → Build Config

Extract from user description or STRATEGY.md and build a JSON config:

```json
{
  "name": "strategy_name",
  "symbols": ["BTC/USDT"],
  "timeframe": "4h",
  "indicators": [{"name": "rsi", "params": {"period": 14}}],
  "entry_conditions": [{"indicator": "rsi", "field": "value", "operator": "lt", "value": 30}],
  "exit_conditions": [{"indicator": "rsi", "field": "value", "operator": "gt", "value": 70}],
  "direction": "long",
  "stop_loss_pct": 5,
  "take_profit_pct": 15,
  "trade_size_pct": 100
}
```

If critical info is missing, ask **one** clarifying question. Otherwise, fill reasonable defaults.

**IMPORTANT — Pre-validate conditions before calling the tool:**
Before sending, sanity-check for these known contradictory patterns:
- RSI > 70 + Price < MA → RSI超买=价格暴涨, 此时价格几乎必然在均线上方
- RSI < 30 + Price > MA → 同理, RSI超卖=价格暴跌, 不太可能在均线上方
- MACD cross_above + MACD histogram < 0 → 金叉时柱状图刚转正, 不会<0
- 5+ AND conditions with strict thresholds → 交集趋近于零

If you spot a contradiction, **warn the user before running** and suggest a fix.

### Step 2: Pick Exchange & Call the Tool

The `backtest` tool has an `exchange` parameter (`"binance"` or `"bitget"`). **You decide which exchange to use based on context:**

- If user mentions Bitget / uses Bitget tools → try `exchange="bitget"` first
- If user mentions Binance or no preference → use `exchange="binance"` (wider coverage)
- **If the result comes back with many `failed_symbols`** → re-run with the OTHER exchange for just those failed symbols. Merge results yourself.

**Symbol format for exchange backtest:** Use ccxt format: `"BTC/USDT"` or `"BTCUSDT"` (the tool does basic normalization, but prefer explicit format).
**Symbol format for ohlcv_json:** The symbol in the JSON is used as-is — match whatever the data source returned.

**Data availability reality:**
- Binance has more spot pairs; Bitget has different futures listings
- Low-volume / newly-listed coins often only exist on one exchange
- If ALL symbols fail on one exchange, try the other before giving up

```
backtest(action="run", strategy_config='{ ... }', period="6m", exchange="bitget")
```

If result shows `failed_symbols`, retry those on the other exchange:
```
backtest(action="run", strategy_config='{ ... only failed symbols ... }', period="6m", exchange="binance")
```

### Step 3: Interpret — Follow This Report Structure

Your text response MUST follow this structure (adapt wording to context):

```
## 📊 [Strategy Name] 回测结果 ([period] / [timeframe] / [symbols])

[VERDICT EMOJI] **结论:** [one-sentence pass/fail judgment with key number]

| 指标 | 策略 | 基准 |
|------|------|------|
| 总收益 | X% | Y% |
| 年化收益 | X% | - |
| 最大回撤 | X% | - |
| Sharpe | X | - |
| 胜率 | X% | - |
| 盈亏比 | X | - |
| Profit Factor | X | - |
| 交易次数 | X | - |

[WARNINGS if any — see quality flags below]

💡 **建议:** [one concrete, actionable next step]
```

**Critical formatting notes:**
- The `## 📊 ...` heading MUST be the very first line — it becomes the card header.
- The table MUST have the `|------|------|------|` separator row after the header.
- Use `**bold**` for emphasis (e.g. `**结论:**`), NOT plain colons or headings.
- Do NOT use box-drawing characters (┌─┬─┐ │ │ └─┴─┘) — they render as raw text.

Verdict emojis:
- 🟢 Strategy looks promising (PF > 1.5, excess return > 0, Sharpe > 1.0)
- 🟡 Mixed results, needs refinement (PF 1.0-1.5, or weak Sharpe)
- 🔴 Strategy fails (PF < 1.0, or negative excess return, or DD > 25%)

### Step 4: Send Chart + Text Together

```
message(content="[your analysis text above]", media=["[chart_path from metrics]"])
```

The dashboard chart has 4 panels: equity+benchmark, drawdown, monthly heatmap, metrics box. It speaks for itself — your text adds the *judgment* and *suggestion*.

## Zero-Trade Diagnosis (CRITICAL — handle when total_trades == 0)

When `total_trades == 0`, the result contains two diagnostic fields:

### 1. `entry_signal_diagnostics` — per-condition hit rates

```json
[
  {"symbol": "IP/USDT:USDT", "condition": "rsi.value gt 70",      "hits": 45,   "hit_pct": 1.56,  "valid_bars": 2880, "nan_bars": 14,  "status": "ok"},
  {"symbol": "IP/USDT:USDT", "condition": "macd.histogram lt 0",  "hits": 1400, "hit_pct": 48.87, "valid_bars": 2866, "nan_bars": 14,  "status": "ok"},
  {"symbol": "IP/USDT:USDT", "condition": "close.value lt ema20", "hits": 320,  "hit_pct": 11.17, "valid_bars": 2866, "nan_bars": 14,  "status": "ok"},
  {"symbol": "IP/USDT:USDT", "condition": "combined (and)",       "hits": 0,    "hit_pct": 0.0,   "valid_bars": 2880, "nan_bars": 0,   "status": "combined"}
]
```

**Key fields:**
- `hits` / `hit_pct`: how many bars this condition is True (individually)
- `valid_bars` / `nan_bars`: how many bars have real data vs NaN (indicator warmup)
- `status`: `ok` = evaluated, `not_found` / `unresolved_threshold` / `missing_indicator` = config error

### 2. `signal_analysis` — root cause diagnosis with drop-one analysis

The engine automatically runs a **drop-one analysis**: removes each condition one at a time and checks how many signals the remaining conditions produce.

```json
{
  "problem": "contradictory_conditions",
  "detail": "Each condition fires individually, but they never overlap (AND = 0). Removing 'rsi.value gt 70' would produce 280 signals (9.72%).",
  "bottleneck": "rsi.value gt 70",
  "drop_one": [
    {"dropped": "rsi.value gt 70",      "remaining_hits": 280, "remaining_pct": 9.72},
    {"dropped": "macd.histogram lt 0",   "remaining_hits": 5,   "remaining_pct": 0.17},
    {"dropped": "close.value lt ema20",  "remaining_hits": 38,  "remaining_pct": 1.32}
  ],
  "suggestions": ["Remove or relax 'rsi.value gt 70'...", ...],
  "symbols_with_zero_entries": 20,
  "symbols_total": 20
}
```

**`problem` types and how to report each:**

| problem | 含义 | 报告方式 |
|---|---|---|
| `contradictory_conditions` | 条件互斥, AND永远=0 | 展示 drop_one 数据, 指出移除哪个条件能产生信号 |
| `all_pairs_contradictory` | 移除任何单个条件仍=0 | 建议重新设计入场逻辑, 保留1-2个条件先测试 |
| `impossible_condition` | 某条件在数据中从未触发 | 指出具体哪个条件, 建议放宽阈值 |
| `insufficient_data_after_warmup` | 指标预热耗尽大部分数据 | 建议延长回测周期或缩短指标周期 |
| `all_conditions_skipped` | 全部条件因配置错误被跳过 | 列出 skipped 原因, 属于 config bug |

**Reporting template for zero trades:**

```
## 📊 [Strategy Name] 回测排查结果 ([period] / [timeframe] / [N]币)

🔴 **结论:** 0 笔交易 — [problem detail from signal_analysis]

**信号诊断 (以 [symbol] 为样本, 共 [total_bars] 根K线):**
- [condition 1]: 命中 [hits] 次 ([hit_pct]%)
- [condition 2]: 命中 [hits] 次 ([hit_pct]%)
- [condition 3]: 命中 [hits] 次 ([hit_pct]%)
- 三者同时满足: 0 次

**Drop-one 分析:**
- 去掉 [condition A]: 剩余命中 [N] 次 → [condition A] 是主要瓶颈
- 去掉 [condition B]: 剩余命中 [N] 次
- 去掉 [condition C]: 剩余命中 [N] 次

💡 **建议:** [use signal_analysis.suggestions[0], be specific]
```

## Quality Flags (always check, mention if triggered)

| Condition | Flag |
|---|---|
| total_trades == 0 | 🔴 零交易 — MUST use signal_analysis to explain root cause |
| total_trades < 30 | ⚠️ 样本量不足 (<30笔), 结论参考性有限 |
| win_rate_pct > 80 | ⚠️ 胜率异常高, 可能过拟合 |
| max_drawdown_pct > 25 | ⚠️ 回撤过大, 多数交易者无法承受 |
| excess_return_pct < 0 | ⚠️ 跑输持币不动, 策略不创造 alpha |
| sharpe_ratio < 0.5 | ⚠️ 风险调整收益差 |
| profit_factor < 1.0 | ⚠️ 策略在亏钱 (每赚1块要亏>1块) |
| max_consecutive_losses > 8 | ⚠️ 连续亏损过长, 心理承受力考验 |

## Iteration Suggestions (pick the most relevant ONE)

| Problem | Suggestion |
|---|---|
| Zero trades (contradictory) | 根据 drop_one 分析, 移除或放宽瓶颈条件 |
| Zero trades (impossible) | 放宽该条件阈值 (如 RSI>70 → RSI>65) |
| Zero trades (warmup) | 延长回测周期, 或缩短指标参数 |
| Zero trades (config error) | 检查指标名/算子拼写, 修复后重试 |
| Low win rate | 放宽入场阈值 or 加确认指标 |
| High drawdown | 收紧止损 or 减小仓位 |
| Few trades | 放宽条件 or 缩短 timeframe |
| Negative excess | 换方向 or 换策略类型 (趋势→均值回归) |
| Good results | 跑不同时段做 out-of-sample 验证 |

## Available Indicators

| Category | Names | Default |
|---|---|---|
| MA | sma, ema, dema, hma, wma | period=20 |
| Momentum | rsi, macd, roc, cci | period=14 |
| Bands | bollinger, stoch | period=20 |
| Volatility | atr | period=14 |

Multi-output field references: `macd.signal`, `bollinger.lower`, `stoch.k`

## Condition Operators

`lt` (or `<`), `gt` (or `>`), `lte` (or `<=`), `gte` (or `>=`), `eq` (or `==`), `cross_above` (or `crossover`), `cross_below` (or `crossunder`)

Threshold: number (30) or indicator ref ("macd.signal", "bollinger.lower", "close", "price")

## Data Source Strategy (YOU decide — never hardcoded)

You have 3 layers of data sources. Pick the best one for the situation:

**Layer 1: Exchange direct (best for futures/specific exchange data)**
- `backtest(exchange="binance")` — widest CEX coverage, good default
- `backtest(exchange="bitget")` — for Bitget-specific symbols
- Also: `okx`, `bybit`, `coinbase`, `kraken`, `kucoin`

**Layer 2: CoinGecko (widest overall coverage)**
- `coingecko(action="search", query="...")` → find coin ID
- `coingecko(action="ohlcv", coin_id="...", days="365")` → get candles
- `backtest(action="run", ohlcv_json="<coingecko output>", strategy_config="...")` → run backtest
- Best for: DeFi tokens, small caps, anything not on a single CEX

**Layer 3: Yahoo Finance (zero-failure mainstream fallback)**
- `yfinance_ohlcv(symbol="BTC-USD", period="1y")` → get candles
- `backtest(action="run", ohlcv_json="<yfinance output>", strategy_config="...")` → run backtest
- Best for: BTC/ETH/SOL daily data, long history, never fails

**Decision flow:**
1. User mentions specific exchange → start with that exchange
2. General request with common coins → `exchange="binance"`
3. Result has `failed_symbols` → retry those on another exchange (okx, bybit, etc.)
4. Still failing → try `coingecko` for those symbols
5. Mainstream coins as last resort → `yfinance_ohlcv`
6. **Never report "all failed" without trying at least 2 sources**
7. **Always report transparently:** "12 symbols from Bitget, 5 from Binance, 3 via CoinGecko"
