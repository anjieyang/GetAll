# Available Tools

This document describes the tools available to getall.

## File Operations

### read_file
Read the contents of a file.
```
read_file(path: str) -> str
```

### write_file
Write content to a file (creates parent directories if needed).
```
write_file(path: str, content: str) -> str
```

### edit_file
Edit a file by replacing specific text.
```
edit_file(path: str, old_text: str, new_text: str) -> str
```

### list_dir
List contents of a directory.
```
list_dir(path: str) -> str
```

## Shell Execution

### exec
Execute a shell command and return output.
```
exec(command: str, working_dir: str = None) -> str
```

**Safety Notes:**
- Commands have a configurable timeout (default 60s)
- Dangerous commands are blocked (rm -rf, format, dd, shutdown, etc.)
- Output is truncated at 10,000 characters
- Optional `restrictToWorkspace` config to limit paths

## Web Access

### web_search
Search the web using Brave, OpenAI web search, or DuckDuckGo.
```
web_search(query: str, count: int = 5) -> str
```

Returns search results with titles, URLs, and snippets.
If Brave/OpenAI keys are missing, it can fall back to DuckDuckGo (no key).

### web_fetch
Fetch and extract main content from a URL.
```
web_fetch(url: str, extractMode: str = "markdown", maxChars: int = 50000) -> str
```

**Notes:**
- Content is extracted using readability
- Supports markdown or plain text extraction
- Output is truncated at 50,000 characters by default

## Communication

### message
Send a message to the user (used internally).
```
message(content: str, channel: str = None, chat_id: str = None) -> str
```

## Voice (Built-in TTS/STT)

You have **native** voice capability — no third-party API keys or skills needed.

### How it works
- **STT (Speech-to-Text)**: When a user sends a voice message, it is automatically transcribed and delivered to you as text. You don't need to do anything special.
- **TTS (Text-to-Speech)**: When voice mode is active, your text reply is automatically converted to an audio message and sent back to the user.
- **Voice mode** is enabled automatically when the user sends a voice message, or when they say "用语音"/"发语音". It is disabled when the user says "用文字"/"别用语音".

### Choosing a voice
When voice mode is active, pick a voice that fits the mood by adding a tag anywhere in your reply:
```
[[voice:coral]]           ← just pick a voice
[[voice:onyx instructions=用低沉严肃的语气]]  ← voice + style instructions
```
The tag is stripped before audio synthesis — the user won't see it.

**Available voices:**
| Voice | Characteristics |
|---|---|
| `alloy` | Neutral, balanced, versatile |
| `ash` | Warm, conversational male |
| `ballad` | Expressive, melodic |
| `coral` | Warm, engaging female (default) |
| `echo` | Smooth, deep male |
| `fable` | Expressive, British accent |
| `nova` | Warm, upbeat female |
| `onyx` | Deep, authoritative male |
| `sage` | Calm, composed |
| `shimmer` | Bright, optimistic female |
| `verse` | Versatile, expressive |
| `marin` | Natural high-quality female ★ |
| `cedar` | Natural high-quality male ★ |

### Voice mode guidelines
- Keep replies concise and conversational (≤300 chars ideal)
- Avoid code blocks, tables, and complex formatting — these don't work as speech
- If the answer needs code or long text, just reply in text — the system will automatically skip TTS for code-heavy or long responses
- Use `instructions` to control emotion, speed, accent, whispering, etc.
- **Do NOT** look for or try to use external TTS skills/APIs — your voice capability is built-in

## Background Tasks

### spawn
Spawn a subagent to handle a task in the background.
```
spawn(task: str, label: str = None) -> str
```

Use for complex or time-consuming tasks that can run independently. The subagent will complete the task and report back when done.

## Shared Workbench

### workbench
Create/run reusable scripts and shared skills, discover capabilities, and install skills from URLs.
```
workbench(
  action: str,
  script_name: str = "",
  skill_name: str = "",
  source_url: str = "",
  content: str = "",
  request: str = "",
  risk_level: str = "readonly",
  confirmed: bool = false
) -> str
```

Common actions:
- `list_scripts`, `create_script`, `run_script`
- `list_skills`, `create_skill`
- `discover_capabilities` (natural-language discovery, e.g. from ClawHub/MCP/GitHub)
- `install_skill_from_url` (install into shared workspace)
- `list_capabilities` (shared registry view)

Security note:
- For `install_skill_from_url`, set `risk_level="privileged"` only when needed.
- Privileged installs require explicit confirmation via `confirmed=true`.

## Scheduled Reminders (Cron)

Use the `exec` tool to create scheduled reminders with `getall cron add`:

### Set a recurring reminder
```bash
# Every day at 9am
getall cron add --name "morning" --message "Good morning! ☀️" --cron "0 9 * * *"

# Every 2 hours
getall cron add --name "water" --message "Drink water! 💧" --every 7200
```

### Set a one-time reminder
```bash
# At a specific time (ISO format)
getall cron add --name "meeting" --message "Meeting starts now!" --at "2025-01-31T15:00:00"
```

### Manage reminders
```bash
getall cron list              # List all jobs
getall cron remove <job_id>   # Remove a job
```

## Heartbeat Task Management

The `HEARTBEAT.md` file in the workspace is checked every 30 minutes.
Use file operations to manage periodic tasks:

### Add a heartbeat task
```python
# Append a new task
edit_file(
    path="HEARTBEAT.md",
    old_text="## Example Tasks",
    new_text="- [ ] New periodic task here\n\n## Example Tasks"
)
```

### Remove a heartbeat task
```python
# Remove a specific task
edit_file(
    path="HEARTBEAT.md",
    old_text="- [ ] Task to remove\n",
    new_text=""
)
```

### Rewrite all tasks
```python
# Replace the entire file
write_file(
    path="HEARTBEAT.md",
    content="# Heartbeat Tasks\n\n- [ ] Task 1\n- [ ] Task 2\n"
)
```

## Backtesting

### backtest
Run historical backtests using the built-in VectorBT engine. Returns structured JSON metrics.
```
backtest(action: str, strategy_config: str, period: str = "6m", exchange: str = "binance", starting_balance: float = 100000) -> str
```

**Actions:**
- `run` — Execute a backtest. Requires `strategy_config` (JSON string). Returns JSON with: `total_return_pct`, `win_rate_pct`, `profit_factor`, `sharpe_ratio`, `max_drawdown_pct`, `total_trades`, `chart_path`, `failed_symbols`, etc.
- `chart` — Generate equity chart from previous metrics JSON.

**Exchange parameter:**
- `"binance"` — Binance futures (wider coverage, good default)
- `"bitget"` — Bitget futures
- Choose based on user context. **If result contains `failed_symbols`, retry those on the other exchange.**

**Symbol format:** Accepts any format: `"BTCUSDT"`, `"BTC/USDT"`, `"BTC/USDT:USDT"` — auto-normalized.

**Operators:** Accepts both canonical (`lt`, `gt`, `cross_above`) and symbol forms (`<`, `>`, `crossover`).

**Indicator aliases:** `"Bollinger Bands"` → `bollinger`, `"SMA"` → `sma`, `"price"` → close price. Parameters accept `length`/`window`/`period` interchangeably.

**CRITICAL: This is the ONLY way to backtest.** Never install or use external backtesting frameworks (jesse, freqtrade, backtrader, nautilus_trader, backtesting.py, etc.). All backtesting goes through this tool.

See the `backtest-runner` skill for workflow guidance on building strategy configs.

## Market Data Sources

You have multiple native data sources. These are general-purpose — use them for price checks, analysis, research, charting, backtesting, or any task that needs market data.

**IMPORTANT — Symbol Formats Differ Per Source:**
Each data source uses its own symbol format. YOU decide the correct format based on the source:
- Exchange (ccxt): `BTC/USDT:USDT` (futures) or `BTC/USDT` (spot)
- CoinGecko: `bitcoin`, `ethereum` (their own IDs — use `search` to find)
- Yahoo Finance: `BTC-USD`, `ETH-USD`, `AAPL`, `GC=F`
- Bitget API: `BTCUSDT`

### coingecko
General-purpose crypto data from CoinGecko (28M+ tokens, 1700+ exchanges, 10+ years history).
```
coingecko(action: "search" | "price" | "ohlcv" | "markets" | "trending" | "global", ...) -> str
```

**Actions:**
- `search` — Find CoinGecko coin ID by name or ticker. Use this first if you only know "BTC".
- `price` — Current price + 24h change for one or more coins (`coin_ids="bitcoin,ethereum"`).
- `ohlcv` — OHLCV candle data for a coin (`coin_id="bitcoin"`, `days="365"`).
- `markets` — Top coins by market cap with price/volume/change data.
- `trending` — Currently trending coins (what people are searching on CoinGecko right now).
- `global` — Global crypto market stats: total market cap, 24h volume, BTC/ETH dominance.

**Best for:** DeFi tokens, small caps, cross-exchange aggregated price, market overview, trending sentiment.

### yfinance
Market data from Yahoo Finance. No API key needed. Covers crypto + stocks + forex + commodities.
```
yfinance(action: "ohlcv" | "price", symbol: str, ...) -> str
```

**Actions:**
- `ohlcv` — Candle data. Period: '1mo' to 'max'. Interval: '1h' to '1mo'.
- `price` — Current quote with change%.

**Symbol format (YOU decide):** `BTC-USD`, `ETH-USD`, `AAPL`, `EURUSD=X`, `GC=F`
**Best for:** Long-term daily data (5+ years), mainstream assets, zero-failure fallback.

### defillama
On-chain DeFi data from DefiLlama (5000+ protocols, 200+ chains). **No API key needed.**
```
defillama(action: "tvl_rank" | "protocol" | "chains" | "fees" | "yields" | "stablecoins", ...) -> str
```

**Actions:**
- `tvl_rank` — Top protocols ranked by TVL. Optional: `chain` filter, `limit`.
- `protocol` — Detailed info for one protocol (`protocol="aave-v3"`): TVL, chains, category, description.
- `chains` — All chains ranked by TVL.
- `fees` — Top protocols by 24h/30d fees and revenue. Optional: `chain` filter.
- `yields` — Top yield pools by APY. Optional: `chain` filter, `min_tvl` (default: $1M).
- `stablecoins` — Stablecoin market overview: supply, peg type, chains.

**Best for:** DeFi fundamental analysis, TVL trends, yield farming research, protocol comparison, chain ecosystem health, stablecoin monitoring.

### fear_greed
Crypto Fear & Greed Index — market sentiment indicator. **No API key needed.**
```
fear_greed(action: "current" | "history" | "realtime") -> str
```

**Actions:**
- `current` — Latest Fear & Greed value (0-100) + classification.
- `history` — Daily historical data (`days=30`, up to 365). Source: Alternative.me.
- `realtime` — High-frequency 5-min interval data (`hours=24`, up to 192h). Source: CoinyBubble.

**Scale:** 0-24 Extreme Fear, 25-49 Fear, 50 Neutral, 51-74 Greed, 75-100 Extreme Greed.
**Best for:** Market timing, risk assessment, contrarian signals.

### finnhub
Financial data from Finnhub: earnings calendar, market news, company profiles. **Requires API key.**
```
finnhub(action: "earnings" | "news" | "crypto_news" | "company", ...) -> str
```

**Actions:**
- `earnings` — Upcoming/recent earnings with EPS estimates. Optional: `from_date`, `to_date`, `symbol`.
- `news` — General market news headlines (100+ sources).
- `crypto_news` — Crypto-specific news from CoinTelegraph, CoinDesk, etc.
- `company` — Company profile: market cap, industry, IPO date (`symbol="AAPL"`).

**Best for:** Tracking crypto-related earnings (COIN, MSTR, RIOT), reading latest market news, researching companies before investing.

### freecrypto
Real-time crypto prices from FreeCryptoAPI (3000+ coins, Binance source). **Requires API key.**
```
freecrypto(action: "price" | "symbols", ...) -> str
```

**Actions:**
- `price` — Live price + 24h change for one or more symbols (`symbol="BTC,ETH,SOL"`).
- `symbols` — List all 3000+ supported cryptocurrencies.

**Symbol format:** Just the base symbol: `BTC`, `ETH`, `SOL` (no quote currency).
**Best for:** Fast price checks with millisecond latency, alternative source when CoinGecko/YFinance are rate-limited.

### Data source decision tree

1. **Exchange-specific data** (e.g. Bitget funding rate, OKX order book) → use exchange tools
2. **Backtesting with exchange data** → `backtest(exchange="binance")`, check `failed_symbols`
3. **Failed symbols / DeFi / small caps** → `coingecko` for OHLCV, feed to `backtest(ohlcv_json=...)`
4. **Mainstream crypto/stocks/macro** → `yfinance` as reliable alternative
5. **Price check / market overview** → `coingecko(action="price")` or `freecrypto(action="price")` for speed
6. **DeFi TVL / protocol analysis** → `defillama(action="tvl_rank")` or `defillama(action="protocol")`
7. **Market sentiment** → `fear_greed(action="current")` for quick check
8. **Trending / hype detection** → `coingecko(action="trending")` for what people are searching
9. **Yield farming research** → `defillama(action="yields", chain="Ethereum")`
10. **Earnings / company research** → `finnhub(action="earnings")` or `finnhub(action="company")`
11. **Crypto news** → `finnhub(action="crypto_news")` for latest headlines
12. **Never give up after one source** — try alternatives before reporting failure

---

## Adding Custom Tools

To add custom tools:
1. Create a class that extends `Tool` in `getall/agent/tools/`
2. Implement `name`, `description`, `parameters`, and `execute`
3. Register it in `AgentLoop._register_default_tools()`
