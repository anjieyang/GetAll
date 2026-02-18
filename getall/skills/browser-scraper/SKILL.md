---
name: browser-scraper
description: "Use AI-controlled browser to scrape websites without APIs. Use when user asks about Dune Analytics, DEXScreener, GMGN, exchange announcements, airdrop eligibility, or any data that requires navigating a real webpage."
metadata: '{"getall":{"always":false}}'
---

# Browser Scraper — AI-Driven Web Automation

You have a `browser_use` tool that drives a real Chromium browser with AI. It's your **last resort** for data that no API tool can provide.

## Hard Rules

1. **API FIRST** — ALWAYS try `market_data`, `coingecko`, `web_fetch`, `web_search` before `browser_use`
2. **BE SPECIFIC** — vague tasks waste steps and fail. Tell the browser exactly what to extract and in what format
3. **START WITH URL** — always provide the `url` parameter when you know the target page
4. **KEEP max_steps LOW** — start with 10-15. Only increase if the task needs multiple page navigations
5. **WARN THE USER** — before invoking, tell the user "let me check via browser, this may take a minute"
6. **CACHE RESULTS** — if you scraped data, save it to a file in workspace for future reference

## When to Use

| Scenario | Use browser_use? | Better alternative |
|---|---|---|
| Dune Analytics query results | YES | No free API |
| DEXScreener / GMGN token data | YES | No public API |
| Exchange announcement pages | YES | `web_fetch` may work for simple pages |
| Twitter/X posts | YES (carefully) | May be blocked by anti-bot |
| Token airdrop eligibility | YES | No API for most airdrops |
| CoinGecko price data | NO | `coingecko` tool |
| Binance market data | NO | `market_data` tool |
| General web search | NO | `web_search` tool |
| Simple article content | NO | `web_fetch` tool |

## Task Writing Guide

### Good Tasks (specific, extractable)

```
browser_use(
  url="https://dune.com/queries/3245678",
  task="Wait for the data table to fully load. Extract all rows from the results table as a JSON array with column headers as keys. Include the query title."
)
```

```
browser_use(
  url="https://dexscreener.com/solana/TOKEN_ADDRESS",
  task="Extract the token name, price, 24h volume, liquidity, market cap, and the top 10 transactions from the page. Return as structured JSON."
)
```

```
browser_use(
  url="https://www.binance.com/en/support/announcement/new-listings",
  task="Find the 5 most recent token listing announcements. For each, extract the title, date, and token name. Return as a JSON array."
)
```

### Bad Tasks (vague, wasteful)

- "Check this website" — what to check? what to extract?
- "Find information about BTC" — use `market_data` instead
- "Search Google for crypto news" — use `web_search` instead

## Workflow

### Step 1: Decide if browser_use is needed

Ask yourself:
- Can `web_fetch(url)` get this data? (for static/simple pages → YES, use web_fetch)
- Can `web_search(query)` find it? (for general info → YES, use web_search)
- Does it require JavaScript rendering, login, or interaction? → use `browser_use`

### Step 2: Prepare and notify user

```
"Let me check that via browser — this takes about 30-60 seconds..."
```

### Step 3: Call with specific task

```
browser_use(
  url="https://target-site.com/page",
  task="[Specific extraction instructions with desired output format]",
  max_steps=15
)
```

### Step 4: Process and present results

- Parse the returned text into structured data
- Present in a clean format (table or summary)
- Save raw data to workspace if valuable for future reference:
  ```
  write_file(path="data/dune_query_3245678.json", content="[scraped data]")
  ```

## Error Handling

| Error | Action |
|---|---|
| "browser-use is not installed" | Tell admin to run `pip install browser-use langchain-openai` |
| Timeout (>180s) | Simplify the task, reduce max_steps, or try `web_fetch` fallback |
| Page blocked by anti-bot | Try `web_fetch` as fallback, or inform user the site blocks automation |
| No extractable content | Retry with more specific task description, or try different URL |

## Cost Awareness

Each `browser_use` call consumes:
- ~5-30 LLM calls (one per browser step) — multiplied token cost
- 200-500MB RAM for Chromium (temporary)
- 30-120 seconds of wall time

**Always prefer API tools.** Reserve `browser_use` for data that truly has no API alternative.
