"""市场行情数据工具 - 价格、衍生品数据、链上数据、市场指标 + 批量扫描"""

import asyncio
import json
import math
import re
from typing import Any

from getall.agent.tools.base import Tool
from getall.trading.data.hub import DataHub


# batch_scan 支持的维度及其对应的 Coinglass 方法名
_BATCH_DIMENSIONS = {
    "oi": "get_exchange_oi",                  # 持仓量 (各交易所实时快照)
    "funding": "get_funding_rate_exchange",    # 资金费率 (交易所列表)
    "long_short": "get_long_short_ratio",     # 全网多空比 (Binance h1 history)
    "taker": "get_taker_buy_sell",            # 主动买卖比
    "cvd": "get_cvd",                         # 累计成交量差
    "liquidations": "get_liquidations",       # 清算数据
    "whale": "get_whale_transfers",           # 巨鲸转账 (chain/v2/whale-transfer)
    "coin_flow": "get_coin_flow",             # 合约资金流向 (futures/netflow-list)
    "spot_netflow": "get_spot_netflow",       # 现货资金流入流出 (spot/netflow-list)
    "large_orderbook": "get_large_orderbook", # 大额挂单 (futures/orderbook/large-limit-order)
}


class MarketDataTool(Tool):
    """获取加密货币市场行情数据的工具，覆盖价格、衍生品、链上、情绪指标等维度。
    支持 batch_scan / multi_price 批量操作，大幅减少 cron 任务的 tool call 次数。
    """

    def __init__(self, hub: DataHub):
        # 注入 DataHub，统一管理所有数据源
        self.hub = hub

    @property
    def name(self) -> str:
        return "market_data"

    @property
    def description(self) -> str:
        return (
            "Get crypto market data: prices, derivatives, options, spot, on-chain, "
            "and market indicators.\n"
            "Batch operations:\n"
            "- batch_scan: multi-coin × multi-dimension scan\n"
            "- multi_price / batch_klines / volatility_rank\n"
            "Global (no symbol needed):\n"
            "- token_unlocks: upcoming token unlock schedules\n"
            "- liquidation_coin_list: all coins liquidation ranking (24h/12h/4h/1h)\n"
            "- ahr999: Bitcoin AHR999 valuation index\n"
            "- top_funding_rates / top_oi / top_liquidations\n"
            "Per-coin:\n"
            "- price, klines, ticker_24h, open_interest, oi_history\n"
            "- funding_rate, funding_rate_exchange, long_short_ratio\n"
            "- taker_buy_sell, cvd, coin_flow, spot_netflow\n"
            "- liquidations, large_orderbook, whale_transfers\n"
            "- basis_history: futures-spot basis (arbitrage analysis)\n"
            "- spot_pairs: spot pairs data by exchange\n"
            "- option_info: options OI & volume overview\n"
            "- option_max_pain: options max pain by expiry\n"
            "- fear_greed: crypto fear & greed index"
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "The type of market data to fetch",
                    "enum": [
                        # ── 批量操作 ──
                        "batch_scan", "multi_price", "binance_symbols", "batch_klines",
                        "volatility_rank",
                        # ── 全市场 (no symbol needed) ──
                        "top_funding_rates", "top_oi", "top_liquidations",
                        "token_unlocks", "liquidation_coin_list", "ahr999",
                        # ── 单币: 行情 ──
                        "price", "klines", "ticker_24h",
                        # ── 单币: 持仓量 ──
                        "open_interest", "oi_history",
                        # ── 单币: 资金费率 ──
                        "funding_rate", "funding_rate_exchange",
                        # ── 单币: 多空比 ──
                        "long_short_ratio",
                        # ── 单币: 主动买卖 / CVD ──
                        "taker_buy_sell", "taker_buy_sell_history", "cvd",
                        "coin_flow", "spot_netflow",
                        # ── 单币: 清算 & 大单 ──
                        "liquidations", "large_orderbook",
                        # ── 单币: 期现差价 ──
                        "basis_history",
                        # ── 单币: 现货 ──
                        "spot_pairs",
                        # ── 单币: 期权 ──
                        "option_info", "option_max_pain",
                        # ── 单币: 链上 ──
                        "whale_transfers",
                        # ── 单币: 指标 ──
                        "fear_greed",
                    ],
                },
                "symbol": {
                    "type": "string",
                    "description": "Trading pair symbol, e.g. 'BTC/USDT' or 'BTC'. Required for single-coin actions.",
                },
                "symbols": {
                    "type": "string",
                    "description": (
                        "Comma-separated symbols for batch actions. "
                        "Accepts base coins (e.g. 'BTC,ETH,SOL') or trading pairs "
                        "(e.g. 'BTC/USDT,ETH/USDT'). Required for batch_scan, multi_price, and batch_klines."
                    ),
                },
                "symbols_key": {
                    "type": "string",
                    "description": "Cache key returned by binance_symbols; use it instead of passing a huge symbols list.",
                },
                "dimensions": {
                    "type": "string",
                    "description": (
                        "Comma-separated data dimensions for batch_scan. "
                        "Available: oi,funding,long_short,taker,cvd,"
                        "liquidations,whale,coin_flow,spot_netflow,large_orderbook. "
                        "Default: oi,funding,long_short,taker,cvd,liquidations"
                    ),
                },
                "exchange": {
                    "type": "string",
                    "description": "Exchange name, e.g. 'binance', 'okx'. Optional for most actions.",
                },
                "timeframe": {
                    "type": "string",
                    "description": "Timeframe for klines / history, e.g. '1h', '4h', '1d'. Default varies by action.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max number of results to return",
                    "minimum": 1,
                    "maximum": 500,
                },
                "quote": {
                    "type": "string",
                    "description": "Quote currency for building pairs from base coins. Default: USDT.",
                },
                "date": {
                    "type": "string",
                    "description": "Date (UTC) in YYYY-MM-DD for batch_klines, used to compute start/end time range.",
                },
                "start_time": {
                    "type": "integer",
                    "description": "Start time (ms since epoch) for batch_klines.",
                },
                "end_time": {
                    "type": "integer",
                    "description": "End time (ms since epoch) for batch_klines.",
                },
                "min_drop_pct": {
                    "type": "number",
                    "description": "Filter results where (low-open)/open <= -min_drop_pct. Example: 40.",
                },
                "max_symbols": {
                    "type": "integer",
                    "description": "Max symbols to process for binance_symbols or batch_klines (universe scan).",
                    "minimum": 1,
                    "maximum": 5000,
                },
                "range": {
                    "type": "string",
                    "description": "Universe range: top50/top100 or a number. Used by volatility_rank.",
                },
                "window_hours": {
                    "type": "integer",
                    "description": "Lookback window hours for volatility_rank (default 24).",
                    "minimum": 1,
                    "maximum": 720,
                },
                "rank_size": {
                    "type": "integer",
                    "description": "How many gainers/losers to show (default 20).",
                    "minimum": 1,
                    "maximum": 200,
                },
            },
            "required": ["action"],
        }

    async def execute(self, **kwargs: Any) -> str:
        action: str = kwargs["action"]
        symbol: str | None = kwargs.get("symbol")
        symbols_str: str | None = kwargs.get("symbols")
        symbols_key: str | None = kwargs.get("symbols_key")
        exchange: str | None = kwargs.get("exchange")
        timeframe: str | None = kwargs.get("timeframe")
        limit: int | None = kwargs.get("limit")
        dimensions_str: str | None = kwargs.get("dimensions")
        quote: str | None = kwargs.get("quote")
        date: str | None = kwargs.get("date")
        start_time: int | None = kwargs.get("start_time")
        end_time: int | None = kwargs.get("end_time")
        min_drop_pct: float | None = kwargs.get("min_drop_pct")
        max_symbols: int | None = kwargs.get("max_symbols")
        range_str: str | None = kwargs.get("range")
        window_hours: int | None = kwargs.get("window_hours")
        rank_size: int | None = kwargs.get("rank_size")

        try:
            # ── 批量操作: batch_scan / multi_price ──
            if action == "batch_scan":
                return await self._batch_scan(
                    symbols_str=symbols_str or "",
                    dimensions_str=dimensions_str,
                )
            if action == "multi_price":
                return await self._multi_price(
                    symbols_str=symbols_str or "",
                    exchange=exchange,
                )
            if action == "binance_symbols":
                return await self._binance_symbols(
                    quote=quote,
                    max_symbols=max_symbols,
                )
            if action == "batch_klines":
                return await self._batch_klines(
                    symbols_str=symbols_str,
                    symbols_key=symbols_key,
                    exchange=exchange,
                    timeframe=timeframe,
                    limit=limit,
                    quote=quote,
                    date=date,
                    start_time=start_time,
                    end_time=end_time,
                    min_drop_pct=min_drop_pct,
                    max_symbols=max_symbols,
                )
            if action == "volatility_rank":
                return await self._volatility_rank(
                    symbols_str=symbols_str,
                    symbols_key=symbols_key,
                    exchange=exchange,
                    timeframe=timeframe,
                    quote=quote,
                    date=date,
                    start_time=start_time,
                    end_time=end_time,
                    range_str=range_str,
                    window_hours=window_hours,
                    rank_size=rank_size,
                    max_symbols=max_symbols,
                )

            # ── 全市场排行: top_funding_rates / top_oi / top_liquidations ──
            if action == "top_funding_rates":
                return await self._top_funding_rates(limit=limit or 5)
            if action == "top_oi":
                return await self._top_oi(limit=limit or 10)
            if action == "top_liquidations":
                return await self._top_liquidations(timeframe=timeframe or "1h", limit=limit or 10)

            # ── 全市场数据 (no symbol needed) ──
            if action == "token_unlocks":
                return await self._token_unlocks(limit=limit)
            if action == "liquidation_coin_list":
                return await self._liquidation_coin_list(limit=limit)
            if action == "ahr999":
                return await self._ahr999()

            # ── 单币操作: 需要 symbol ──
            if not symbol:
                return "Error: 'symbol' is required for this action."
            handler = self._get_handler(action)
            if handler is None:
                return f"Error: unknown action '{action}'"
            result = await handler(
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                limit=limit,
            )
            return result
        except Exception as e:
            return f"Error fetching {action}: {e}"

    def _get_handler(self, action: str):
        """根据 action 名称返回对应的异步处理函数"""
        mapping = {
            # 基础行情
            "price": self._price,
            "klines": self._klines,
            "ticker_24h": self._ticker_24h,
            # 持仓量
            "open_interest": self._open_interest,
            "oi_history": self._oi_history,
            # 资金费率
            "funding_rate": self._funding_rate,
            "funding_rate_exchange": self._funding_rate_exchange,
            # 多空比
            "long_short_ratio": self._long_short_ratio,
            # 主动买卖 / CVD
            "taker_buy_sell": self._taker_buy_sell,
            "taker_buy_sell_history": self._taker_buy_sell_history,
            "cvd": self._cvd,
            "coin_flow": self._coin_flow,
            "spot_netflow": self._spot_netflow,
            # 清算 & 大单
            "liquidations": self._liquidations,
            "large_orderbook": self._large_orderbook,
            # 期现差价
            "basis_history": self._basis_history,
            # 现货
            "spot_pairs": self._spot_pairs,
            # 期权
            "option_info": self._option_info,
            "option_max_pain": self._option_max_pain,
            # 链上
            "whale_transfers": self._whale_transfers,
            # 指标
            "fear_greed": self._fear_greed,
        }
        return mapping.get(action)

    # ══════════════════════════════════════════════
    # 批量操作
    # ══════════════════════════════════════════════

    async def _batch_scan(self, symbols_str: str, dimensions_str: str | None) -> str:
        """批量扫描多币种×多维度数据 (anomaly-scan cron 专用, 一次 tool call 完成全部扫描).

        并发调用 Coinglass API, 每个 (symbol, dimension) 组合独立请求,
        失败不影响其他组合, 最终返回结构化摘要供 LLM 做阈值判断.
        """
        # 解析参数
        symbols = [s.strip().upper() for s in symbols_str.split(",") if s.strip()]
        if not symbols:
            return "Error: 'symbols' is required for batch_scan, e.g. 'BTC,ETH,SOL'"

        # 解析维度列表 (默认: 异动扫描最常用的 6 个)
        default_dims = "oi,funding,long_short,taker,cvd,liquidations"
        dim_names = [d.strip() for d in (dimensions_str or default_dims).split(",") if d.strip()]
        # 过滤无效维度
        valid_dims = [d for d in dim_names if d in _BATCH_DIMENSIONS]
        if not valid_dims:
            return f"Error: no valid dimensions. Available: {', '.join(_BATCH_DIMENSIONS.keys())}"

        # 构造并发任务: 每个 (symbol, dim) 一个协程
        tasks: list[tuple[str, str, Any]] = []  # (symbol, dim, coroutine)
        for sym in symbols:
            for dim in valid_dims:
                method_name = _BATCH_DIMENSIONS[dim]
                method = getattr(self.hub.coinglass, method_name)
                # 根据方法签名传参
                if dim in ("whale",):
                    coro = method(symbol=sym, limit=5)
                elif dim in ("long_short", "cvd", "taker"):
                    coro = method(symbol=sym, interval="h1")
                else:
                    coro = method(symbol=sym)
                tasks.append((sym, dim, coro))

        # 并发执行 (使用 semaphore 控制并发量, 避免 API 限流)
        sem = asyncio.Semaphore(8)
        results: dict[str, dict[str, Any]] = {sym: {} for sym in symbols}

        async def _run(sym: str, dim: str, coro: Any) -> None:
            async with sem:
                try:
                    data = await coro
                    results[sym][dim] = data
                except Exception as e:
                    results[sym][dim] = {"error": str(e)}

        await asyncio.gather(*[_run(s, d, c) for s, d, c in tasks])

        # 格式化输出: 每个币种一个区块, 每个维度压缩为关键数值摘要
        lines = [f"📊 Batch Market Scan ({len(symbols)} coins × {len(valid_dims)} dimensions)"]
        lines.append("═" * 60)

        for sym in symbols:
            lines.append(f"\n── {sym} ──")
            sym_data = results[sym]
            for dim in valid_dims:
                data = sym_data.get(dim)
                summary = self._summarize_dimension(dim, data)
                lines.append(f"  {dim}: {summary}")

        lines.append(f"\n{'═' * 60}")
        lines.append(f"Total API calls: {len(tasks)} | Symbols: {', '.join(symbols)}")
        lines.append("Use these values against anomaly thresholds to detect unusual activity.")
        return "\n".join(lines)

    async def _multi_price(self, symbols_str: str, exchange: str | None) -> str:
        """批量获取多个币种的当前价格 (morning-briefing / 快速概览专用).

        并发调用交易所 API 获取价格, 一次 tool call 返回全部结果.
        """
        symbols = [s.strip() for s in symbols_str.split(",") if s.strip()]
        if not symbols:
            return "Error: 'symbols' is required for multi_price, e.g. 'BTC/USDT,ETH/USDT,SOL/USDT'"

        # Normalize base coins (BTC) to trading pairs (BTC/USDT) for exchange tickers.
        normalized_symbols = [self._pair_symbol(s) for s in symbols]

        ex = self._resolve_exchange(exchange)
        sem = asyncio.Semaphore(5)
        results: dict[str, dict[str, Any]] = {}

        async def _fetch(sym: str) -> None:
            async with sem:
                try:
                    data = await ex.get_ticker(sym)
                    results[sym] = data
                except Exception as e:
                    results[sym] = {"error": str(e)}

        await asyncio.gather(*[_fetch(s) for s in normalized_symbols])

        lines = [f"📊 Multi-Price ({len(normalized_symbols)} symbols)"]
        lines.append("═" * 60)
        for sym in normalized_symbols:
            d = results[sym]
            if isinstance(d, dict) and "error" in d:
                lines.append(f"  {sym}: ❌ {d['error']}")
            else:
                price = d.get("last", "N/A")
                change = d.get("change_pct", "N/A")
                high = d.get("high", "N/A")
                low = d.get("low", "N/A")
                vol = d.get("quote_volume", "N/A")
                lines.append(
                    f"  {sym}: ${price}  ({change:+.2f}% 24h)"
                    if isinstance(change, (int, float))
                    else f"  {sym}: ${price}  ({change} 24h)"
                )
                lines.append(f"    H: {high} | L: {low} | Vol: {vol}")
        return "\n".join(lines)

    async def _binance_symbols(self, quote: str | None, max_symbols: int | None) -> str:
        """从 Coinglass 获取 Binance 合约市场币种 Universe (base symbols).

        结果会写入 hub.cache, 并返回 cache key，便于后续 batch_klines 复用而无需把全部 symbols 塞进 prompt。
        """
        q = (quote or "USDT").strip().upper() or "USDT"

        base_symbols, cache_key = await self._get_binance_universe_cached(
            quote=q,
            max_symbols=max_symbols,
        )

        # Build a short preview. The full list is stored in cache.
        pairs_preview = [f"{s}/{q}" for s in base_symbols[:50]]
        lines = [
            f"📋 Binance Symbols Universe (Coinglass futures/coins-markets)",
            "═" * 60,
            f"Quote: {q}",
            f"Total base symbols: {len(base_symbols)}",
            f"Cache key: {cache_key}",
        ]
        if pairs_preview:
            lines.append("\nPreview (first 50 pairs):")
            lines.append(", ".join(pairs_preview))
        lines.append(
            "\n💡 Use batch_klines with symbols_key=<Cache key> to scan without passing a huge symbols list."
        )
        return "\n".join(lines)

    async def _batch_klines(
        self,
        symbols_str: str | None,
        symbols_key: str | None,
        exchange: str | None,
        timeframe: str | None,
        limit: int | None,
        quote: str | None,
        date: str | None,
        start_time: int | None,
        end_time: int | None,
        min_drop_pct: float | None,
        max_symbols: int | None,
        **_,
    ) -> str:
        """批量获取 K 线数据 (ExchangeAdapter), 支持按日期/时间窗统计涨跌幅."""
        ex = self._resolve_exchange(exchange)
        if ex is None:
            return "Error: no exchange configured. Check exchanges.yaml."

        tf = self._normalize_timeframe(timeframe or "1h")
        q = (quote or "USDT").strip().upper() or "USDT"

        # Resolve time range (optional)
        try:
            start_ms, end_ms = self._resolve_time_range(date=date, start_time=start_time, end_time=end_time)
        except ValueError as e:
            return f"Error: {e}"

        # Resolve symbols
        raw_symbols: list[str] = []
        used_cache_key: str | None = None
        if symbols_key:
            raw_symbols = self._load_symbols_from_cache(symbols_key)
            used_cache_key = symbols_key
            if not raw_symbols:
                return f"Error: symbols_key not found or empty: {symbols_key}"
        elif symbols_str:
            raw_symbols = [s.strip() for s in symbols_str.split(",") if s.strip()]
        else:
            # Default to Binance universe if caller did not pass symbols.
            base_symbols, cache_key = await self._get_binance_universe_cached(
                quote=q,
                max_symbols=max_symbols,
            )
            raw_symbols = base_symbols
            used_cache_key = cache_key

        if not raw_symbols:
            return "Error: 'symbols' or 'symbols_key' is required for batch_klines."

        if max_symbols:
            raw_symbols = raw_symbols[: max_symbols]

        # Normalize symbols to pairs for exchange queries.
        symbols = [self._pair_symbol(s, default_quote=q) for s in raw_symbols]

        # Decide fetch limit.
        request_limit = limit if limit is not None else 100
        if start_ms is not None and end_ms is not None:
            tf_ms = self._timeframe_to_millis(tf)
            est = self._estimate_candle_count(tf_ms, start_ms, end_ms)
            if est > 1000:
                return (
                    f"Error: timeframe/range too large for one fetch (need ~{est} candles, ccxt usually caps at 1000).\n"
                    f"Try a larger timeframe (e.g. 15m/1h/4h/1d) or a shorter range."
                )
            request_limit = est if limit is None else max(request_limit, est)
        request_limit = min(request_limit, 1000)

        sem = asyncio.Semaphore(5)
        results: list[dict[str, Any]] = []
        errors: list[str] = []

        async def _fetch(sym: str) -> None:
            async with sem:
                try:
                    data = await ex.get_klines(sym, tf, limit=request_limit, since=start_ms)
                    if isinstance(data, dict) and "error" in data:
                        errors.append(f"{sym}: {data['error']}")
                        return
                    if not data:
                        errors.append(f"{sym}: no data")
                        return

                    if start_ms is None and end_ms is None:
                        # No range requested: keep a compact last-candle snapshot.
                        last = data[-1]
                        results.append(
                            {
                                "symbol": sym,
                                "timestamp": last.get("timestamp"),
                                "open": last.get("open"),
                                "high": last.get("high"),
                                "low": last.get("low"),
                                "close": last.get("close"),
                                "volume": last.get("volume"),
                            }
                        )
                        return

                    stats = self._compute_range_stats(data, start_ms=start_ms, end_ms=end_ms)
                    if stats is None:
                        errors.append(f"{sym}: no candles in range")
                        return
                    stats["symbol"] = sym
                    results.append(stats)
                except Exception as e:
                    errors.append(f"{sym}: {e}")

        await asyncio.gather(*[_fetch(s) for s in symbols])

        # Filter/sort results (range mode)
        drop_threshold = float(min_drop_pct) if min_drop_pct is not None else None
        if start_ms is not None or end_ms is not None:
            if drop_threshold is not None:
                threshold = -abs(drop_threshold)
                results = [r for r in results if (r.get("low_change_pct") is not None and r["low_change_pct"] <= threshold)]
            results.sort(key=lambda r: r.get("low_change_pct", 0))

        # Format output
        header = [
            f"📈 Batch Klines ({ex.name if hasattr(ex, 'name') else (exchange or 'default')}, {tf})",
            "═" * 60,
        ]
        if date:
            header.append(f"Date (UTC): {date}")
        if start_ms is not None or end_ms is not None:
            header.append(f"Range: {self._fmt_ts(start_ms) if start_ms else 'N/A'} -> {self._fmt_ts(end_ms) if end_ms else 'N/A'}")
        if used_cache_key:
            header.append(f"Symbols source cache: {used_cache_key}")
        header.append(f"Processed: {len(symbols)} | OK: {len(results)} | Errors: {len(errors)}")
        if drop_threshold is not None and (start_ms is not None or end_ms is not None):
            header.append(f"Filter: (low-open)/open <= -{abs(drop_threshold):.2f}%")

        lines = header

        if not results:
            lines.append("No results.")
        else:
            # Limit printed rows for readability
            max_rows = 50
            for i, r in enumerate(results[:max_rows], 1):
                if "low_change_pct" in r:
                    lines.append(
                        f"{i}. {r['symbol']} | "
                        f"drop:{self._fmt_pct(r.get('low_change_pct'))} "
                        f"change:{self._fmt_pct(r.get('change_pct'))} "
                        f"O:{self._fmt_num(r.get('open'))} C:{self._fmt_num(r.get('close'))} "
                        f"L:{self._fmt_num(r.get('low'))} H:{self._fmt_num(r.get('high'))}"
                    )
                else:
                    ts = r.get("timestamp") or 0
                    lines.append(
                        f"{i}. {r['symbol']} {self._fmt_ts(ts)} | "
                        f"O:{self._fmt_num(r.get('open'))} H:{self._fmt_num(r.get('high'))} "
                        f"L:{self._fmt_num(r.get('low'))} C:{self._fmt_num(r.get('close'))} "
                        f"V:{self._fmt_num(r.get('volume'))}"
                    )
            if len(results) > max_rows:
                lines.append(f"... ({len(results) - max_rows} more omitted)")

        if errors:
            lines.append("")
            lines.append("Errors (sample):")
            lines.append(", ".join(errors[:10]))
            if len(errors) > 10:
                lines.append(f"... ({len(errors) - 10} more omitted)")

        return "\n".join(lines)

    async def _volatility_rank(
        self,
        symbols_str: str | None,
        symbols_key: str | None,
        exchange: str | None,
        timeframe: str | None,
        quote: str | None,
        date: str | None,
        start_time: int | None,
        end_time: int | None,
        range_str: str | None,
        window_hours: int | None,
        rank_size: int | None,
        max_symbols: int | None,
        **_,
    ) -> str:
        """24h gainers/losers ranking for an exchange universe."""
        ex = self._resolve_exchange(exchange)
        if ex is None:
            return "Error: no exchange configured. Check exchanges.yaml."

        tf = self._normalize_timeframe(timeframe or "1h")
        q = (quote or "USDT").strip().upper() or "USDT"
        top_n = int(rank_size or 20)

        # Resolve time range
        if date or start_time is not None or end_time is not None:
            try:
                start_ms, end_ms = self._resolve_time_range(
                    date=date, start_time=start_time, end_time=end_time
                )
            except ValueError as e:
                return f"Error: {e}"
        else:
            from datetime import datetime, timezone, timedelta

            hours = int(window_hours or 24)
            now = datetime.now(tz=timezone.utc)
            start_ms = int((now - timedelta(hours=hours)).timestamp() * 1000)
            end_ms = int(now.timestamp() * 1000)

        # Resolve universe size
        universe_size = self._parse_range_size(range_str, max_symbols)

        # Resolve symbols
        raw_symbols: list[str] = []
        used_cache_key: str | None = None
        if symbols_key:
            raw_symbols = self._load_symbols_from_cache(symbols_key)
            used_cache_key = symbols_key
        elif symbols_str:
            raw_symbols = [s.strip() for s in symbols_str.split(",") if s.strip()]
        else:
            ex_name = getattr(ex, "name", None) or exchange or "binance"
            base_syms, cache_key = await self._get_exchange_universe_cached(
                exchange_name=str(ex_name),
                quote=q,
                max_symbols=universe_size,
            )
            raw_symbols = base_syms
            used_cache_key = cache_key

        if not raw_symbols:
            return "Error: unable to resolve symbols for volatility_rank."

        if universe_size:
            raw_symbols = raw_symbols[:universe_size]

        symbols = [self._pair_symbol(s, default_quote=q) for s in raw_symbols]

        # Estimate candle limit
        tf_ms = self._timeframe_to_millis(tf)
        est = self._estimate_candle_count(tf_ms, start_ms, end_ms)
        if est > 1000:
            return (
                f"Error: timeframe/range too large for one fetch (need ~{est} candles, ccxt usually caps at 1000).\n"
                f"Try a larger timeframe (e.g. 15m/1h/4h/1d) or a shorter range."
            )

        sem = asyncio.Semaphore(5)
        results: list[dict[str, Any]] = []
        errors: list[str] = []

        async def _fetch(sym: str) -> None:
            async with sem:
                try:
                    data = await ex.get_klines(sym, tf, limit=est, since=start_ms)
                    if isinstance(data, dict) and "error" in data:
                        errors.append(f"{sym}: {data['error']}")
                        return
                    if not data:
                        errors.append(f"{sym}: no data")
                        return
                    stats = self._compute_range_stats(data, start_ms=start_ms, end_ms=end_ms)
                    if stats is None:
                        errors.append(f"{sym}: no candles in range")
                        return
                    stats["symbol"] = sym
                    results.append(stats)
                except Exception as e:
                    errors.append(f"{sym}: {e}")

        await asyncio.gather(*[_fetch(s) for s in symbols])

        # Rankers
        gainers = sorted(results, key=lambda r: r.get("change_pct", 0), reverse=True)[:top_n]
        losers = sorted(results, key=lambda r: r.get("change_pct", 0))[:top_n]

        lines = [
            f"📊 24h Volatility Rank ({getattr(ex, 'name', None) or exchange or 'default'}, {tf})",
            "═" * 60,
            f"Universe: {len(symbols)} | Range: {universe_size or len(symbols)} | Window: {self._fmt_ts(start_ms)} -> {self._fmt_ts(end_ms)}",
        ]
        if used_cache_key:
            lines.append(f"Symbols source cache: {used_cache_key}")
        lines.append(f"Errors: {len(errors)}")
        lines.append("")
        lines.append("Top Gainers:")
        for i, r in enumerate(gainers, 1):
            lines.append(
                f"{i}. {r['symbol']} {self._fmt_pct(r.get('change_pct'))} "
                f"O:{self._fmt_num(r.get('open'))} C:{self._fmt_num(r.get('close'))}"
            )
        lines.append("")
        lines.append("Top Losers:")
        for i, r in enumerate(losers, 1):
            lines.append(
                f"{i}. {r['symbol']} {self._fmt_pct(r.get('change_pct'))} "
                f"O:{self._fmt_num(r.get('open'))} C:{self._fmt_num(r.get('close'))}"
            )

        if errors:
            lines.append("")
            lines.append("Errors (sample):")
            lines.append(", ".join(errors[:10]))
            if len(errors) > 10:
                lines.append(f"... ({len(errors) - 10} more omitted)")

        return "\n".join(lines)

    async def _top_funding_rates(self, limit: int) -> str:
        """全市场资金费率排行榜 (按 OI 加权平均资金费率排序).

        返回资金费率最高/最低的币种, 用于发现高费率套利机会.
        """
        data = await self.hub.coinglass.get_coins_markets()
        if isinstance(data, dict) and "error" in data:
            return f"Error: {data['error']}"
        if not isinstance(data, list):
            return "Error: Invalid data format from coins_markets"

        # 按资金费率排序 (取绝对值最大的, 高费率和负费率都有意义)
        sorted_data = sorted(
            data,
            key=lambda x: abs(x.get("avg_funding_rate_by_oi", 0) or 0),
            reverse=True,
        )

        lines = [f"📊 Top {limit} Funding Rates (OI-weighted)"]
        lines.append("═" * 60)
        lines.append("Sorted by absolute value | Positive=多付空 | Negative=空付多\n")

        for i, coin in enumerate(sorted_data[:limit], 1):
            sym = coin.get("symbol", "?")
            rate = coin.get("avg_funding_rate_by_oi", 0) or 0
            oi_usd = coin.get("open_interest_usd", 0) or 0
            price = coin.get("current_price", 0) or 0

            # 年化费率 (资金费率通常 8h 一次, 一年 365*3=1095 次)
            rate_annual = rate * 1095 * 100  # 转为百分比

            direction = "📈 多付空" if rate > 0 else "📉 空付多"
            lines.append(
                f"{i}. {sym}: {rate:+.4%} ({direction})"
                f" | 年化: {rate_annual:+.1f}%"
                f" | OI: ${oi_usd:,.0f} | 价格: ${price:,.2f}"
            )

        return "\n".join(lines)

    async def _top_oi(self, limit: int) -> str:
        """全市场持仓量排行榜.

        返回持仓量最大的币种及其变化情况.
        """
        data = await self.hub.coinglass.get_coins_markets()
        if isinstance(data, dict) and "error" in data:
            return f"Error: {data['error']}"
        if not isinstance(data, list):
            return "Error: Invalid data format from coins_markets"

        # 按持仓量 USD 排序
        sorted_data = sorted(
            data,
            key=lambda x: x.get("open_interest_usd", 0) or 0,
            reverse=True,
        )

        lines = [f"📊 Top {limit} Open Interest"]
        lines.append("═" * 60)

        for i, coin in enumerate(sorted_data[:limit], 1):
            sym = coin.get("symbol", "?")
            oi_usd = coin.get("open_interest_usd", 0) or 0
            oi_chg_1h = coin.get("open_interest_change_percent_1h", 0) or 0
            oi_chg_24h = coin.get("open_interest_change_percent_24h", 0) or 0
            price = coin.get("current_price", 0) or 0

            lines.append(
                f"{i}. {sym}: ${oi_usd:,.0f}"
                f" | 1h: {oi_chg_1h:+.2f}% | 24h: {oi_chg_24h:+.2f}%"
                f" | 价格: ${price:,.2f}"
            )

        return "\n".join(lines)

    async def _top_liquidations(self, timeframe: str, limit: int) -> str:
        """全市场清算排行榜 (最近 N 小时清算量最大的币种).

        timeframe: "1h", "4h", "12h", "24h"
        """
        data = await self.hub.coinglass.get_coins_markets()
        if isinstance(data, dict) and "error" in data:
            return f"Error: {data['error']}"
        if not isinstance(data, list):
            return "Error: Invalid data format from coins_markets"

        # 根据 timeframe 选择对应的字段
        tf_map = {
            "1h": "liquidation_usd_1h",
            "4h": "liquidation_usd_4h",
            "12h": "liquidation_usd_12h",
            "24h": "liquidation_usd_24h",
        }
        liq_field = tf_map.get(timeframe, "liquidation_usd_1h")
        long_field = liq_field.replace("liquidation", "long_liquidation")
        short_field = liq_field.replace("liquidation", "short_liquidation")

        # 按清算总量排序
        sorted_data = sorted(
            data,
            key=lambda x: x.get(liq_field, 0) or 0,
            reverse=True,
        )

        lines = [f"📊 Top {limit} Liquidations ({timeframe})"]
        lines.append("═" * 60)

        for i, coin in enumerate(sorted_data[:limit], 1):
            sym = coin.get("symbol", "?")
            total_liq = coin.get(liq_field, 0) or 0
            long_liq = coin.get(long_field, 0) or 0
            short_liq = coin.get(short_field, 0) or 0
            price = coin.get("current_price", 0) or 0

            # 判断爆仓方向
            if long_liq > short_liq * 1.5:
                dom = "📉 多头爆仓为主"
            elif short_liq > long_liq * 1.5:
                dom = "📈 空头爆仓为主"
            else:
                dom = "⚖️  多空均衡"

            lines.append(
                f"{i}. {sym}: ${total_liq:,.0f} ({dom})"
                f" | 多头爆: ${long_liq:,.0f} | 空头爆: ${short_liq:,.0f}"
                f" | 价格: ${price:,.2f}"
            )

        return "\n".join(lines)

    @staticmethod
    def _pair_symbol(symbol: str, *, default_quote: str = "USDT") -> str:
        """
        Normalize various user inputs to a ccxt trading pair for exchange queries.

        Accepts:
        - base coin: "BTC" -> "BTC/USDT"
        - dashed pair: "BTC-USDT" -> "BTC/USDT"
        - ccxt pair: "BTC/USDT" -> "BTC/USDT"
        - ccxt futures: "BTC/USDT:USDT" -> unchanged
        """
        s = (symbol or "").strip()
        if not s:
            return s
        # Normalize common dashed format
        if "-" in s and "/" not in s:
            s = s.replace("-", "/")
        # If user passed a base coin, default to USDT quote
        if "/" not in s and ":" not in s:
            upper = s.upper()
            if upper == default_quote.upper():
                return s
            dq = default_quote.upper()
            if upper.endswith(dq) and len(upper) > len(dq):
                base = upper[: -len(dq)]
                return f"{base}/{dq}"
            return f"{upper}/{default_quote.upper()}"
        return s

    @staticmethod
    def _summarize_dimension(dim: str, data: Any) -> str:
        """将单个维度的 API 返回压缩为一行关键数值摘要 (供 LLM 做阈值判断)."""
        if data is None:
            return "N/A (no data)"
        if isinstance(data, dict) and "error" in data:
            return f"❌ {data['error']}"

        # 根据维度类型提取关键字段
        try:
            if dim == "oi":
                # OI exchange-list: 各交易所实时快照, 取 top3 + 变化率
                if isinstance(data, list) and data:
                    parts = []
                    for item in data[:5]:
                        if isinstance(item, dict):
                            name = item.get("exchangeName", "?")
                            oi = item.get("openInterest", item.get("oi", "N/A"))
                            chg = item.get("change1h", item.get("h1OiChangePercent", ""))
                            chg_str = f"({chg:+.2f}%)" if isinstance(chg, (int, float)) else ""
                            parts.append(f"{name}=${oi}{chg_str}")
                    return " | ".join(parts) if parts else str(data)[:120]
                elif isinstance(data, dict):
                    return f"OI=${data.get('openInterest', data.get('oi', 'N/A'))}"
                return str(data)[:120]

            elif dim == "funding":
                # 资金费率: 通常是交易所列表
                if isinstance(data, list):
                    rates = [
                        f"{item.get('exchangeName', '?')}={item.get('rate', item.get('fundingRate', 'N/A'))}"
                        for item in data[:3]
                    ]
                    return " | ".join(rates) if rates else str(data)[:120]
                return str(data)[:120]

            elif dim in ("long_short",):
                # 多空比 history: 对比最新 vs 1h/4h 前, 展示趋势
                if isinstance(data, list) and len(data) >= 2:
                    def _get_ls(item: dict) -> float | None:
                        v = item.get("longRate", item.get("longAccount"))
                        return float(v) if v is not None else None

                    now = _get_ls(data[-1]) if isinstance(data[-1], dict) else None
                    h1_ago = _get_ls(data[-2]) if len(data) >= 2 and isinstance(data[-2], dict) else None
                    h4_ago = _get_ls(data[-5]) if len(data) >= 5 and isinstance(data[-5], dict) else None

                    parts = [f"long={now}%"]
                    if now is not None and h1_ago is not None:
                        parts.append(f"1h_chg={now - h1_ago:+.2f}%")
                    if now is not None and h4_ago is not None:
                        parts.append(f"4h_chg={now - h4_ago:+.2f}%")
                    return " | ".join(parts)
                elif isinstance(data, dict):
                    return f"long={data.get('longRate', data.get('longAccount', 'N/A'))}"
                return str(data)[:120]

            elif dim == "taker":
                # 主动买卖比 history: 对比最新 vs 1h 前的买卖力度
                if isinstance(data, list) and len(data) >= 2:
                    def _get_buy_ratio(item: dict) -> float | None:
                        buy = float(item.get("taker_buy_volume_usd", 0) or 0)
                        sell = float(item.get("taker_sell_volume_usd", 0) or 0)
                        total = buy + sell
                        return (buy / total * 100) if total > 0 else None

                    now = _get_buy_ratio(data[-1]) if isinstance(data[-1], dict) else None
                    h1_ago = _get_buy_ratio(data[-2]) if len(data) >= 2 and isinstance(data[-2], dict) else None

                    parts = []
                    if now is not None:
                        direction = "买盘占优" if now > 50 else ("卖盘占优" if now < 50 else "买卖均衡")
                        parts.append(f"buy={now:.1f}% ({direction})")
                    if now is not None and h1_ago is not None:
                        delta = now - h1_ago
                        parts.append(f"1h_chg={delta:+.1f}%")
                    return " | ".join(parts) if parts else str(data)[:120]
                return str(data)[:120]

            elif dim == "cvd":
                # CVD history: 对比最新 vs 1h/4h 前, 展示趋势和方向
                if isinstance(data, list) and len(data) >= 2:
                    def _get_cvd(item: dict) -> float | None:
                        v = item.get("cvd", item.get("value"))
                        return float(v) if v is not None else None

                    now = _get_cvd(data[-1]) if isinstance(data[-1], dict) else None
                    h1_ago = _get_cvd(data[-2]) if len(data) >= 2 and isinstance(data[-2], dict) else None
                    h4_ago = _get_cvd(data[-5]) if len(data) >= 5 and isinstance(data[-5], dict) else None

                    parts = [f"cvd={now}"]
                    if now is not None and h1_ago is not None:
                        delta = now - h1_ago
                        parts.append(f"1h_delta={delta:+.0f}({'买>卖' if delta > 0 else '卖>买'})")
                    if now is not None and h4_ago is not None:
                        delta = now - h4_ago
                        parts.append(f"4h_delta={delta:+.0f}")
                    return " | ".join(parts)
                return str(data)[:120]

            elif dim == "liquidations":
                # 清算 history: 汇总最近 1h/4h 的总清算量
                if isinstance(data, list) and data:
                    def _sum_liq(items: list, key_long: str, key_short: str) -> tuple[float, float]:
                        total_l, total_s = 0.0, 0.0
                        for it in items:
                            if isinstance(it, dict):
                                total_l += float(it.get(key_long, it.get("longVolUsd", 0)) or 0)
                                total_s += float(it.get(key_short, it.get("shortVolUsd", 0)) or 0)
                        return total_l, total_s

                    # 最近 1 条 = 最近 1h, 最近 4 条 = 最近 4h
                    l1, s1 = _sum_liq(data[-1:], "longLiquidationUsd", "shortLiquidationUsd")
                    l4, s4 = _sum_liq(data[-4:], "longLiquidationUsd", "shortLiquidationUsd")
                    total_1h = l1 + s1
                    total_4h = l4 + s4
                    dom = "多头爆仓为主" if l1 > s1 * 1.5 else ("空头爆仓为主" if s1 > l1 * 1.5 else "多空均衡")

                    return (
                        f"1h: long_liq=${l1:,.0f}, short_liq=${s1:,.0f}, total=${total_1h:,.0f} ({dom})"
                        f" | 4h_total=${total_4h:,.0f}"
                    )
                return str(data)[:120]

            elif dim == "whale":
                # 巨鲸转账: 显示最近几条
                if isinstance(data, list):
                    count = len(data)
                    return f"{count} recent transfers" + (
                        f" (latest: {data[0].get('amount', '?')} @ {data[0].get('blockchain', '?')})"
                        if count > 0 and isinstance(data[0], dict)
                        else ""
                    )
                return str(data)[:120]

            elif dim == "coin_flow":
                if isinstance(data, dict):
                    return f"netFlow={data.get('netFlow', data.get('inflowVolUsd', 'N/A'))}"
                return str(data)[:120]

            elif dim == "spot_netflow":
                # 现货资金流入流出: 各交易所净流入/流出
                if isinstance(data, list) and data:
                    parts = []
                    for item in data[:5]:
                        if isinstance(item, dict):
                            name = item.get("exchangeName", "?")
                            net = item.get("netFlow", item.get("netInflowVolUsd", "N/A"))
                            net_str = f"${net:+,.0f}" if isinstance(net, (int, float)) else str(net)
                            parts.append(f"{name}={net_str}")
                    return " | ".join(parts) if parts else str(data)[:120]
                return str(data)[:120]

            elif dim == "large_orderbook":
                if isinstance(data, dict):
                    return f"bidWall={data.get('bidSize', 'N/A')}, askWall={data.get('askSize', 'N/A')}"
                elif isinstance(data, list) and data:
                    return f"{len(data)} large orders"
                return str(data)[:120]

        except Exception:
            pass

        # 兜底: 截断原始数据
        raw = str(data)
        return raw[:150] + "..." if len(raw) > 150 else raw

    # ──────────────────────────────────────────────
    # 基础行情
    # ──────────────────────────────────────────────

    async def _price(self, symbol: str, exchange: str | None, **_) -> str:
        """获取当前价格"""
        ex = self._resolve_exchange(exchange)
        sym = self._pair_symbol(symbol)
        data = await ex.get_price(sym)
        if isinstance(data, dict) and "error" in data:
            return f"Error: {data['error']}"
        return (
            f"📊 {sym} Price\n"
            f"  Last: {data.get('price', 'N/A')}\n"
            f"  Bid: {data.get('bid', 'N/A')}  |  Ask: {data.get('ask', 'N/A')}"
        )

    async def _klines(self, symbol: str, exchange: str | None, timeframe: str | None, limit: int | None, **_) -> str:
        """获取 K 线数据"""
        ex = self._resolve_exchange(exchange)
        sym = self._pair_symbol(symbol)
        tf = timeframe or "1h"
        lim = limit or 50
        ohlcv = await ex.get_klines(sym, tf, limit=lim)
        if isinstance(ohlcv, dict) and "error" in ohlcv:
            return f"Error: {ohlcv['error']}"
        if not ohlcv:
            return f"No kline data for {sym} ({tf})"
        # 格式化最近的 K 线
        lines = [f"📈 {sym} Klines ({tf}, last {len(ohlcv)} bars)"]
        lines.append("  Time | Open | High | Low | Close | Volume")
        lines.append("  " + "-" * 60)
        for candle in ohlcv[-20:]:  # 最多展示 20 根
            ts = candle.get("timestamp", 0)
            o = candle.get("open", "N/A")
            h = candle.get("high", "N/A")
            lo = candle.get("low", "N/A")
            c = candle.get("close", "N/A")
            v = candle.get("volume", "N/A")
            lines.append(f"  {self._fmt_ts(ts)} | {o} | {h} | {lo} | {c} | {v}")
        if len(ohlcv) > 20:
            lines.append(f"  ... ({len(ohlcv) - 20} more bars omitted)")
        return "\n".join(lines)

    async def _ticker_24h(self, symbol: str, exchange: str | None, **_) -> str:
        """获取 24 小时 Ticker"""
        ex = self._resolve_exchange(exchange)
        sym = self._pair_symbol(symbol)
        t = await ex.get_ticker(sym)
        if isinstance(t, dict) and "error" in t:
            return f"Error: {t['error']}"
        return (
            f"📋 {sym} 24h Ticker\n"
            f"  Last: {t.get('last', 'N/A')}\n"
            f"  24h Change: {t.get('change_pct', 'N/A')}%\n"
            f"  24h High: {t.get('high', 'N/A')}  |  24h Low: {t.get('low', 'N/A')}\n"
            f"  24h Volume (quote): {t.get('quote_volume', 'N/A')}"
        )

    # ──────────────────────────────────────────────
    # 持仓量 (Open Interest)
    # ──────────────────────────────────────────────

    async def _open_interest(self, symbol: str, **_) -> str:
        """获取当前总持仓量"""
        data = await self.hub.coinglass.get_aggregated_oi(symbol=self._base_symbol(symbol))
        return self._format_response(f"{symbol} Open Interest", data)

    async def _oi_history(self, symbol: str, timeframe: str | None, limit: int | None, **_) -> str:
        """获取持仓量历史"""
        tf = timeframe or "1h"
        data = await self.hub.coinglass.get_oi_history(
            symbol=self._base_symbol(symbol), interval=tf, limit=limit or 50
        )
        return self._format_response(f"{symbol} OI History ({tf})", data)

    # ──────────────────────────────────────────────
    # 资金费率 (Funding Rate)
    # ──────────────────────────────────────────────

    async def _funding_rate(self, symbol: str, **_) -> str:
        """获取当前资金费率"""
        data = await self.hub.coinglass.get_funding_rate_history(symbol=self._base_symbol(symbol))
        return self._format_response(f"{symbol} Funding Rate", data)

    async def _funding_rate_exchange(self, symbol: str, exchange: str | None, **_) -> str:
        """获取指定交易所的资金费率"""
        data = await self.hub.coinglass.get_funding_rate_exchange(
            symbol=self._base_symbol(symbol)
        )
        return self._format_response(f"{symbol} Funding Rate ({exchange or 'all exchanges'})", data)

    # ──────────────────────────────────────────────
    # 多空比 (Long/Short Ratio)
    # ──────────────────────────────────────────────

    async def _long_short_ratio(self, symbol: str, timeframe: str | None, **_) -> str:
        """获取全网多空比"""
        tf = timeframe or "1h"
        data = await self.hub.coinglass.get_long_short_ratio(
            symbol=self._base_symbol(symbol), interval=tf
        )
        return self._format_response(f"{symbol} Long/Short Ratio ({tf})", data)

    # ──────────────────────────────────────────────
    # 主动买卖 / CVD (Taker Buy/Sell)
    # ──────────────────────────────────────────────

    async def _taker_buy_sell(self, symbol: str, timeframe: str | None, limit: int | None, **_) -> str:
        """获取主动买卖比 (使用 history 端点)"""
        tf = timeframe or "h1"
        lim = limit or 10
        data = await self.hub.coinglass.get_taker_buy_sell(
            symbol=self._base_symbol(symbol), interval=tf, limit=lim
        )
        return self._format_response(f"{symbol} Taker Buy/Sell ({tf})", data)

    async def _taker_buy_sell_history(self, symbol: str, timeframe: str | None, limit: int | None, **_) -> str:
        """获取主动买卖历史"""
        tf = timeframe or "1h"
        data = await self.hub.coinglass.get_taker_buy_sell_history(
            symbol=self._base_symbol(symbol), interval=tf, limit=limit or 50
        )
        return self._format_response(f"{symbol} Taker Buy/Sell History ({tf})", data)

    async def _cvd(self, symbol: str, timeframe: str | None, **_) -> str:
        """获取累计成交量差 (CVD)"""
        tf = timeframe or "1h"
        data = await self.hub.coinglass.get_cvd(
            symbol=self._base_symbol(symbol), interval=tf
        )
        return self._format_response(f"{symbol} CVD ({tf})", data)

    async def _coin_flow(self, symbol: str, **_) -> str:
        """获取合约资金流向"""
        data = await self.hub.coinglass.get_coin_flow(symbol=self._base_symbol(symbol))
        return self._format_response(f"{symbol} Coin Flow", data)

    async def _spot_netflow(self, symbol: str, **_) -> str:
        """获取现货资金流入流出 (各交易所)"""
        data = await self.hub.coinglass.get_spot_netflow(symbol=self._base_symbol(symbol))
        return self._format_response(f"{symbol} Spot Net Flow", data)

    # ──────────────────────────────────────────────
    # 清算 & 大单 (Liquidations & Orderbook)
    # ──────────────────────────────────────────────

    async def _liquidations(self, symbol: str, **_) -> str:
        """获取清算数据"""
        data = await self.hub.coinglass.get_liquidations(symbol=self._base_symbol(symbol))
        return self._format_response(f"{symbol} Liquidations", data)

    async def _large_orderbook(self, symbol: str, **_) -> str:
        """获取大单挂单数据"""
        data = await self.hub.coinglass.get_large_orderbook(symbol=self._base_symbol(symbol))
        return self._format_response(f"{symbol} Large Orderbook", data)

    # ──────────────────────────────────────────────
    # 期现差价 (Basis)
    # ──────────────────────────────────────────────

    async def _basis_history(self, symbol: str, exchange: str | None = None, timeframe: str | None = None, **_) -> str:
        """获取期现差价历史"""
        data = await self.hub.coinglass.get_basis_history(
            symbol=self._base_symbol(symbol),
            exchange=exchange or "Binance",
            interval=timeframe or "h1",
        )
        return self._format_response(f"{symbol} Futures Basis History", data)

    # ──────────────────────────────────────────────
    # 现货 (Spot)
    # ──────────────────────────────────────────────

    async def _spot_pairs(self, symbol: str, **_) -> str:
        """获取现货交易对数据（按交易所）"""
        data = await self.hub.coinglass.get_spot_pairs(symbol=self._base_symbol(symbol))
        return self._format_response(f"{symbol} Spot Pairs by Exchange", data)

    # ──────────────────────────────────────────────
    # 期权 (Options)
    # ──────────────────────────────────────────────

    async def _option_info(self, symbol: str, **_) -> str:
        """获取期权市场概览（OI、成交量、按交易所分）"""
        data = await self.hub.coinglass.get_option_info(symbol=self._base_symbol(symbol))
        return self._format_response(f"{symbol} Options Market Overview", data)

    async def _option_max_pain(self, symbol: str, exchange: str | None = None, **_) -> str:
        """获取期权最大痛点（按到期日）"""
        data = await self.hub.coinglass.get_option_max_pain(
            symbol=self._base_symbol(symbol),
            exchange=exchange or "Deribit",
        )
        return self._format_response(f"{symbol} Options Max Pain", data)

    # ──────────────────────────────────────────────
    # 链上数据 (On-chain)
    # ──────────────────────────────────────────────

    async def _whale_transfers(self, symbol: str, limit: int | None, **_) -> str:
        """获取巨鲸转账记录"""
        data = await self.hub.coinglass.get_whale_transfers(
            symbol=self._base_symbol(symbol), limit=limit or 20
        )
        return self._format_response(f"{symbol} Whale Transfers", data)

    # ──────────────────────────────────────────────
    # 市场指标 (Indicators)
    # ──────────────────────────────────────────────

    async def _fear_greed(self, symbol: str, **_) -> str:
        """获取恐惧贪婪指数"""
        data = await self.hub.coinglass.get_fear_greed()
        return self._format_response("Fear & Greed Index", data)

    # ──────────────────────────────────────────────
    # AHR999 Bitcoin Valuation
    # ──────────────────────────────────────────────

    async def _ahr999(self, **_) -> str:
        """获取 AHR999 BTC 估值指标"""
        data = await self.hub.coinglass.get_ahr999()
        if isinstance(data, list) and len(data) > 30:
            # Only return recent 30 entries to keep context manageable
            data = data[-30:]
        return self._format_response("AHR999 Bitcoin Valuation Index", data)

    # ──────────────────────────────────────────────
    # Liquidation Coin List (全币种清算排行)
    # ──────────────────────────────────────────────

    async def _liquidation_coin_list(self, limit: int | None = None, **_) -> str:
        """获取全币种清算排行（24h/12h/4h/1h）"""
        data = await self.hub.coinglass.get_liquidation_coin_list()
        if isinstance(data, dict) and "error" in data:
            return self._format_response("Liquidation Coin List", data)
        if not isinstance(data, list):
            return self._format_response("Liquidation Coin List", {"error": "Unexpected response"})

        # Sort by 24h total liquidation USD descending
        data.sort(
            key=lambda t: float(t.get("liquidation_usd_24h") or 0),
            reverse=True,
        )
        cap = min(limit or 20, 50)
        top = data[:cap]

        results = []
        for t in top:
            results.append({
                "symbol": t.get("symbol"),
                "liq_24h_usd": round(float(t.get("liquidation_usd_24h") or 0), 2),
                "long_liq_24h": round(float(t.get("long_liquidation_usd_24h") or 0), 2),
                "short_liq_24h": round(float(t.get("short_liquidation_usd_24h") or 0), 2),
                "liq_4h_usd": round(float(t.get("liquidation_usd_4h") or 0), 2),
                "liq_1h_usd": round(float(t.get("liquidation_usd_1h") or 0), 2),
            })
        return self._format_response("Liquidation Coin Ranking (top by 24h)", {
            "count": len(results),
            "coins": results,
        })

    # ──────────────────────────────────────────────
    # Token Unlocks
    # ──────────────────────────────────────────────

    async def _token_unlocks(self, limit: int | None = None, **_) -> str:
        """获取即将到来的代币解锁时间表"""
        # Fetch enough tokens to have a good selection after filtering
        per_page = min(max(limit or 50, 50), 100)
        data = await self.hub.coinglass.get_token_unlocks(per_page=per_page)
        if isinstance(data, dict) and "error" in data:
            return self._format_response("Token Unlocks", data)

        if not isinstance(data, list):
            return self._format_response("Token Unlocks", {"error": "Unexpected response"})

        # Keep tokens that have a next unlock date (Coinglass auto-updates after unlock)
        upcoming = [t for t in data if t.get("next_unlock_date")]
        # Sort by next unlock date ascending (soonest first)
        upcoming.sort(key=lambda t: t["next_unlock_date"])

        results = []
        for t in upcoming:
            from datetime import datetime, timezone
            unlock_dt = datetime.fromtimestamp(
                t["next_unlock_date"] / 1000, tz=timezone.utc
            )
            results.append({
                "symbol": t.get("symbol"),
                "name": t.get("name"),
                "price": t.get("price"),
                "market_cap": t.get("market_cap"),
                "next_unlock_date": unlock_dt.strftime("%Y-%m-%d"),
                "next_unlock_tokens": t.get("next_unlock_tokens"),
                "next_unlock_usd": t.get("next_unlock_usd"),
                "next_unlock_pct_circulating": round(
                    (t.get("next_unlock_of_circulating") or 0) * 1, 4
                ),
                "next_unlock_pct_supply": round(
                    (t.get("next_unlock_of_supply") or 0) * 1, 4
                ),
                "circulating_supply": t.get("circulating_supply"),
                "total_locked": t.get("total_locked"),
            })

        return self._format_response("Token Unlocks (upcoming)", {
            "count": len(results),
            "tokens": results,
        })

    # ──────────────────────────────────────────────
    # 工具方法
    # ──────────────────────────────────────────────

    def _resolve_exchange(self, exchange: str | None):
        """解析交易所实例：指定名称则获取对应交易所，否则用默认"""
        if exchange:
            return self.hub.get_exchange_sync(exchange)
        return self.hub.exchange

    def _cache_get(self, key: str) -> Any | None:
        cache = getattr(self.hub, "cache", None)
        if cache is None:
            return None
        try:
            return cache.get(key)
        except Exception:
            return None

    def _cache_set(self, key: str, value: Any, ttl_seconds: int) -> None:
        cache = getattr(self.hub, "cache", None)
        if cache is None:
            return
        try:
            cache.set(key, value, ttl_seconds=ttl_seconds)
        except Exception:
            return

    def _load_symbols_from_cache(self, key: str) -> list[str]:
        cached = self._cache_get(key)
        if isinstance(cached, list):
            return [str(x) for x in cached if str(x).strip()]
        if isinstance(cached, dict):
            for k in ("base", "symbols", "pairs"):
                v = cached.get(k)
                if isinstance(v, list):
                    return [str(x) for x in v if str(x).strip()]
        return []

    async def _get_binance_universe_cached(
        self, *, quote: str, max_symbols: int | None
    ) -> tuple[list[str], str]:
        return await self._get_exchange_universe_cached(
            exchange_name="Binance",
            quote=quote,
            max_symbols=max_symbols,
        )

    async def _get_exchange_universe_cached(
        self, *, exchange_name: str, quote: str, max_symbols: int | None
    ) -> tuple[list[str], str]:
        ex_key = (exchange_name or "binance").strip().lower()
        cache_key = f"universe:coinglass:{ex_key}:futures:{quote}"
        cached_syms = self._load_symbols_from_cache(cache_key)
        if cached_syms:
            return (cached_syms[: max_symbols] if max_symbols else cached_syms), cache_key

        per_page = 200
        page = 1
        out: list[str] = []
        seen: set[str] = set()

        # Coinglass expects exchange name in its canonical form (e.g., Binance, OKX)
        ex_name = exchange_name.strip()
        if not ex_name:
            ex_name = "Binance"

        while True:
            data = await self.hub.coinglass.get_coins_markets(
                exchange_list=ex_name,
                per_page=per_page,
                page=page,
            )
            if isinstance(data, dict) and "error" in data:
                raise ValueError(f"Coinglass error: {data['error']}")
            if not isinstance(data, list) or not data:
                break

            for item in data:
                if not isinstance(item, dict):
                    continue
                sym = item.get("symbol")
                if not sym:
                    continue
                s = str(sym).strip().upper()
                if not s or s in seen:
                    continue
                seen.add(s)
                out.append(s)
                if max_symbols and len(out) >= max_symbols:
                    break

            if max_symbols and len(out) >= max_symbols:
                break
            if len(data) < per_page:
                break

            page += 1
            if page > 50:  # safety guard
                break

        self._cache_set(cache_key, {"base": out, "quote": quote}, ttl_seconds=6 * 3600)
        return out, cache_key

    @staticmethod
    def _base_symbol(symbol: str) -> str:
        """从交易对中提取基础币种，如 'BTC/USDT' -> 'BTC'"""
        s = (symbol or "").strip().upper()
        s = s.split(":")[0]
        if "/" in s:
            s = s.split("/")[0]
        for suffix in ("USDT", "USDC", "BUSD", "USD"):
            if s.endswith(suffix) and len(s) > len(suffix):
                s = s[: -len(suffix)]
                break
        return s

    @staticmethod
    def _parse_range_size(range_str: str | None, max_symbols: int | None) -> int | None:
        if max_symbols:
            return int(max_symbols)
        if not range_str:
            return None
        s = range_str.strip().lower()
        if s.startswith("top"):
            s = s[3:]
        try:
            value = int(s)
            return value if value > 0 else None
        except ValueError:
            return None

    @staticmethod
    def _normalize_timeframe(timeframe: str) -> str:
        s = (timeframe or "").strip().lower().replace(" ", "")
        if not s:
            return "1h"

        # Already in ccxt format
        m = re.fullmatch(r"(\\d+)([mhdw])", s)
        if m:
            return f"{int(m.group(1))}{m.group(2)}"

        # Common aliases: 15min, 1hour, 1day, ...
        m = re.fullmatch(r"(\\d+)(min|minute|minutes|h|hour|hours|d|day|days|w|week|weeks)", s)
        if not m:
            return s
        value = int(m.group(1))
        unit = m.group(2)
        if unit.startswith("m"):
            return f"{value}m"
        if unit.startswith("h"):
            return f"{value}h"
        if unit.startswith("d"):
            return f"{value}d"
        if unit.startswith("w"):
            return f"{value}w"
        return s

    @staticmethod
    def _timeframe_to_millis(timeframe: str) -> int:
        m = re.fullmatch(r"(\\d+)([mhdw])", timeframe)
        if not m:
            raise ValueError(f"invalid timeframe: {timeframe}")
        value = int(m.group(1))
        unit = m.group(2)
        mult = {"m": 60_000, "h": 3_600_000, "d": 86_400_000, "w": 604_800_000}
        return value * mult[unit]

    @staticmethod
    def _estimate_candle_count(tf_ms: int, start_ms: int, end_ms: int) -> int:
        span_ms = max(0, int(end_ms) - int(start_ms)) + 1
        return max(2, int(math.ceil(span_ms / tf_ms)) + 2)

    @staticmethod
    def _resolve_time_range(
        *, date: str | None, start_time: int | None, end_time: int | None
    ) -> tuple[int | None, int | None]:
        if date:
            from datetime import datetime, timezone

            try:
                dt = datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            except ValueError as e:
                raise ValueError(f"invalid date format: {date} (expected YYYY-MM-DD)") from e
            start_ms = int(dt.timestamp() * 1000)
            end_ms = start_ms + 86_400_000 - 1
            return start_ms, end_ms

        s = int(start_time) if start_time is not None else None
        e = int(end_time) if end_time is not None else None
        if s is not None and e is not None and s > e:
            raise ValueError("start_time must be <= end_time")
        return s, e

    @staticmethod
    def _to_float(value: Any) -> float | None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _compute_range_stats(
        self, candles: list[dict[str, Any]], *, start_ms: int | None, end_ms: int | None
    ) -> dict[str, Any] | None:
        def _ts(item: dict[str, Any]) -> int | None:
            try:
                return int(item.get("timestamp"))
            except (TypeError, ValueError):
                return None

        filtered: list[dict[str, Any]] = []
        for c in candles:
            if not isinstance(c, dict):
                continue
            ts = _ts(c)
            if ts is None:
                continue
            if start_ms is not None and ts < start_ms:
                continue
            if end_ms is not None and ts > end_ms:
                continue
            filtered.append(c)

        if not filtered:
            return None

        o = self._to_float(filtered[0].get("open"))
        c = self._to_float(filtered[-1].get("close"))
        if o is None or o == 0 or c is None:
            return None

        highs = [self._to_float(x.get("high")) for x in filtered]
        lows = [self._to_float(x.get("low")) for x in filtered]
        highs2 = [x for x in highs if x is not None]
        lows2 = [x for x in lows if x is not None]
        if not highs2 or not lows2:
            return None

        h = max(highs2)
        l = min(lows2)

        change_pct = (c - o) / o * 100
        low_change_pct = (l - o) / o * 100
        high_change_pct = (h - o) / o * 100

        return {
            "start_ts": filtered[0].get("timestamp"),
            "end_ts": filtered[-1].get("timestamp"),
            "open": o,
            "close": c,
            "high": h,
            "low": l,
            "change_pct": change_pct,
            "low_change_pct": low_change_pct,
            "high_change_pct": high_change_pct,
        }

    @staticmethod
    def _fmt_num(value: Any) -> str:
        try:
            v = float(value)
        except (TypeError, ValueError):
            return "N/A"
        if abs(v) >= 1000:
            return f"{v:,.2f}"
        return f"{v:.6g}"

    @staticmethod
    def _fmt_pct(value: Any) -> str:
        try:
            v = float(value)
        except (TypeError, ValueError):
            return "N/A"
        return f"{v:+.2f}%"

    @staticmethod
    def _fmt_ts(ts: int | float) -> str:
        """将毫秒时间戳格式化为可读字符串"""
        from datetime import datetime, timezone
        dt = datetime.fromtimestamp(ts / 1000, tz=timezone.utc)
        return dt.strftime("%m-%d %H:%M")

    @staticmethod
    def _format_response(title: str, data: Any) -> str:
        """将 API 返回的数据格式化为 LLM 可读的文本"""
        header = f"📊 {title}\n{'─' * 40}\n"

        if data is None:
            return header + "No data available."

        # 如果是列表，逐项格式化
        if isinstance(data, list):
            if not data:
                return header + "Empty result set."
            lines = []
            for i, item in enumerate(data[:30], 1):  # 最多 30 条
                if isinstance(item, dict):
                    parts = [f"  {k}: {v}" for k, v in item.items()]
                    lines.append(f"[{i}]\n" + "\n".join(parts))
                else:
                    lines.append(f"  {i}. {item}")
            result = header + "\n".join(lines)
            if len(data) > 30:
                result += f"\n  ... ({len(data) - 30} more items omitted)"
            return result

        # 如果是字典，展开 key-value
        if isinstance(data, dict):
            lines = [f"  {k}: {v}" for k, v in data.items()]
            return header + "\n".join(lines)

        # 其他类型直接转字符串
        return header + str(data)
