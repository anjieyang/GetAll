"""DefiLlama tool — on-chain DeFi data (TVL, protocols, chains, fees, yields).

100% free, no API key required. Covers 5000+ protocols across 200+ chains.
This is a NATIVE data capability for the agent.

Actions:
  - tvl_rank:   Top protocols ranked by TVL.
  - protocol:   Detailed info for a specific protocol (TVL, chains, category).
  - chains:     All chains ranked by TVL.
  - fees:       Top protocols by fees/revenue (24h or 30d).
  - yields:     Top yield pools by APY.
  - stablecoins: Stablecoin market overview.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
from loguru import logger

from getall.agent.tools.base import Tool

_BASE = "https://api.llama.fi"
_YIELDS_BASE = "https://yields.llama.fi"
_STABLECOINS_BASE = "https://stablecoins.llama.fi"
_TIMEOUT = 30.0


async def _get(url: str, params: dict[str, Any] | None = None) -> Any:
    """GET with error handling."""
    async with httpx.AsyncClient(timeout=_TIMEOUT) as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        return resp.json()


class DefiLlamaTool(Tool):
    """DefiLlama — on-chain DeFi data (TVL, protocols, chains, fees, yields). No API key needed."""

    @property
    def name(self) -> str:
        return "defillama"

    @property
    def description(self) -> str:
        return (
            "On-chain DeFi data from DefiLlama (5000+ protocols, 200+ chains). "
            "No API key needed. "
            "Actions: 'tvl_rank' (top protocols by TVL), 'protocol' (detail for one protocol), "
            "'chains' (chain TVL ranking), 'fees' (protocol fees/revenue), "
            "'yields' (top yield pools by APY), 'stablecoins' (stablecoin market cap overview). "
            "Use for: DeFi fundamental analysis, TVL trends, yield farming research, "
            "protocol comparison, chain ecosystem health."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "description": "Action to perform",
                    "enum": ["tvl_rank", "protocol", "chains", "fees", "yields", "stablecoins"],
                },
                "protocol": {
                    "type": "string",
                    "description": (
                        "For 'protocol': protocol slug (e.g. 'aave-v3', 'lido', 'uniswap-v3'). "
                        "Use tvl_rank first if you don't know the slug."
                    ),
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results to return. Default: 20, max: 100",
                    "minimum": 1,
                    "maximum": 100,
                },
                "chain": {
                    "type": "string",
                    "description": (
                        "Filter by chain name (e.g. 'Ethereum', 'Solana', 'Arbitrum'). "
                        "For 'yields': filter pools by chain."
                    ),
                },
                "min_tvl": {
                    "type": "number",
                    "description": "For 'yields': minimum TVL in USD to filter pools. Default: 1000000",
                },
            },
            "required": ["action"],
        }

    async def execute(self, **kwargs: Any) -> str:
        action = kwargs.get("action", "")
        try:
            if action == "tvl_rank":
                return await self._tvl_rank(**kwargs)
            if action == "protocol":
                return await self._protocol(**kwargs)
            if action == "chains":
                return await self._chains(**kwargs)
            if action == "fees":
                return await self._fees(**kwargs)
            if action == "yields":
                return await self._yields(**kwargs)
            if action == "stablecoins":
                return await self._stablecoins(**kwargs)
            return json.dumps({"error": f"Unknown action '{action}'"})
        except httpx.HTTPStatusError as e:
            logger.error(f"DefiLlama HTTP error: {e}")
            return json.dumps({"error": f"DefiLlama API error: {e.response.status_code}"})
        except Exception as e:
            logger.error(f"DefiLlama tool error: {e}")
            return json.dumps({"error": str(e)})

    # ── tvl_rank ──

    async def _tvl_rank(self, **kwargs: Any) -> str:
        limit = min(kwargs.get("limit", 20) or 20, 100)
        chain_filter = (kwargs.get("chain") or "").strip()

        data = await _get(f"{_BASE}/protocols")
        if not isinstance(data, list):
            return json.dumps({"error": "Unexpected response format"})

        # Filter by chain if specified
        if chain_filter:
            cf_lower = chain_filter.lower()
            data = [
                p for p in data
                if any(c.lower() == cf_lower for c in (p.get("chains") or []))
            ]

        # Sort by TVL descending (already sorted, but ensure)
        data.sort(key=lambda p: float(p.get("tvl") or 0), reverse=True)

        results = []
        for p in data[:limit]:
            results.append({
                "name": p.get("name"),
                "slug": p.get("slug"),
                "category": p.get("category"),
                "tvl": round(float(p.get("tvl") or 0), 2),
                "change_1d": p.get("change_1d"),
                "change_7d": p.get("change_7d"),
                "chains": p.get("chains", []),
                "symbol": p.get("symbol"),
            })
        return json.dumps({
            "action": "tvl_rank",
            "chain_filter": chain_filter or None,
            "count": len(results),
            "protocols": results,
        })

    # ── protocol ──

    async def _protocol(self, **kwargs: Any) -> str:
        slug = (kwargs.get("protocol") or "").strip()
        if not slug:
            return json.dumps({"error": "protocol slug is required (e.g. 'aave-v3', 'lido')"})

        data = await _get(f"{_BASE}/protocol/{slug}")
        if not isinstance(data, dict) or "name" not in data:
            return json.dumps({"error": f"Protocol not found: {slug}"})

        # Extract chain TVLs (non-staking, non-borrowed)
        chain_tvls: dict[str, float] = {}
        for k, v in (data.get("chainTvls") or {}).items():
            if "-" not in k and k not in ("staking", "borrowed", "pool2", "vesting"):
                chain_tvls[k] = round(float(v), 2)

        # Sort chains by TVL
        sorted_chains = sorted(chain_tvls.items(), key=lambda x: x[1], reverse=True)

        return json.dumps({
            "name": data.get("name"),
            "slug": slug,
            "category": data.get("category"),
            "description": data.get("description"),
            "tvl": round(float(data.get("tvl") or 0), 2),
            "change_1d": data.get("change_1d"),
            "change_7d": data.get("change_7d"),
            "chains": data.get("chains", []),
            "chain_tvls": dict(sorted_chains[:20]),
            "symbol": data.get("symbol"),
            "url": data.get("url"),
            "twitter": data.get("twitter"),
            "mcap": data.get("mcap"),
        })

    # ── chains ──

    async def _chains(self, **kwargs: Any) -> str:
        limit = min(kwargs.get("limit", 30) or 30, 100)

        data = await _get(f"{_BASE}/v2/chains")
        if not isinstance(data, list):
            return json.dumps({"error": "Unexpected response format"})

        # Sort by TVL descending
        data.sort(key=lambda c: float(c.get("tvl") or 0), reverse=True)

        results = []
        for c in data[:limit]:
            tvl = float(c.get("tvl") or 0)
            if tvl <= 0:
                continue
            results.append({
                "name": c.get("name"),
                "tvl": round(tvl, 2),
                "token_symbol": c.get("tokenSymbol"),
                "gecko_id": c.get("gecko_id"),
            })
        return json.dumps({"action": "chains", "count": len(results), "chains": results})

    # ── fees ──

    async def _fees(self, **kwargs: Any) -> str:
        limit = min(kwargs.get("limit", 20) or 20, 100)
        chain_filter = (kwargs.get("chain") or "").strip()

        url = f"{_BASE}/overview/fees"
        params: dict[str, str] = {
            "excludeTotalDataChart": "true",
            "excludeTotalDataChartBreakdown": "true",
        }
        if chain_filter:
            params["chain"] = chain_filter

        data = await _get(url, params)
        protocols = data.get("protocols", [])
        if not isinstance(protocols, list):
            return json.dumps({"error": "No fee data available"})

        # Sort by 24h fees descending
        protocols.sort(key=lambda p: float(p.get("total24h") or 0), reverse=True)

        results = []
        for p in protocols[:limit]:
            total_24h = p.get("total24h")
            total_30d = p.get("total30d")
            if total_24h is None and total_30d is None:
                continue
            results.append({
                "name": p.get("name") or p.get("displayName"),
                "category": p.get("category"),
                "fees_24h": round(float(total_24h), 2) if total_24h is not None else None,
                "fees_30d": round(float(total_30d), 2) if total_30d is not None else None,
                "change_1d": p.get("change_1d"),
                "chains": p.get("chains", []),
            })
        return json.dumps({
            "action": "fees",
            "chain_filter": chain_filter or None,
            "total_24h": data.get("total24h"),
            "total_30d": data.get("total30d"),
            "count": len(results),
            "protocols": results,
        })

    # ── yields ──

    async def _yields(self, **kwargs: Any) -> str:
        limit = min(kwargs.get("limit", 20) or 20, 100)
        chain_filter = (kwargs.get("chain") or "").strip()
        min_tvl = float(kwargs.get("min_tvl", 1_000_000) or 1_000_000)

        data = await _get(f"{_YIELDS_BASE}/pools")
        pools = data.get("data", [])
        if not isinstance(pools, list):
            return json.dumps({"error": "No yield data available"})

        # Filter: min TVL and optionally by chain
        filtered = []
        for p in pools:
            tvl = float(p.get("tvlUsd") or 0)
            if tvl < min_tvl:
                continue
            if chain_filter and (p.get("chain") or "").lower() != chain_filter.lower():
                continue
            apy = float(p.get("apy") or 0)
            if apy <= 0 or apy > 10000:  # skip unrealistic APYs
                continue
            filtered.append(p)

        # Sort by APY descending
        filtered.sort(key=lambda p: float(p.get("apy") or 0), reverse=True)

        results = []
        for p in filtered[:limit]:
            results.append({
                "pool": p.get("pool"),
                "project": p.get("project"),
                "symbol": p.get("symbol"),
                "chain": p.get("chain"),
                "apy": round(float(p.get("apy") or 0), 2),
                "apy_base": round(float(p.get("apyBase") or 0), 2),
                "apy_reward": round(float(p.get("apyReward") or 0), 2),
                "tvl_usd": round(float(p.get("tvlUsd") or 0), 2),
                "stable_coin": p.get("stablecoin", False),
            })
        return json.dumps({
            "action": "yields",
            "chain_filter": chain_filter or None,
            "min_tvl": min_tvl,
            "count": len(results),
            "pools": results,
        })

    # ── stablecoins ──

    async def _stablecoins(self, **kwargs: Any) -> str:
        limit = min(kwargs.get("limit", 20) or 20, 100)

        data = await _get(f"{_STABLECOINS_BASE}/stablecoins?includePrices=true")
        coins = data.get("peggedAssets", [])
        if not isinstance(coins, list):
            return json.dumps({"error": "No stablecoin data available"})

        # Sort by circulating supply (pegged to USD)
        def _mcap(c: dict[str, Any]) -> float:
            chains = c.get("chainCirculating", {})
            total = 0.0
            for chain_data in chains.values():
                current = chain_data.get("current", {})
                total += float(current.get("peggedUSD") or 0)
            return total

        coins.sort(key=_mcap, reverse=True)

        results = []
        for c in coins[:limit]:
            mcap = _mcap(c)
            if mcap <= 0:
                continue
            results.append({
                "name": c.get("name"),
                "symbol": c.get("symbol"),
                "peg_type": c.get("pegType"),
                "peg_mechanism": c.get("pegMechanism"),
                "circulating_usd": round(mcap, 2),
                "gecko_id": c.get("gecko_id"),
                "chains": list((c.get("chainCirculating") or {}).keys())[:10],
            })
        return json.dumps({
            "action": "stablecoins",
            "count": len(results),
            "stablecoins": results,
        })
