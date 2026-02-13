"""Trading action log - records all real trading operations for auditing and analysis."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

from getall.utils.atomic_io import get_atomic_writer


class TradingActionLog:
    """
    Records all real trading operations (not paper trades) to actions.jsonl.

    Provides a persistent audit trail for:
    - Order placement
    - Order cancellation
    - Stop-loss/take-profit settings

    Each action is logged with full context (timestamp, exchange, symbol, parameters, result).
    """

    def __init__(self, trading_dir: Path):
        """
        Initialize the action log.

        Args:
            trading_dir: Base trading directory (usually workspace/trading/).
        """
        self.log_path = trading_dir / "actions.jsonl"
        self._writer = get_atomic_writer()

    async def log_order(
        self,
        action: str,
        exchange: str,
        symbol: str,
        side: str,
        order_type: str,
        amount: float,
        price: float | None = None,
        leverage: int | None = None,
        order_id: str | None = None,
        status: str = "success",
        error: str | None = None,
        stop_loss: float | None = None,
        take_profit: float | None = None,
        **extra: Any,
    ) -> bool:
        """
        Log a trading order action.

        Args:
            action: Action type (place_order, cancel_order, set_stop_loss, set_take_profit).
            exchange: Exchange name (binance, bitget, etc.).
            symbol: Trading pair symbol.
            side: Order side (buy/sell).
            order_type: Order type (market, limit, stop_market, etc.).
            amount: Order amount.
            price: Order price (None for market orders).
            leverage: Leverage multiplier (for futures).
            order_id: Exchange order ID.
            status: Execution status (success, failed).
            error: Error message if status is failed.
            stop_loss: Stop loss price (if applicable).
            take_profit: Take profit price (if applicable).
            **extra: Additional fields to log.

        Returns:
            True if logged successfully, False otherwise.
        """
        record: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": action,
            "exchange": exchange,
            "symbol": symbol,
            "side": side,
            "type": order_type,
            "amount": amount,
            "status": status,
        }

        # Optional fields
        if price is not None:
            record["price"] = price
        if leverage is not None:
            record["leverage"] = leverage
        if order_id is not None:
            record["order_id"] = order_id
        if error is not None:
            record["error"] = error
        if stop_loss is not None:
            record["stop_loss"] = stop_loss
        if take_profit is not None:
            record["take_profit"] = take_profit

        # Merge extra fields
        record.update(extra)

        success = await self._writer.append_json_line(self.log_path, record)
        if success:
            logger.debug(f"Trading action logged: {action} {symbol} {side}")
        else:
            logger.error(f"Failed to log trading action: {action} {symbol}")

        return success

    def read_recent(self, limit: int = 50) -> list[dict[str, Any]]:
        """
        Read recent trading actions.

        Args:
            limit: Maximum number of actions to return (most recent first).

        Returns:
            List of action records, newest first.
        """
        if not self.log_path.exists():
            return []

        try:
            import json
            actions = []
            with open(self.log_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        actions.append(json.loads(line))

            # Return most recent first
            return actions[-limit:][::-1] if len(actions) > limit else actions[::-1]
        except Exception as e:
            logger.error(f"Failed to read trading action log: {e}")
            return []

    def read_by_symbol(self, symbol: str, limit: int = 20) -> list[dict[str, Any]]:
        """
        Read recent actions for a specific symbol.

        Args:
            symbol: Trading pair symbol (e.g., "BTC/USDT:USDT").
            limit: Maximum number of actions to return.

        Returns:
            List of action records for the symbol, newest first.
        """
        all_actions = self.read_recent(limit=200)  # Read more to filter
        symbol_actions = [a for a in all_actions if a.get("symbol") == symbol]
        return symbol_actions[:limit]

    def get_summary(self, days: int = 7) -> dict[str, Any]:
        """
        Get a summary of recent trading activity.

        Args:
            days: Number of days to look back.

        Returns:
            Summary statistics (total orders, success rate, symbols traded, etc.).
        """
        from datetime import timedelta

        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        all_actions = self.read_recent(limit=500)

        # Filter by time
        recent = [
            a for a in all_actions
            if datetime.fromisoformat(a.get("timestamp", "1970-01-01T00:00:00+00:00")) >= cutoff
        ]

        if not recent:
            return {
                "period_days": days,
                "total_actions": 0,
                "success_count": 0,
                "failed_count": 0,
                "symbols": [],
            }

        success = [a for a in recent if a.get("status") == "success"]
        failed = [a for a in recent if a.get("status") == "failed"]
        symbols = list(set(a.get("symbol") for a in recent if a.get("symbol")))

        return {
            "period_days": days,
            "total_actions": len(recent),
            "success_count": len(success),
            "failed_count": len(failed),
            "success_rate": f"{len(success) / len(recent) * 100:.1f}%" if recent else "N/A",
            "symbols": sorted(symbols),
            "actions_by_type": self._count_by_field(recent, "action"),
        }

    @staticmethod
    def _count_by_field(records: list[dict[str, Any]], field: str) -> dict[str, int]:
        """Count records grouped by a field value."""
        counts: dict[str, int] = {}
        for r in records:
            value = r.get(field, "unknown")
            counts[value] = counts.get(value, 0) + 1
        return counts
