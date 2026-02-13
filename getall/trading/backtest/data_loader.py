"""数据加载: 将 ccxt OHLCV 数据转换为 Nautilus Trader 格式.

支持:
  - 从 ccxt (list[dict]) 直接加载
  - 从 pandas DataFrame 加载
  - 写入 / 读取 Parquet catalog
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger


class OHLCVLoader:
    """将 ccxt OHLCV 数据转换为 Nautilus Trader 可用的格式."""

    @staticmethod
    def ccxt_to_dataframe(
        ohlcv_list: list[dict[str, Any]],
        symbol: str,
    ) -> pd.DataFrame:
        """将 ccxt get_klines 返回的 list[dict] 转换为标准 DataFrame.

        Args:
            ohlcv_list: ccxt 返回的 OHLCV 数据 (含 timestamp/open/high/low/close/volume).
            symbol: 交易对 (如 'BTC/USDT').

        Returns:
            DataFrame with datetime index and OHLCV columns.
        """
        if not ohlcv_list:
            return pd.DataFrame()

        df = pd.DataFrame(ohlcv_list)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df.set_index("timestamp", inplace=True)

        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df.attrs["symbol"] = symbol
        return df

    @staticmethod
    def write_bars_to_catalog(
        df: pd.DataFrame,
        catalog_path: Path,
        instrument_id: str,
    ) -> None:
        """将 OHLCV DataFrame 写入 Nautilus Parquet catalog.

        Args:
            df: OHLCV DataFrame.
            catalog_path: Parquet catalog 目录路径.
            instrument_id: Nautilus instrument ID (如 'BTCUSDT-PERP.BINANCE').
        """
        try:
            from nautilus_trader.persistence.catalog import ParquetDataCatalog
            from nautilus_trader.persistence.wranglers import BarDataWrangler
            from nautilus_trader.model import BarType, InstrumentId

            catalog = ParquetDataCatalog(str(catalog_path))

            # 构建 bar type
            bar_type = BarType.from_str(f"{instrument_id}-1-MINUTE-LAST-EXTERNAL")

            wrangler = BarDataWrangler(bar_type)
            bars = wrangler.process(df)

            catalog.write_data(bars)
            logger.info(f"Wrote {len(bars)} bars for {instrument_id} to catalog")

        except ImportError:
            logger.warning("nautilus_trader not installed, skipping catalog write")
            raise

    @staticmethod
    def symbol_to_nautilus_id(symbol: str, exchange: str = "BINANCE") -> str:
        """将 ccxt symbol 转换为 Nautilus instrument ID.

        Args:
            symbol: ccxt 格式 (如 'BTC/USDT').
            exchange: 交易所名称.

        Returns:
            Nautilus ID (如 'BTCUSDT-PERP.BINANCE').
        """
        # 移除 / 和 :, 标准化
        base = symbol.replace("/", "").replace(":USDT", "").replace(":USD", "")
        # 如果是合约 (含 :), 加 -PERP
        if ":" in symbol:
            return f"{base}-PERP.{exchange.upper()}"
        return f"{base}.{exchange.upper()}"

    @staticmethod
    def timeframe_to_seconds(timeframe: str) -> int:
        """将 K 线 timeframe 转换为秒数.

        Args:
            timeframe: K 线时间框架 (如 '1m', '5m', '1h', '4h', '1d').

        Returns:
            秒数.
        """
        multipliers = {"m": 60, "h": 3600, "d": 86400, "w": 604800}
        unit = timeframe[-1]
        value = int(timeframe[:-1])
        return value * multipliers.get(unit, 60)

    @staticmethod
    def estimate_candle_count(period_str: str, timeframe: str) -> int:
        """根据回测周期和 K 线 timeframe 估算需要的蜡烛数量.

        Args:
            period_str: 回测周期 (如 '6m', '1y', '30d').
            timeframe: K 线 timeframe (如 '4h').

        Returns:
            估算的蜡烛数量.
        """
        # 解析周期
        unit = period_str[-1]
        value = int(period_str[:-1])
        period_days = {
            "d": value,
            "w": value * 7,
            "m": value * 30,
            "y": value * 365,
        }.get(unit, value)

        tf_seconds = OHLCVLoader.timeframe_to_seconds(timeframe)
        return int(period_days * 86400 / tf_seconds)
