import pandas as pd

from getall.trading.backtest.engine import compute_indicators, evaluate_conditions, run_backtest


def _sample_ohlcv(rows: int = 40) -> pd.DataFrame:
    index = pd.date_range("2025-01-01", periods=rows, freq="D", tz="UTC")
    close = [90.0 if i % 2 == 0 else 110.0 for i in range(rows)]
    data = {
        "open": close,
        "high": [v + 1.0 for v in close],
        "low": [v - 1.0 for v in close],
        "close": close,
        "volume": [1000.0] * rows,
    }
    return pd.DataFrame(data, index=index)


def test_compute_indicators_supports_aliases_and_price_series() -> None:
    df = _sample_ohlcv()
    indicators = compute_indicators(
        df,
        [
            {"name": "Bollinger Bands", "params": {"length": 20, "stddev": 2}},
            {"name": "SMA", "params": {"period": 1}, "key": "FAST MA"},
        ],
    )

    assert "price" in indicators
    pd.testing.assert_series_equal(indicators["price"], df["close"], check_names=False)
    assert "bollinger_lower" in indicators
    assert "fast_ma" in indicators
    pd.testing.assert_series_equal(indicators["fast_ma"], df["close"], check_names=False)


def test_evaluate_conditions_accepts_symbol_operators_and_price_alias() -> None:
    df = _sample_ohlcv()
    indicators = compute_indicators(df, [])
    conditions = [{"indicator": "price", "field": "value", "operator": "<", "value": 95}]

    signal = evaluate_conditions(conditions, indicators, logic="AND")

    assert len(signal) == len(df)
    assert int(signal.sum()) > 0


def test_evaluate_conditions_skips_invalid_operator_without_killing_valid_ones() -> None:
    df = _sample_ohlcv()
    indicators = compute_indicators(df, [])
    conditions = [
        {"indicator": "price", "operator": "totally_invalid", "value": 100},
        {"indicator": "price", "operator": ">", "value": 105},
    ]

    signal = evaluate_conditions(conditions, indicators, logic="and")

    assert len(signal) == len(df)
    assert int(signal.sum()) > 0


def test_evaluate_conditions_supports_crossunder_and_indicator_key_variants() -> None:
    df = _sample_ohlcv()
    indicators = compute_indicators(df, [{"name": "SMA", "params": {"length": 2}, "key": "MA FAST"}])
    conditions = [{"indicator": "price", "operator": "crossunder", "value": "MA FAST.value"}]

    signal = evaluate_conditions(conditions, indicators, logic="and")

    assert len(signal) == len(df)
    assert int(signal.sum()) > 0


def test_run_backtest_handles_noncanonical_strategy_config_and_generates_trades() -> None:
    df = _sample_ohlcv(rows=60)
    config = {
        "name": "robustness-check",
        "symbols": ["TEST/USDT:USDT"],
        "timeframe": "1d",
        "direction": "LONG_ONLY",
        "trade_size_pct": "100",
        "fees": "0.0006",
        "indicators": [{"name": "SMA", "params": {"length": 2}, "key": "MA FAST"}],
        "entry_conditions": [{"indicator": "price", "operator": "<", "value": "MA FAST.value"}],
        "exit_conditions": [{"indicator": "price", "operator": ">", "value": "ma_fast.value"}],
    }

    result = run_backtest({"TEST/USDT:USDT": df}, config, starting_balance=10_000.0)

    assert "error" not in result
    assert result["symbol"] == "TEST/USDT:USDT"
    assert result["total_trades"] > 0
