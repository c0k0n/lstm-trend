"""Tests for the feature matrix.

The property that matters is that nothing looks forward. Every other bug in
this module is visible on a chart; leakage is not, and it shows up as skill.
"""

import numpy as np
import pandas as pd
import pytest

from src.core.features import (
    FEATURE_COLUMNS,
    OHLCV_COLUMNS,
    build_features,
    forward_log_return,
)


def _prices(n: int = 300, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    volume = rng.integers(1_000_000, 5_000_000, n).astype(float)
    index = pd.bdate_range("2022-01-03", periods=n)
    return pd.DataFrame({"Close": close, "Volume": volume}, index=index)


def test_forward_log_return_is_the_log_of_the_future_over_today():
    close = pd.Series([100.0, 110.0, 121.0])
    result = forward_log_return(close, horizon=1)
    assert result.iloc[0] == pytest.approx(np.log(110.0 / 100.0))
    assert result.iloc[1] == pytest.approx(np.log(121.0 / 110.0))


def test_forward_log_return_is_nan_where_the_future_is_unknown():
    result = forward_log_return(pd.Series([100.0, 110.0, 121.0]), horizon=2)
    assert result.iloc[0] == pytest.approx(np.log(121.0 / 100.0))
    assert np.isnan(result.iloc[1])
    assert np.isnan(result.iloc[2])


def test_build_features_has_the_declared_columns():
    feats = build_features(_prices())
    assert tuple(feats.columns) == FEATURE_COLUMNS
    assert len(feats) == 300


def test_build_features_works_without_volume():
    feats = build_features(_prices().drop(columns="Volume"))
    assert feats["vol_ratio"].isna().all()


def test_no_feature_looks_forward():
    """Changing the future must not change any earlier row's features."""
    data = _prices(n=300)
    cutoff = 200

    before = build_features(data)

    mutated = data.copy()
    mutated.iloc[cutoff:, mutated.columns.get_loc("Close")] *= 3.0
    mutated.iloc[cutoff:, mutated.columns.get_loc("Volume")] *= 7.0
    after = build_features(mutated)

    pd.testing.assert_frame_equal(before.iloc[:cutoff], after.iloc[:cutoff])


def test_calendar_columns_are_populated():
    feats = build_features(_prices(n=120))
    assert feats["dow"].between(0, 6).all()
    assert feats["month"].between(1, 12).all()


def test_warmup_rows_are_nan_and_the_rest_are_not():
    feats = build_features(_prices(n=300))
    assert feats["vol_60"].iloc[:59].isna().all()
    assert feats["vol_60"].iloc[100:].notna().all()


def _candles(n: int = 300, seed: int = 3) -> pd.DataFrame:
    """A full OHLCV frame, the way yfinance actually delivers it."""
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    # A candle contains its close, and the open sits near the previous close.
    intraday = np.abs(rng.normal(0, 0.008, n))
    high = close * (1 + intraday)
    low = close * (1 - intraday)
    open_ = close * (1 + rng.normal(0, 0.003, n))
    volume = rng.integers(1_000_000, 5_000_000, n).astype(float)
    index = pd.bdate_range("2022-01-03", periods=n)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=index,
    )


def test_ohlcv_columns_appear_when_the_frame_has_a_full_candle():
    feats = build_features(_candles())
    for column in OHLCV_COLUMNS:
        assert column in feats.columns, column
    assert feats[list(OHLCV_COLUMNS)].iloc[100:].notna().all().all()


def test_ohlcv_is_omitted_rather_than_filled_with_nan():
    """A NaN column silently empties every training row downstream."""
    feats = build_features(_prices())
    assert not any(column in feats.columns for column in OHLCV_COLUMNS)
    assert tuple(feats.columns) == FEATURE_COLUMNS


def test_no_ohlcv_feature_looks_forward():
    data = _candles(n=300)
    cutoff = 200
    before = build_features(data)

    mutated = data.copy()
    mutated.iloc[cutoff:] *= 1.5
    after = build_features(mutated)

    pd.testing.assert_frame_equal(before.iloc[:cutoff], after.iloc[:cutoff])


def test_parkinson_volatility_rises_with_a_wider_range():
    calm = _candles(n=200, seed=3)
    wild = calm.copy()
    width = (wild["High"] - wild["Low"]) * 4.0
    mid = (wild["High"] + wild["Low"]) / 2.0
    wild["High"] = mid + width / 2.0
    wild["Low"] = mid - width / 2.0
    wild["Close"] = np.minimum(np.maximum(wild["Close"], wild["Low"]), wild["High"])

    calm_vol = build_features(calm)["parkinson_20"].iloc[-1]
    wild_vol = build_features(wild)["parkinson_20"].iloc[-1]
    assert wild_vol > calm_vol


def test_ohlcv_can_be_switched_off_for_a_recursive_forecast():
    """Recursive forecasting needs day H's candle, which does not exist yet."""
    feats = build_features(_candles(), ohlcv=False)
    assert tuple(feats.columns) == FEATURE_COLUMNS
