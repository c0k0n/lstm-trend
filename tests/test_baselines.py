"""Unit tests for the naive and moving-average baselines."""

import numpy as np
import pandas as pd

from src.core.baselines import (
    MOVING_AVERAGE,
    NAIVE,
    baseline_forecasts,
    evaluate_baselines,
    moving_average_forecast,
    naive_forecast,
)

CLOSE = pd.Series(
    [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
    index=pd.bdate_range("2026-01-01", periods=7),
)


def test_naive_forecast_is_shifted():
    forecast = naive_forecast(CLOSE, test_start_index=4)
    expected = [13.0, 14.0, 15.0]
    np.testing.assert_array_equal(forecast.values, expected)
    assert list(forecast.index) == list(CLOSE.index[4:])


def test_moving_average_forecast():
    forecast = moving_average_forecast(CLOSE, test_start_index=4, window=3)
    # index 4 -> mean(11, 12, 13); index 5 -> mean(12, 13, 14); index 6 -> mean(13, 14, 15)
    np.testing.assert_allclose(forecast.values, [12.0, 13.0, 14.0])


def test_evaluate_baselines_returns_metrics():
    metrics = evaluate_baselines(CLOSE, test_start_index=4, window=3)
    assert set(metrics.keys()) == {NAIVE, MOVING_AVERAGE}
    assert set(metrics[NAIVE].keys()) == {"mse", "rmse", "mae", "mape", "r2"}


def test_baseline_forecasts_combines_columns():
    frame = baseline_forecasts(CLOSE, test_start_index=4, window=3)
    assert list(frame.columns) == [NAIVE, MOVING_AVERAGE]
    assert list(frame.index) == list(CLOSE.index[4:])
    np.testing.assert_array_equal(frame[NAIVE].to_numpy(), [13.0, 14.0, 15.0])
    np.testing.assert_allclose(frame[MOVING_AVERAGE].to_numpy(), [12.0, 13.0, 14.0])
