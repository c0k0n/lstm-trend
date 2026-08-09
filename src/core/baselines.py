"""Simple benchmark forecasts used to put the LSTM results into perspective."""

from typing import cast

import numpy as np
import pandas as pd

from .metrics import regression_metrics

NAIVE = "Naive (yesterday)"
MOVING_AVERAGE = "Moving average"


def naive_forecast(close: pd.Series, test_start_index: int) -> pd.Series:
    """Predict each test day's price as the previous day's close."""
    test_index = close.index[test_start_index:]
    return pd.Series(
        close.iloc[test_start_index - 1 : len(close) - 1].values,
        index=test_index,
    )


def moving_average_forecast(
    close: pd.Series, test_start_index: int, window: int
) -> pd.Series:
    """Predict each test day's price as the mean of the previous `window` closes."""
    test_index = close.index[test_start_index:]
    values = cast(pd.Series, close.shift(1).rolling(window, min_periods=1).mean())
    return values.iloc[test_start_index:].set_axis(test_index)


def baseline_forecasts(
    close: pd.Series, test_start_index: int, window: int
) -> pd.DataFrame:
    """Test-window forecasts from both baselines, indexed by test date."""
    return pd.DataFrame(
        {
            NAIVE: naive_forecast(close, test_start_index).values,
            MOVING_AVERAGE: moving_average_forecast(
                close, test_start_index, window
            ).values,
        },
        index=close.index.to_numpy()[test_start_index:],
    )


def evaluate_baselines(
    actual: np.ndarray, frame: pd.DataFrame
) -> dict[str, dict[str, float]]:
    """Evaluate every forecast column of a baseline frame against the test window."""
    return {
        name: regression_metrics(actual, frame[name].to_numpy())
        for name in frame.columns
    }
