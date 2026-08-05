"""Simple benchmark forecasts used to put the LSTM results into perspective."""

from typing import Dict

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
    values = []
    for i in range(test_start_index, len(close)):
        values.append(close.iloc[max(0, i - window) : i].mean())
    return pd.Series(values, index=test_index)


def evaluate_baselines(
    close: pd.Series, test_start_index: int, window: int
) -> Dict[str, Dict[str, float]]:
    """Evaluate naive and moving-average forecasts on the test window."""
    test_index = close.index[test_start_index:]
    actual = close.iloc[test_start_index:].values

    naive = naive_forecast(close, test_start_index)
    moving = moving_average_forecast(close, test_start_index, window)

    return {
        NAIVE: regression_metrics(actual, naive.values),
        MOVING_AVERAGE: regression_metrics(actual, moving.values),
    }
