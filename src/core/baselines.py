"""Simple benchmark forecasts used to put the LSTM results into perspective."""

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
    close: pd.Series, test_start_index: int, window: int
) -> dict[str, dict[str, float]]:
    """Evaluate naive and moving-average forecasts on the test window."""
    actual = close.iloc[test_start_index:].to_numpy()
    frame = baseline_forecasts(close, test_start_index, window)

    return {
        NAIVE: regression_metrics(actual, frame[NAIVE].to_numpy()),
        MOVING_AVERAGE: regression_metrics(actual, frame[MOVING_AVERAGE].to_numpy()),
    }
