"""Scaling and sequence creation for the LSTM."""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def fit_scaler(train_values: np.ndarray) -> MinMaxScaler:
    """Fit a [0, 1] MinMaxScaler on the training window only.

    Fitting on the whole series would let the test window's minimum and maximum
    shape the values the model trains on — information it could not have had at
    the time. Only the fit is held back; ``transform`` still runs over the full
    series so the sequences stay on one continuous scale.
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_values)
    return scaler


def log_returns(close: pd.Series, periods: int = 1) -> pd.Series:
    """Log return over `periods` trading days — the stationary counterpart of a
    price series.

    Prices carry a unit root, which this project's own Dickey-Fuller test
    reports. Modelling the level asks the network to reproduce a moving target;
    modelling the return asks it to predict the increment, which is the thing
    that is actually forecastable.

    The single definition for the whole codebase: the LSTM pipeline, the
    tabular features and the market-context covariates all go through here, so
    a 5-day return means the same thing to every model.
    """
    return np.log(close / close.shift(periods))


def calendar_frame(index: pd.Index) -> np.ndarray:
    """Day-of-week and month, cyclically encoded.

    These are used because they are the one class of covariate that is known in
    advance: when forecasting day H, its weekday and month are already facts.
    Anything derived from volume or from prices after the forecast date is not,
    and including it would leak.
    """
    stamps = pd.Series(pd.DatetimeIndex(index))
    dow = stamps.dt.dayofweek.to_numpy()
    month = stamps.dt.month.to_numpy()
    return np.column_stack(
        [
            np.sin(2 * np.pi * dow / 5),
            np.cos(2 * np.pi * dow / 5),
            np.sin(2 * np.pi * month / 12),
            np.cos(2 * np.pi * month / 12),
        ]
    )


def build_return_windows(
    matrix: np.ndarray, sequence_length: int
) -> tuple[np.ndarray, np.ndarray] | None:
    """Windows over a feature matrix whose first column is the scaled return.

    X has shape (samples, sequence_length, features) and y is the scaled return
    on the day after each window ends.
    """
    if len(matrix) <= sequence_length:
        return None

    x = np.stack(
        [matrix[i - sequence_length : i] for i in range(sequence_length, len(matrix))]
    )
    y = matrix[sequence_length:, 0].reshape(-1, 1)
    return x, y
