"""Scaling and sequence creation for the LSTM."""

import numpy as np
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


def create_sequences(
    data: np.ndarray, sequence_length: int
) -> tuple[np.ndarray, np.ndarray] | None:
    """Build (X, y) sliding windows from a (n, 1) array.

    X has shape (samples, sequence_length, 1); y has shape (samples, 1).
    Returns None when there is not enough data for a single sequence.
    """
    if len(data) <= sequence_length:
        return None

    windows = np.lib.stride_tricks.sliding_window_view(data, (sequence_length, 1))
    x = windows[:-1, 0]
    y = data[sequence_length:, 0].reshape(-1, 1)
    return x, y
