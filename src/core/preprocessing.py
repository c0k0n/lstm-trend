"""Scaling and sequence creation for the LSTM."""

import numpy as np
from sklearn.preprocessing import MinMaxScaler


def scale_data(data: np.ndarray) -> tuple[np.ndarray, MinMaxScaler]:
    """Scale a (n, 1) price array to [0, 1] and return the fitted scaler."""
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(data)
    return scaled, scaler


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
