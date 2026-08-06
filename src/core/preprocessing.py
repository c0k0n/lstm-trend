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

    x = np.stack(
        [data[i - sequence_length : i, 0] for i in range(sequence_length, len(data))]
    )
    y = np.array([data[i, 0] for i in range(sequence_length, len(data))])
    return x.reshape(x.shape[0], sequence_length, 1), y.reshape(-1, 1)
