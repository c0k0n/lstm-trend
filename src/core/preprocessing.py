"""Scaling and sequence creation for the LSTM."""

from typing import Any, Optional, Tuple

import numpy as np
from sklearn.preprocessing import MinMaxScaler


def scale_data(data: np.ndarray) -> Tuple[np.ndarray, MinMaxScaler]:
    """Scale a (n, 1) price array to [0, 1] and return the fitted scaler."""
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(data)
    return scaled, scaler


def inverse_scale_data(scaled_data: np.ndarray, scaler: MinMaxScaler) -> np.ndarray:
    """Map scaled values back to the original price scale."""
    return scaler.inverse_transform(scaled_data)


def create_sequences(
    data: np.ndarray, sequence_length: int
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
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
