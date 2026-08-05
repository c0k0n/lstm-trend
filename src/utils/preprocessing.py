import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Optional, Any  # Added Any for scaler type


def scale_data(data: np.ndarray) -> Tuple[np.ndarray, Any]:
    """
    Scales the input data using MinMaxScaler between 0 and 1.

    Args:
        data (np.ndarray): A numpy array (usually N x 1) of data to scale.

    Returns:
        Tuple[np.ndarray, Any]: A tuple containing the scaled data and the fitted scaler object.
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler


def inverse_scale_data(scaled_data: np.ndarray, scaler: Any) -> np.ndarray:
    """
    Applies inverse transformation to scaled data using the provided scaler.

    Args:
        scaled_data (np.ndarray): The data that was previously scaled (e.g., predictions).
        scaler (Any): The fitted scaler object used for the original scaling.

    Returns:
        np.ndarray: The data transformed back to its original scale.
    """
    original_data = scaler.inverse_transform(scaled_data)
    return original_data


def create_sequences(
    data: np.ndarray, sequence_length: int
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Creates input sequences (X) and corresponding target values (y) for time series forecasting.

    Args:
        data (np.ndarray): The scaled time series data (usually N x 1).
        sequence_length (int): The number of time steps in each input sequence (lookback window).

    Returns:
        Optional[Tuple[np.ndarray, np.ndarray]]: A tuple containing X (sequences)
                                                 and y (targets), or None if not
                                                 enough data is available.
                                                 X shape: (num_sequences, sequence_length, 1)
                                                 y shape: (num_sequences, 1)
    """
    X = []
    y = []
    if len(data) <= sequence_length:
        # Not enough data points to create even one sequence
        return None  # Indicate failure clearly

    for i in range(sequence_length, len(data)):
        X.append(data[i - sequence_length : i, 0])  # Input sequence
        y.append(data[i, 0])  # Target value (next step)

    if not X:
        # Should not happen if len(data) > sequence_length, but as a safeguard
        return None

    # Convert lists to numpy arrays
    X = np.array(X)
    y = np.array(y)

    # Reshape X to be [samples, time steps, features] as expected by LSTM
    # In this case, features = 1 (just the 'Close' price)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    # Reshape y to be [samples, 1] (optional but good practice)
    y = np.reshape(y, (y.shape[0], 1))

    return X, y
