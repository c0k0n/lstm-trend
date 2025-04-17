import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple

# Change the default target column to 'Close'
def scale_data(data: pd.DataFrame, target_col: str = 'Close') -> Tuple[np.ndarray, MinMaxScaler]:
    """Scales the target column using MinMaxScaler."""
    scaler = MinMaxScaler(feature_range=(0, 1))
    # Ensure data[target_col] is treated as a DataFrame for scaling
    scaled_data = scaler.fit_transform(data[[target_col]])
    return scaled_data, scaler

def create_sequences(data: np.ndarray, seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """Creates sequences and labels for LSTM model training."""
    sequences, labels = [], []
    for i in range(len(data) - seq_length):
        seq = data[i:i + seq_length]
        label = data[i + seq_length]
        sequences.append(seq)
        labels.append(label)
    X = np.array(sequences)
    y = np.array(labels)
    return X, y