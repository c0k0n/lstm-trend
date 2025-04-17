import numpy as np
import pandas as pd
from tensorflow.keras.models import  Model # Import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input # Import Input
from tensorflow.keras.callbacks import Callback, EarlyStopping # Added more callbacks
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from typing import Tuple, List, Any, Optional # Added Optional

# Type alias for Keras model and history
KerasModel = Any
History = Any

def create_lstm_model(input_shape: Tuple[int, int], units: int = 50, dropout_rate: float = 0.2) -> KerasModel:
    """
    Creates an LSTM model using the Keras Functional API.

    Args:
        input_shape (Tuple[int, int]): Shape of the input data (sequence_length, num_features).
        units (int): Number of LSTM units in the main layer.
        dropout_rate (float): Dropout rate for regularization.

    Returns:
        KerasModel: A compiled Keras Functional API model.
    """
    # Define input layer explicitly
    inputs = Input(shape=input_shape)

    # Build the model layers functionally
    x = LSTM(units=units, return_sequences=True)(inputs)
    x = Dropout(dropout_rate)(x)
    x = LSTM(units=units // 2, return_sequences=False)(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(units=25, activation='relu')(x) # Added relu activation
    outputs = Dense(units=1)(x) # Output layer

    # Create the model instance
    model = Model(inputs=inputs, outputs=outputs)

    # Compile
    model.compile(optimizer='adam', loss='mean_absolute_error')
    # print(model.summary()) # Optional: print summary
    return model

def train_model(
    model: KerasModel,
    X_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int,
    batch_size: int,
    validation_split: float = 0.1,
    callbacks: Optional[List[Callback]] = None
) -> History:
    """
    Trains the LSTM model.

    Args:
        model (KerasModel): The compiled Keras model.
        X_train (np.ndarray): Training input sequences.
        y_train (np.ndarray): Training target values.
        epochs (int): Number of training epochs.
        batch_size (int): Training batch size.
        validation_split (float): Fraction of training data to use for validation.
        callbacks (Optional[List[Callback]]): List of Keras callbacks to use during training.

    Returns:
        History: Keras History object containing training metrics.
    """
    # Add common useful callbacks if not provided
    if callbacks is None:
        callbacks = []

    # Early stopping to prevent overfitting
    if not any(isinstance(cb, EarlyStopping) for cb in callbacks):
         callbacks.append(EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True))

    # Optional: Checkpoint to save the best model
    # if not any(isinstance(cb, ModelCheckpoint) for cb in callbacks):
    #     callbacks.append(ModelCheckpoint('best_lstm_model.keras', save_best_only=True, monitor='val_loss'))

    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=callbacks,
        verbose=0 # Set verbose=0 to rely on custom Streamlit callback for progress
    )
    return history

def evaluate_model(
    model: KerasModel,
    X_test: np.ndarray,
    y_test: np.ndarray, # Note: y_test is expected to be SCALED here
    scaler: MinMaxScaler # Scaler is needed to inverse transform
) -> Tuple[float, float]:
    """
    Evaluates the trained LSTM model on the test set using original scale values.

    Args:
        model (KerasModel): The trained Keras model.
        X_test (np.ndarray): Test input sequences.
        y_test (np.ndarray): True test target values (scaled).
        scaler (MinMaxScaler): The scaler used for preprocessing 'Close' prices.

    Returns:
        Tuple[float, float]: Mean Squared Error (MSE) and R-squared (R2) score,
                             calculated on the original price scale.
    """
    # 1. Predict on the test set (predictions are scaled)
    predictions_scaled = model.predict(X_test)

    # 2. Inverse transform predictions and actual test values to original scale
    predictions_original = scaler.inverse_transform(predictions_scaled)
    # Reshape y_test if it's flat (e.g., (n,)) to (n, 1) for inverse_transform
    if y_test.ndim == 1:
        y_test = y_test.reshape(-1, 1)
    y_test_original = scaler.inverse_transform(y_test)

    # 3. Calculate metrics on the original scale
    mse = mean_squared_error(y_test_original, predictions_original)
    r2 = r2_score(y_test_original, predictions_original)

    return mse, r2

def make_future_predictions(
    model: KerasModel,
    last_sequence: np.ndarray, # Should be shape (1, sequence_length, 1)
    future_steps: int
) -> np.ndarray:
    """
    Predicts future values step-by-step using the last known sequence.

    Args:
        model (KerasModel): The trained Keras model.
        last_sequence (np.ndarray): The last available sequence from the data,
                                    already scaled and correctly shaped (1, seq_len, 1).
        future_steps (int): The number of future time steps to predict.

    Returns:
        np.ndarray: An array of predicted values for the future steps (scaled).
                    Shape will be (future_steps, 1).
    """
    future_predictions_scaled = []
    current_sequence = last_sequence.copy() # Use a copy to avoid modifying the original

    for _ in range(future_steps):
        # Predict the next step
        next_pred_scaled = model.predict(current_sequence)[0, 0] # Get scalar prediction

        # Append the prediction
        future_predictions_scaled.append(next_pred_scaled)

        # Update the sequence: remove the first element, append the prediction
        # Reshape prediction to (1, 1) before appending
        next_pred_reshaped = np.array([[next_pred_scaled]]) # Shape (1, 1)
        # Append requires compatible shape, need (1, 1, 1) to match feature dim
        next_pred_for_seq = next_pred_reshaped.reshape(1, 1, 1)

        current_sequence = np.append(current_sequence[:, 1:, :], next_pred_for_seq, axis=1)

    return np.array(future_predictions_scaled).reshape(-1, 1) # Return as (future_steps, 1)