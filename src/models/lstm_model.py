import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.metrics import MeanSquaredError, MeanAbsoluteError
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from typing import Tuple
import logging

# Assuming CustomProgressBarCallback is in the same directory or properly imported
from .callbacks import CustomProgressBarCallback

def create_lstm_model(input_shape: Tuple[int, int]) -> Sequential:
    """Creates the LSTM model architecture."""
    model = Sequential([
        LSTM(units=128, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(units=128, return_sequences=True),
        Dropout(0.2),
        LSTM(units=128),
        Dropout(0.2),
        Dense(units=1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=[MeanAbsoluteError(), MeanSquaredError()])
    logging.info("LSTM Model compiled successfully.")
    # model.summary(print_fn=logging.info) # Log model summary if needed
    return model


def train_model(model: Sequential, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, epochs: int, batch_size: int) -> tf.keras.callbacks.History:
    """Trains the LSTM model and displays progress in Streamlit."""
    st.subheader("LSTM Model Training Progress")
    progress_bar = st.progress(0)
    custom_callback = CustomProgressBarCallback(progress_bar, epochs)

    with st.spinner("Training model... This may take a while."):
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            callbacks=[custom_callback],
            verbose=0 # Set verbose to 0 to avoid duplicate output
        )
    st.success("Model training finished!")
    return history


def evaluate_model(model: Sequential, X_test: np.ndarray, y_test: np.ndarray, scaler: MinMaxScaler) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Evaluates the model on the test set and returns metrics and predictions."""
    logging.info("Evaluating model...")
    predictions_scaled = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions_scaled)
    # y_test needs to be 2D for inverse_transform if it's not already
    y_test_reshaped = y_test.reshape(-1, 1)
    y_test_original = scaler.inverse_transform(y_test_reshaped)

    mse = mean_squared_error(y_test_original, predictions)
    r2 = r2_score(y_test_original, predictions)

    logging.info(f'Test Mean Squared Error: {mse}')
    logging.info(f'Test R2 Score: {r2}')

    return mse, r2, y_test_original, predictions


def make_future_predictions(model: Sequential, last_sequence_scaled: np.ndarray, scaler: MinMaxScaler, future_steps: int, sequence_length: int) -> np.ndarray:
    """Predicts future stock prices."""
    logging.info(f"Making {future_steps} future predictions...")
    future_predictions_scaled = []
    current_sequence = last_sequence_scaled.copy() # Ensure it's a copy

    for _ in range(future_steps):
        # Reshape current_sequence for prediction: [1, sequence_length, 1]
        current_sequence_reshaped = current_sequence.reshape(1, sequence_length, 1)
        # Predict the next step
        next_prediction_scaled = model.predict(current_sequence_reshaped)[0, 0]
        future_predictions_scaled.append(next_prediction_scaled)
        # Update the sequence: remove the first element, append the prediction
        # Ensure the prediction is treated as a single element array for np.append
        current_sequence = np.append(current_sequence[1:], [next_prediction_scaled])

    # Inverse transform the predictions
    future_predictions = scaler.inverse_transform(np.array(future_predictions_scaled).reshape(-1, 1))
    logging.info("Future predictions generated.")
    return future_predictions