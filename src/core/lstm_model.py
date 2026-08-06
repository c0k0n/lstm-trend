"""LSTM model definition, training, evaluation and step-wise forecasting."""

import warnings

import numpy as np
import pandas as pd
from keras import callbacks as keras_callbacks
from keras import layers, models
from sklearn.preprocessing import MinMaxScaler

from .callbacks import ProgressReporterCallback

# PyTorch's cuDNN LSTM path warns that Keras-built weights are not one
# contiguous chunk of memory. We cannot call flatten_parameters() because
# Keras invokes the functional torch._VF.lstm API, so the hint (which fires
# on every GPU forward pass) is only noise — suppress it.
warnings.filterwarnings(
    "ignore",
    message=r"RNN module weights are not part of single contiguous chunk of memory.*",
    category=UserWarning,
)


def create_lstm_model(
    input_shape: tuple[int, int],
    units: int,
    dropout_rate: float,
    dense_units: int,
) -> models.Model:
    """Build a two-layer LSTM network with the Keras Functional API."""
    inputs = layers.Input(shape=input_shape)
    x = layers.LSTM(units=units, return_sequences=True)(inputs)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.LSTM(units=units // 2, return_sequences=False)(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(dense_units, activation="relu")(x)
    outputs = layers.Dense(1)(x)

    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss="mean_absolute_error")
    return model


def train_model(
    model: models.Model,
    x_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int,
    batch_size: int,
    validation_split: float,
    patience: int,
    progress_callback: ProgressReporterCallback | None = None,
) -> keras_callbacks.History:
    """Train the model with early stopping and an optional progress callback."""
    callbacks: list[keras_callbacks.Callback] = []
    if progress_callback is not None:
        callbacks.append(progress_callback)
    callbacks.append(
        keras_callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True
        )
    )

    return model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=callbacks,
        verbose="0",
    )


def predict_on_test(
    model: models.Model,
    x_test: np.ndarray,
    y_test: np.ndarray,
    scaler: MinMaxScaler,
) -> pd.DataFrame:
    """Return a DataFrame of actual vs predicted test prices (original scale)."""
    predictions = scaler.inverse_transform(model.predict(x_test, verbose="0"))
    actual = scaler.inverse_transform(y_test.reshape(-1, 1))
    return pd.DataFrame(
        {"actual": actual.flatten(), "predicted": predictions.flatten()}
    )


def make_future_predictions(
    model: models.Model,
    scaled_close: np.ndarray,
    scaler: MinMaxScaler,
    sequence_length: int,
    future_steps: int,
    last_date: pd.Timestamp,
) -> pd.DataFrame:
    """Forecast the next `future_steps` business days, one step at a time.

    Each prediction is appended to the window and the oldest value drops off,
    so small errors can accumulate as the horizon grows.
    """
    current = scaled_close[-sequence_length:].reshape(1, sequence_length, 1).copy()
    predicted = []

    for _ in range(future_steps):
        next_step = model.predict(current, verbose="0")[0, 0]
        predicted.append(next_step)
        current = np.append(current[:, 1:, :], [[[next_step]]], axis=1)

    dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=future_steps)
    values = scaler.inverse_transform(np.array(predicted).reshape(-1, 1)).flatten()
    return pd.DataFrame({"date": dates, "predicted_close": values})
