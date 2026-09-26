"""LSTM model definition, training, evaluation and step-wise forecasting."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from .callbacks import ProgressReporterCallback
from .preprocessing import calendar_frame

if TYPE_CHECKING:  # pragma: no cover
    from keras import models

# PyTorch's cuDNN LSTM path warns that Keras-built weights are not one
# contiguous chunk of memory. We cannot call flatten_parameters() because
# Keras invokes the functional torch._VF.lstm API, so the hint (which fires
# on every GPU forward pass) is only noise — suppress it.
warnings.filterwarnings(
    "ignore",
    message=r"RNN module weights are not part of single contiguous chunk of memory.*",
    category=UserWarning,
)


def build_lstm_stack(
    input_shape: tuple[int, int],
    units: int,
    dropout_rate: float,
    dense_units: int,
    output_units: int = 1,
) -> Any:
    """The shared LSTM trunk: two recurrent layers, norm, dropout, dense head.

    One definition for both forecasters, because they are the same network
    with a different number of output heads. They were written out separately
    and had already drifted in unit count and dropout placement; a head-count
    difference does not justify two architectures that must stay in step.

    `output_units` is the only thing that differs — 1 for the point forecast,
    one per quantile level for the distribution forecast.
    """
    _, layers, models, _ = _keras()

    inputs = layers.Input(shape=input_shape)
    x = layers.LSTM(units=units, return_sequences=True)(inputs)
    x = layers.LayerNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.LSTM(units=units // 2, return_sequences=False)(x)
    x = layers.LayerNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(dense_units, activation="relu")(x)
    return models.Model(inputs=inputs, outputs=layers.Dense(output_units)(x))


def _keras() -> tuple[Any, Any, Any, Any]:
    """Import Keras lazily and bind the backend, as `backtest.py` does.

    Importing torch eagerly would make every module that touches this one pay a
    multi-second import for a model it may never train.
    """
    import os

    os.environ.setdefault("KERAS_BACKEND", "torch")

    import keras
    from keras import layers, models
    from keras.losses import Loss

    return keras, layers, models, Loss


def early_stopping(patience: int = 5) -> Any:
    """The project's one early-stopping configuration.

    `monitor="val_loss"` with `restore_best_weights=True`, everywhere. Three
    separate copies of these five lines had drifted, and a run that restores
    the *last* weights instead of the best silently reports a worse model than
    it trained.
    """
    from keras import callbacks

    return callbacks.EarlyStopping(
        monitor="val_loss", patience=patience, restore_best_weights=True
    )


def create_lstm_model(
    input_shape: tuple[int, int],
    units: int,
    dropout_rate: float,
    dense_units: int,
) -> "models.Model":
    """The point forecaster: one output, mean-absolute-error loss."""
    model = build_lstm_stack(input_shape, units, dropout_rate, dense_units, 1)
    model.compile(optimizer="adam", loss="mean_absolute_error")
    return model


def train_model(
    model: "models.Model",
    x_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int,
    batch_size: int,
    validation_split: float,
    patience: int,
    progress_callback: ProgressReporterCallback | None = None,
) -> Any:
    """Train with early stopping and an optional progress callback.

    `verbose="0"` on every training call in this project, deliberately: the
    epoch lines are ~50 lines of console per run and the progress bar in the
    UI is the intended channel.
    """
    callbacks: list[Any] = []
    if progress_callback is not None:
        callbacks.append(progress_callback)
    callbacks.append(early_stopping(patience))
    return model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=callbacks,
        verbose="0",
    )


def _scaled_to_return(value: float, scaler: MinMaxScaler) -> float:
    """Undo the scaling on one predicted log-return."""
    return float(scaler.inverse_transform(np.array([[value]]))[0, 0])


def predict_test_prices(
    model: models.Model,
    x_test: np.ndarray,
    origin_prices: np.ndarray,
    scaler: MinMaxScaler,
) -> np.ndarray:
    """Predicted prices for the test windows.

    The model predicts log-returns, so each prediction is converted back to a
    price using the real close on the day the window ends. That keeps the
    comparison honest: the model never gets to see the day it is predicting.
    """
    scaled = model.predict(x_test, verbose="0").flatten()
    returns = np.array([_scaled_to_return(value, scaler) for value in scaled])
    return origin_prices * np.exp(returns)


def forecast_path(
    model: models.Model,
    matrix: np.ndarray,
    sequence_length: int,
    future_steps: int,
    last_price: float,
    last_date: pd.Timestamp,
    scaler: MinMaxScaler,
) -> pd.DataFrame:
    """Forecast the next `future_steps` business days as a price path.

    The recursion happens in log-return space, which is the standard way to do
    it: predicting returns rather than levels means each step is a plausible
    increment instead of an attempt to reproduce a whole price.

    Calendar columns are known in advance, so they simply continue into the
    future. Features that depend on volume are deliberately absent from the
    input for exactly that reason — tomorrow's volume does not exist yet.

    Errors still accumulate: each predicted return feeds the next window.
    """
    dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=future_steps)
    calendar = calendar_frame(dates)

    current = matrix[-sequence_length:].copy()
    price = float(last_price)
    prices: list[float] = []

    for step in range(future_steps):
        scaled = model.predict(
            current.reshape(1, sequence_length, matrix.shape[1]), verbose="0"
        )[0, 0]
        price *= np.exp(_scaled_to_return(scaled, scaler))
        prices.append(price)
        current = np.vstack([current[1:], np.concatenate([[scaled], calendar[step]])])

    return pd.DataFrame({"date": dates, "predicted_close": prices})
