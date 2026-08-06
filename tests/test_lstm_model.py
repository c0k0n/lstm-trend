"""Unit tests for the LSTM model helpers (tiny models, quick runs)."""

from typing import cast

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.core import lstm_model
from src.core.preprocessing import create_sequences


def _fit_small_model(scaled: np.ndarray, seq: int = 5):
    """Train a tiny LSTM just enough to give sensible predictions."""
    sequences = create_sequences(scaled, seq)
    assert sequences is not None
    x, y = sequences
    model = lstm_model.create_lstm_model(
        (seq, 1), units=8, dropout_rate=0.1, dense_units=4
    )
    lstm_model.train_model(
        model, x, y, epochs=2, batch_size=16, validation_split=0.1, patience=2
    )
    return model


def test_create_lstm_model_predicts_shape():
    model = lstm_model.create_lstm_model(
        input_shape=(10, 1), units=8, dropout_rate=0.2, dense_units=4
    )
    out = model.predict(np.zeros((2, 10, 1)), verbose="0")
    assert out.shape == (2, 1)


def test_train_model_returns_history():
    rng = np.random.default_rng(0)
    y = rng.uniform(0, 1, (20, 1)).astype("float32")
    x = np.stack([y[i : i + 3, 0] for i in range(17)]).reshape(17, 3, 1)
    model = lstm_model.create_lstm_model(
        (3, 1), units=8, dropout_rate=0.1, dense_units=4
    )
    history = lstm_model.train_model(
        model, x, y[3:], epochs=2, batch_size=8, validation_split=0.1, patience=2
    )
    assert "loss" in history.history
    assert "val_loss" in history.history


def test_predict_on_test_inverts_scaling():
    rng = np.random.default_rng(3)
    y = rng.uniform(50, 150, (20, 1))
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(y)
    model = _fit_small_model(scaled)
    sequences = create_sequences(scaled, 5)
    assert sequences is not None
    x, y_seq = sequences

    frame = lstm_model.predict_on_test(model, x, y_seq, scaler)
    assert list(frame.columns) == ["actual", "predicted"]
    assert len(frame) == len(y_seq)
    np.testing.assert_allclose(frame["actual"].to_numpy(), y[5:].flatten())
    assert np.isfinite(frame["predicted"]).all()


def test_make_future_predictions_dates_and_scale():
    rng = np.random.default_rng(1)
    n = 30
    y = np.cumsum(rng.normal(0, 1, n)) + 100
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(y.reshape(-1, 1))
    model = _fit_small_model(scaled)

    future = lstm_model.make_future_predictions(
        model,
        scaled,
        scaler,
        sequence_length=5,
        future_steps=4,
        last_date=cast(pd.Timestamp, pd.Timestamp("2025-01-31")),
    )
    assert len(future) == 4
    assert list(future.columns) == ["date", "predicted_close"]
    # Business-day dates starting the day after the last observed date
    assert future["date"].iloc[0] == pd.Timestamp("2025-02-03")
    assert future["predicted_close"].between(y.min() - 30, y.max() + 30).all()
