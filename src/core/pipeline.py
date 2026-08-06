"""End-to-end analysis pipeline: data -> preprocess -> train -> predict.

Everything in this module is pure Python (no Streamlit), so the exact same
code runs in the app, in tests, and on the CI runner.
"""

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from ..constants import (
    DENSE_UNITS,
    EARLY_STOPPING_PATIENCE,
    LSTM_DROPOUT,
    LSTM_UNITS,
    MOVING_AVERAGE_WINDOW,
    TRAIN_TEST_SPLIT_RATIO,
    VALIDATION_SPLIT,
)
from . import baselines
from .callbacks import ProgressReporterCallback
from .data_loader import download_stock_data
from .lstm_model import (
    create_lstm_model,
    make_future_predictions,
    predict_on_test,
    train_model,
)
from .metrics import regression_metrics
from .preprocessing import create_sequences, scale_data


@dataclass
class AnalysisResult:
    """Everything produced by one run of the analysis pipeline."""

    symbol: str
    params: dict[str, Any]
    data: pd.DataFrame
    close: pd.Series
    scaler: MinMaxScaler
    model: Any
    history: Any
    test_predictions: pd.DataFrame
    future: pd.DataFrame
    lstm_metrics: dict[str, float]
    baselines: dict[str, dict[str, float]]
    baselines_frame: pd.DataFrame
    test_start_index: int

    @property
    def forecast_with_change(self) -> pd.DataFrame:
        """Forecast table with day-over-day percentage change, for display."""
        change = self.future["predicted_close"].pct_change() * 100
        return self.future.assign(change_pct=change)


class PipelineError(RuntimeError):
    """Raised when a step of the pipeline fails (message is user-facing)."""


def run_analysis(
    symbol: str,
    start_date: Any,
    end_date: Any,
    sequence_length: int,
    future_steps: int,
    epochs: int,
    batch_size: int,
    progress_callback: ProgressReporterCallback | None = None,
) -> AnalysisResult:
    """Run the full pipeline for the given parameters and return the result."""
    data = download_stock_data(symbol, start_date, end_date)
    if data is None or data.empty:
        raise PipelineError(
            f"No data found for {symbol} between {start_date} and {end_date}. "
            "Check the ticker and date range."
        )

    close = cast(pd.Series, data["Close"])
    scaled, scaler = scale_data(close.to_numpy().reshape(-1, 1))
    sequences = create_sequences(scaled, sequence_length)
    if sequences is None:
        raise PipelineError(
            f"Not enough data ({len(scaled)} points) for a lookback of "
            f"{sequence_length} days. Use an earlier start date or shorter lookback."
        )
    x, y = sequences

    split = int(len(x) * TRAIN_TEST_SPLIT_RATIO)
    if split == 0 or split == len(x):
        raise PipelineError("Train/test split produced an empty set. Adjust the range.")
    x_train, x_test = x[:split], x[split:]
    y_train, y_test = y[:split], y[split:]
    test_start_index = split + sequence_length

    model = create_lstm_model(
        input_shape=(sequence_length, 1),
        units=LSTM_UNITS,
        dropout_rate=LSTM_DROPOUT,
        dense_units=DENSE_UNITS,
    )
    history = train_model(
        model,
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=VALIDATION_SPLIT,
        patience=EARLY_STOPPING_PATIENCE,
        progress_callback=progress_callback,
    )

    test_predictions = predict_on_test(model, x_test, y_test, scaler)
    lstm_metrics = regression_metrics(
        test_predictions["actual"].to_numpy(),
        test_predictions["predicted"].to_numpy(),
    )

    future = make_future_predictions(
        model,
        scaled,
        scaler,
        sequence_length,
        future_steps,
        last_date=cast(pd.Timestamp, close.index[-1]),
    )

    baseline_frame = baselines.baseline_forecasts(
        close, test_start_index, MOVING_AVERAGE_WINDOW
    )
    baseline_metrics = baselines.evaluate_baselines(
        close, test_start_index, MOVING_AVERAGE_WINDOW
    )

    return AnalysisResult(
        symbol=symbol,
        params={
            "sequence_length": sequence_length,
            "future_steps": future_steps,
            "epochs": epochs,
            "batch_size": batch_size,
        },
        data=data,
        close=close,
        scaler=scaler,
        model=model,
        history=history,
        test_predictions=test_predictions,
        future=future,
        lstm_metrics=lstm_metrics,
        baselines=baseline_metrics,
        baselines_frame=baseline_frame,
        test_start_index=test_start_index,
    )
