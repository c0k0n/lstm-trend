"""End-to-end analysis pipeline: download, train, evaluate, forecast."""

from __future__ import annotations

import datetime
import logging
from dataclasses import dataclass, field
from typing import Any

import keras
import numpy as np
import pandas as pd

from .baselines import baseline_forecasts, evaluate_baselines
from .callbacks import ProgressReporterCallback
from .data_loader import download_stock_data
from .lstm_model import (
    create_lstm_model,
    forecast_path,
    predict_test_prices,
    train_model,
)
from .metrics import regression_metrics
from .preprocessing import (
    build_return_windows,
    calendar_frame,
    fit_scaler,
    log_returns,
)
from ..constants import (
    DENSE_UNITS,
    EARLY_STOPPING_PATIENCE,
    LSTM_DROPOUT,
    LSTM_UNITS,
    MOVING_AVERAGE_WINDOW,
    RANDOM_SEED,
    TRAIN_TEST_SPLIT_RATIO,
    VALIDATION_SPLIT,
)

logger = logging.getLogger(__name__)


class PipelineError(Exception):
    """Raised when the analysis pipeline cannot complete."""


@dataclass
class AnalysisResult:
    """All artefacts produced by a single analysis run."""

    symbol: str
    data: pd.DataFrame
    close: pd.Series
    test_start_index: int
    test_predictions: pd.DataFrame
    lstm_metrics: dict[str, float]
    baselines: dict[str, dict[str, float]]
    baselines_frame: pd.DataFrame
    history: Any
    future: pd.DataFrame
    forecast_with_change: pd.DataFrame
    params: dict[str, Any] = field(default_factory=dict)


def run_analysis(
    *,
    symbol: str,
    start_date: datetime.date,
    end_date: datetime.date,
    sequence_length: int,
    future_steps: int,
    epochs: int,
    batch_size: int,
    progress_callback: ProgressReporterCallback | None = None,
) -> AnalysisResult:
    """Execute the full analysis pipeline and return all results.

    Raises PipelineError on data or model failures.
    """
    keras.utils.set_random_seed(RANDOM_SEED)

    # --- Data ---
    data = download_stock_data(symbol, start_date, end_date)
    min_rows = sequence_length + 10  # sequence + buffer for train/test split
    if data is None or len(data) < min_rows:
        count = len(data) if data is not None else 0
        raise PipelineError(
            f"Not enough data for **{symbol}** in this date range "
            f"({count} rows, need at least {min_rows}). "
            "Try a longer date range or a different ticker."
        )

    close = data["Close"]
    # Returns, not levels. The first row has no prior close, so the aligned
    # close series starts one day later than the price series.
    returns = log_returns(close).dropna()
    close_aligned = close.iloc[1:]

    # --- Split, then scale on the training window only ---
    return_values = returns.to_numpy().reshape(-1, 1)
    split_idx = int((len(return_values) - sequence_length) * TRAIN_TEST_SPLIT_RATIO)
    # The training sequences reach this far into the return series; everything
    # past it is reserve the scaler must not be allowed to see.
    train_end = split_idx + sequence_length

    scaler = fit_scaler(return_values[:train_end])
    matrix = np.column_stack(
        [scaler.transform(return_values), calendar_frame(returns.index)]
    )

    # --- Sequence ---
    seq_result = build_return_windows(matrix, sequence_length)
    if seq_result is None:
        raise PipelineError(
            f"Not enough data to create sequences for **{symbol}** "
            f"(need more than {sequence_length} rows)."
        )
    x, y = seq_result

    x_train, x_test = x[:split_idx], x[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # Window j predicts the return at position sequence_length + j, so its
    # starting price is the close on the day before that.
    aligned = close_aligned.to_numpy()
    origins = aligned[sequence_length - 1 : sequence_length - 1 + len(y)]
    actuals = aligned[sequence_length : sequence_length + len(y)]
    # +1 because the return series sits one row behind the price series.
    test_start_index = split_idx + sequence_length + 1

    # --- Model ---
    model = create_lstm_model(
        input_shape=(sequence_length, matrix.shape[1]),
        units=LSTM_UNITS,
        dropout_rate=LSTM_DROPOUT,
        dense_units=DENSE_UNITS,
    )

    # --- Train ---
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

    # --- Evaluate on test ---
    predicted = predict_test_prices(model, x_test, origins[split_idx:], scaler)
    test_predictions = pd.DataFrame(
        {"actual": actuals[split_idx:], "predicted": predicted}
    )
    lstm_metrics = regression_metrics(
        test_predictions["actual"].to_numpy(),
        test_predictions["predicted"].to_numpy(),
    )

    # --- Baselines ---
    baselines_frame = baseline_forecasts(close, test_start_index, MOVING_AVERAGE_WINDOW)
    baselines = evaluate_baselines(
        test_predictions["actual"].to_numpy(), baselines_frame
    )

    # --- Future forecast ---
    future = forecast_path(
        model,
        matrix,
        sequence_length,
        future_steps,
        float(close.iloc[-1]),
        close.index[-1],
        scaler,
    )

    # --- Forecast with day-over-day change ---
    forecast_with_change = future.copy()
    forecast_with_change["change_pct"] = forecast_with_change[
        "predicted_close"
    ].pct_change()

    return AnalysisResult(
        symbol=symbol,
        data=data,
        close=close,
        test_start_index=test_start_index,
        test_predictions=test_predictions,
        lstm_metrics=lstm_metrics,
        baselines=baselines,
        baselines_frame=baselines_frame,
        history=history,
        future=future,
        forecast_with_change=forecast_with_change,
        params={
            "sequence_length": sequence_length,
            "epochs": epochs,
            "batch_size": batch_size,
        },
    )
