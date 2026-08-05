"""All Plotly chart builders, consistently styled for the dark theme."""

from typing import Any

import pandas as pd
import plotly.graph_objects as go

from ..constants import (
    COLOR_ACTUAL,
    COLOR_FUTURE,
    COLOR_MA,
    COLOR_NAIVE,
    COLOR_PREDICTED,
    PLOT_BGCOLOR,
    PLOT_FONT_COLOR,
    PLOT_GRID_COLOR,
)
from ..core.baselines import MOVING_AVERAGE, NAIVE
from ..core.baselines import moving_average_forecast, naive_forecast


def _layout(
    title: str,
    xaxis_title: str = "Date",
    yaxis_title: str = "Price (USD)",
    height: int = 420,
) -> dict:
    return dict(
        title=dict(text=title),
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        template="plotly_dark",
        paper_bgcolor=PLOT_BGCOLOR,
        plot_bgcolor=PLOT_BGCOLOR,
        font=dict(color=PLOT_FONT_COLOR),
        xaxis=dict(gridcolor=PLOT_GRID_COLOR),
        yaxis=dict(gridcolor=PLOT_GRID_COLOR),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=height,
        margin=dict(t=60, b=40, l=50, r=20),
    )


def plot_close(close: pd.Series) -> go.Figure:
    fig = go.Figure(
        go.Scatter(x=close.index, y=close.values, mode="lines", name="Close price")
    )
    fig.update_layout(**_layout("Closing price"))
    return fig


def plot_volume(volume: pd.Series) -> go.Figure:
    fig = go.Figure(go.Bar(x=volume.index, y=volume.values, name="Volume"))
    fig.update_layout(**_layout("Trading volume", yaxis_title="Volume"))
    return fig


def plot_candlestick(data: pd.DataFrame) -> go.Figure:
    fig = go.Figure(
        go.Candlestick(
            x=data.index,
            open=data["Open"],
            high=data["High"],
            low=data["Low"],
            close=data["Close"],
            name="OHLC",
        )
    )
    fig.update_layout(**_layout("Candlestick chart", yaxis_title="Price (USD)"))
    return fig


def plot_loss_history(history: Any, title: str = "Training loss") -> go.Figure:
    fig = go.Figure()
    if history is not None and hasattr(history, "history"):
        hist = history.history
        if "loss" in hist:
            fig.add_trace(
                go.Scatter(y=hist["loss"], mode="lines", name="Training loss (MAE)")
            )
        if "val_loss" in hist:
            fig.add_trace(
                go.Scatter(y=hist["val_loss"], mode="lines", name="Validation loss")
            )
    fig.update_layout(**_layout(title, xaxis_title="Epoch", yaxis_title="Loss (MAE)"))
    return fig


def plot_test_predictions(
    test_predictions: pd.DataFrame,
    close: pd.Series,
    test_start_index: int,
    sequence_length: int,
) -> go.Figure:
    """Actual vs predicted prices over the test window, with a history tail."""
    history = close.iloc[max(0, test_start_index - sequence_length) : test_start_index]
    test_dates = close.index[test_start_index:]
    fig = go.Figure()

    if len(history) > 0:
        fig.add_trace(
            go.Scatter(
                x=history.index,
                y=history.values,
                mode="lines",
                name="Training history",
                line=dict(color=PLOT_GRID_COLOR, width=1),
            )
        )
    fig.add_trace(
        go.Scatter(
            x=test_dates,
            y=test_predictions["actual"].values,
            mode="lines",
            name="Actual (test)",
            line=dict(color=COLOR_ACTUAL),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=test_dates,
            y=test_predictions["predicted"].values,
            mode="lines",
            name="LSTM prediction",
            line=dict(color=COLOR_PREDICTED, dash="dot"),
        )
    )
    fig.update_layout(**_layout("Actual vs predicted — test window"))
    return fig


def plot_forecast(result) -> go.Figure:
    """Historical close + test predictions + future forecast in one view."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=result.close.index,
            y=result.close.values,
            mode="lines",
            name="Close price",
            line=dict(color=PLOT_GRID_COLOR, width=1.2),
        )
    )
    test_dates = result.close.index[result.test_start_index :]
    fig.add_trace(
        go.Scatter(
            x=test_dates,
            y=result.test_predictions["predicted"].values,
            mode="lines",
            name="LSTM on test window",
            line=dict(color=COLOR_PREDICTED, dash="dot"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=result.future["date"],
            y=result.future["predicted_close"].values,
            mode="lines+markers",
            name=f"Forecast ({len(result.future)} business days)",
            line=dict(color=COLOR_FUTURE, dash="dash"),
            marker=dict(size=6),
        )
    )
    fig.update_layout(**_layout(f"{result.symbol} — history and forecast"))
    return fig


def plot_baseline_comparison(result) -> go.Figure:
    """Actual test prices against every forecast (LSTM + baselines)."""
    test_dates = result.close.index[result.test_start_index :]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=test_dates,
            y=result.test_predictions["actual"].values,
            mode="lines",
            name="Actual (test)",
            line=dict(color=COLOR_ACTUAL, width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=test_dates,
            y=result.test_predictions["predicted"].values,
            mode="lines",
            name="LSTM",
            line=dict(color=COLOR_PREDICTED),
        )
    )

    naive = naive_forecast(result.close, result.test_start_index)
    moving = moving_average_forecast(
        result.close, result.test_start_index, result.params["sequence_length"]
    )
    fig.add_trace(
        go.Scatter(
            x=naive.index,
            y=naive.values,
            mode="lines",
            name=NAIVE,
            line=dict(color=COLOR_NAIVE, dash="dot"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=moving.index,
            y=moving.values,
            mode="lines",
            name=MOVING_AVERAGE,
            line=dict(color=COLOR_MA, dash="dash"),
        )
    )
    fig.update_layout(**_layout("LSTM vs baselines — test window"))
    return fig


def plot_metric_bars(result) -> go.Figure:
    """Horizontal bar chart of RMSE per method."""
    rows = [
        {"Method": "LSTM", "RMSE": result.lstm_metrics["rmse"]},
        *[
            {"Method": name, "RMSE": metrics["rmse"]}
            for name, metrics in result.baselines.items()
        ],
    ]
    df = pd.DataFrame(rows).sort_values("RMSE")
    fig = go.Figure(
        go.Bar(
            x=df["RMSE"],
            y=df["Method"],
            orientation="h",
            marker=dict(color=[COLOR_PREDICTED, COLOR_NAIVE, COLOR_MA]),
        )
    )
    fig.update_layout(
        **_layout("Test RMSE by method (lower is better)", yaxis_title="")
    )
    return fig
