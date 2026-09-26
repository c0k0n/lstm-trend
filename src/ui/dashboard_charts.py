"""Dashboard chart builders: candlestick, loss history, test predictions, forecast, baselines."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd
import plotly.graph_objects as go

from ..constants import COLOR_FUTURE, COLOR_MA, COLOR_NAIVE, COLOR_PREDICTED
from ..core.baselines import MOVING_AVERAGE, NAIVE
from .chart_theme import DARK_THEME, layout, with_alpha

if TYPE_CHECKING:
    from ..core.pipeline import AnalysisResult


def plot_candlestick(data: pd.DataFrame) -> go.Figure:
    df = data[["Open", "High", "Low", "Close"]]
    aggregated = False
    if len(df) > 260:
        df = (
            df.resample("W-FRI")
            .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"})
            .dropna()
        )
        aggregated = True
    fig = go.Figure(
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="OHLC",
            increasing=dict(line=dict(width=1.5)),
            decreasing=dict(line=dict(width=1.5)),
        )
    )
    title = "Candlestick chart (weekly bars)" if aggregated else "Candlestick chart"
    fig.update_layout(**layout(title, yaxis_title="Price (USD)"))
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
    fig.update_layout(**layout(title, xaxis_title="Epoch", yaxis_title="Loss (MAE)"))
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
    t = DARK_THEME
    fig = go.Figure()

    if len(history) > 0:
        fig.add_trace(
            go.Scatter(
                x=history.index,
                y=history.values,
                mode="lines",
                name="Training history",
                line=dict(color=t["grid"], width=1),
            )
        )
    fig.add_trace(
        go.Scatter(
            x=test_dates,
            y=test_predictions["actual"].values,
            mode="lines",
            name="Actual (test)",
            line=dict(color=t["actual"]),
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
    fig.update_layout(**layout("Actual vs predicted — test window"))
    return fig


def plot_forecast(
    result: "AnalysisResult", band: pd.DataFrame | None = None
) -> go.Figure:
    """Historical close + test predictions + future forecast in one view.

    `band` carries `lower` and `upper` columns aligned with the forecast dates;
    when given, the forecast is drawn inside a shaded range instead of as a bare
    line.
    """
    fig = go.Figure()

    if band is not None:
        fig.add_trace(
            go.Scatter(
                x=result.future["date"],
                y=band["upper"].to_numpy(),
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=result.future["date"],
                y=band["lower"].to_numpy(),
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor=with_alpha(COLOR_FUTURE, 0.18),
                name="Uncertainty band",
                hoverinfo="skip",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=result.close.index,
            y=result.close.values,
            mode="lines",
            name="Close price",
            line=dict(color=DARK_THEME["grid"], width=1.2),
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
    fig.update_layout(**layout(f"{result.symbol} — history and forecast"))
    return fig


def plot_baseline_comparison(result: "AnalysisResult") -> go.Figure:
    """Actual test prices against every forecast (LSTM + baselines)."""
    test_dates = result.close.index[result.test_start_index :]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=test_dates,
            y=result.test_predictions["actual"].values,
            mode="lines",
            name="Actual (test)",
            line=dict(color=DARK_THEME["actual"], width=2),
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
    fig.add_trace(
        go.Scatter(
            x=test_dates,
            y=result.baselines_frame[NAIVE].values,
            mode="lines",
            name=NAIVE,
            line=dict(color=COLOR_NAIVE, dash="dot"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=test_dates,
            y=result.baselines_frame[MOVING_AVERAGE].values,
            mode="lines",
            name=MOVING_AVERAGE,
            line=dict(color=COLOR_MA, dash="dash"),
        )
    )
    fig.update_layout(**layout("LSTM vs baselines — test window"))
    return fig


def plot_metric_bars(result: "AnalysisResult") -> go.Figure:
    """Horizontal bar chart of RMSE per method."""
    color_by_method = {
        "LSTM": COLOR_PREDICTED,
        NAIVE: COLOR_NAIVE,
        MOVING_AVERAGE: COLOR_MA,
    }
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
            marker=dict(color=[color_by_method[m] for m in df["Method"]]),
        )
    )
    fig.update_layout(**layout("Test RMSE by method (lower is better)", yaxis_title=""))
    return fig
