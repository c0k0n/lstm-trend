"""All Plotly chart builders, consistently styled for the app theme."""

from typing import Any

import numpy as np
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

_DARK = True


def set_dark(dark: bool) -> None:
    """Switch chart styling between the dark and light app themes."""
    global _DARK
    _DARK = dark


def _theme() -> dict[str, Any]:
    if _DARK:
        return dict(
            template="plotly_dark",
            font=PLOT_FONT_COLOR,
            grid=PLOT_GRID_COLOR,
            actual=COLOR_ACTUAL,
        )
    return dict(
        template="plotly_white",
        font="#31333F",
        grid="rgba(49,51,63,0.10)",
        actual="#31333F",
    )


def _layout(
    title: str,
    xaxis_title: str = "Date",
    yaxis_title: str = "Price (USD)",
    height: int = 420,
) -> dict:
    t = _theme()
    return dict(
        title=dict(text=title),
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        template=t["template"],
        paper_bgcolor=PLOT_BGCOLOR,
        plot_bgcolor=PLOT_BGCOLOR,
        font=dict(color=t["font"]),
        xaxis=dict(gridcolor=t["grid"]),
        yaxis=dict(gridcolor=t["grid"]),
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
    df = data[["Open", "High", "Low", "Close"]]
    aggregated = False
    if len(df) > 260:
        # Too many daily bars to read: aggregate into weekly OHLC
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
    fig.update_layout(**_layout(title, yaxis_title="Price (USD)"))
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
    t = _theme()
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
            line=dict(color=_theme()["grid"], width=1.2),
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
            line=dict(color=_theme()["actual"], width=2),
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


# --------------------------------------------------------------------------- #
# Analytics page charts
# --------------------------------------------------------------------------- #
def plot_cumulative_returns(close: pd.Series, log_scale: bool = True) -> go.Figure:
    cum = (1 + close.pct_change().fillna(0)).cumprod()
    fig = go.Figure(
        go.Scatter(x=cum.index, y=cum.values, mode="lines", name="Cumulative return")
    )
    fig.update_layout(
        **_layout("Cumulative return (1 = start)", yaxis_title="Growth factor"),
        yaxis_type="log" if log_scale else "linear",
    )
    return fig


def plot_return_histogram(returns: pd.Series) -> go.Figure:
    from scipy.stats import gaussian_kde

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(x=returns.values, nbinsx=60, name="Daily returns", opacity=0.7)
    )
    xs = np.linspace(returns.min(), returns.max(), 300)
    kde = gaussian_kde(returns.values)
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=kde(xs) * len(returns) * (returns.max() - returns.min()) / 60,
            mode="lines",
            name="KDE",
            line=dict(color=COLOR_PREDICTED),
        )
    )
    fig.update_layout(
        **_layout(
            "Distribution of daily returns",
            yaxis_title="Count",
            xaxis_title="Daily return",
        )
    )
    return fig


def plot_qq(returns: pd.Series) -> go.Figure:
    from scipy import stats as scipy_stats

    (osm, osr), (slope, intercept, _) = scipy_stats.probplot(
        returns.dropna(), dist="norm"
    )
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=osm, y=osr, mode="markers", name="Observed", marker=dict(size=4))
    )
    fig.add_trace(
        go.Scatter(
            x=osm,
            y=slope * osm + intercept,
            mode="lines",
            name="Normal line",
            line=dict(color=COLOR_PREDICTED, dash="dash"),
        )
    )
    fig.update_layout(
        **_layout(
            "Q-Q plot vs normal distribution",
            xaxis_title="Theoretical quantiles",
            yaxis_title="Sample quantiles",
        )
    )
    return fig


def plot_rolling_volatility(returns: pd.Series) -> go.Figure:
    from ..core.analytics import rolling_volatility

    fig = go.Figure(
        go.Scatter(
            x=returns.index,
            y=rolling_volatility(returns).values,
            mode="lines",
            name="20-day ann. volatility",
            line=dict(color=COLOR_NAIVE),
        )
    )
    fig.update_layout(
        **_layout("Rolling 20-day volatility (annualized)", yaxis_title="Volatility")
    )
    return fig


def plot_acf(acf_values: pd.Series) -> go.Figure:
    fig = go.Figure(
        go.Bar(
            x=acf_values.index,
            y=acf_values.values,
            name="Autocorrelation",
            marker=dict(color=COLOR_NAIVE),
        )
    )
    fig.update_layout(
        **_layout(
            "Autocorrelation of daily returns",
            xaxis_title="Lag (days)",
            yaxis_title="ACF",
        )
    )
    return fig


def plot_weekday_effects(table: pd.DataFrame) -> go.Figure:
    fig = go.Figure(
        go.Bar(
            x=table["weekday"],
            y=table["mean"],
            name="Mean return",
            marker=dict(
                color=[COLOR_PREDICTED if v >= 0 else "#EF5350" for v in table["mean"]]
            ),
        )
    )
    fig.update_layout(
        **_layout(
            "Average daily return by weekday", xaxis_title="", yaxis_title="Mean return"
        )
    )
    return fig


def plot_monthly_heatmap(matrix: pd.DataFrame) -> go.Figure:
    fig = go.Figure(
        go.Heatmap(
            z=matrix.values,
            x=list(matrix.columns),
            y=[str(y) for y in matrix.index],
            colorscale="RdYlGn",
            zmid=0,
            text=matrix.values.round(1).astype(str) + "%",
            texttemplate="%{text}",
            colorbar=dict(title="%"),
        )
    )
    fig.update_layout(
        **_layout("Monthly returns (%)", xaxis_title="", yaxis_title="Year", height=380)
    )
    return fig


def plot_underwater(close: pd.Series) -> go.Figure:
    from ..core.analytics import drawdown_series

    dd = drawdown_series(close)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dd.index, y=dd.values * 100, mode="lines", name="Drawdown", fill="tozeroy"
        )
    )
    fig.update_layout(
        **_layout("Drawdown (underwater curve)", yaxis_title="Drawdown (%)", height=340)
    )
    return fig


def plot_price_with_indicators(
    close: pd.Series, crosses: pd.DataFrame | None = None
) -> go.Figure:
    from ..core.analytics import ema, sma

    fig = go.Figure(
        go.Scatter(
            x=close.index,
            y=close.values,
            mode="lines",
            name="Close",
            line=dict(color=_theme()["actual"]),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=close.index,
            y=sma(close, 20).values,
            mode="lines",
            name="SMA 20",
            line=dict(color=COLOR_NAIVE),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=close.index,
            y=sma(close, 50).values,
            mode="lines",
            name="SMA 50",
            line=dict(color=COLOR_PREDICTED),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=close.index,
            y=sma(close, 200).values,
            mode="lines",
            name="SMA 200",
            line=dict(color=COLOR_MA),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=close.index,
            y=ema(close, 50).values,
            mode="lines",
            name="EMA 50",
            line=dict(color=COLOR_FUTURE, dash="dot"),
        )
    )
    if crosses is not None and len(crosses):
        golden = crosses[crosses["Direction"] == "Golden"]
        death = crosses[crosses["Direction"] == "Death"]
        if len(golden):
            fig.add_trace(
                go.Scatter(
                    x=golden["Date"],
                    y=close.loc[golden["Date"]].values,
                    mode="markers",
                    name="Golden cross",
                    marker=dict(symbol="triangle-up", size=12, color=COLOR_PREDICTED),
                )
            )
        if len(death):
            fig.add_trace(
                go.Scatter(
                    x=death["Date"],
                    y=close.loc[death["Date"]].values,
                    mode="markers",
                    name="Death cross",
                    marker=dict(symbol="triangle-down", size=12, color="#EF5350"),
                )
            )
    fig.update_layout(**_layout("Price with moving averages", height=460))
    return fig


def plot_rsi(close: pd.Series) -> go.Figure:
    from ..core.analytics import rsi

    r = rsi(close)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=r.index,
            y=r.values,
            mode="lines",
            name="RSI 14",
            line=dict(color=COLOR_PREDICTED),
        )
    )
    fig.add_hline(
        y=70, line_dash="dash", line_color="#EF5350", annotation_text="Overbought"
    )
    fig.add_hline(
        y=30, line_dash="dash", line_color="#26A69A", annotation_text="Oversold"
    )
    fig.update_layout(
        **_layout("Relative Strength Index (14)", yaxis_title="RSI", height=280)
    )
    return fig


def plot_macd(close: pd.Series) -> go.Figure:
    from ..core.analytics import macd

    m = macd(close).dropna()
    fig = go.Figure()
    colors = [COLOR_PREDICTED if v >= 0 else "#EF5350" for v in m["Histogram"]]
    fig.add_trace(
        go.Bar(x=m.index, y=m["Histogram"], name="Histogram", marker=dict(color=colors))
    )
    fig.add_trace(
        go.Scatter(
            x=m.index,
            y=m["MACD"],
            mode="lines",
            name="MACD",
            line=dict(color=_theme()["actual"]),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=m.index,
            y=m["Signal"],
            mode="lines",
            name="Signal",
            line=dict(color=COLOR_FUTURE),
        )
    )
    fig.update_layout(**_layout("MACD (12, 26, 9)", yaxis_title="", height=300))
    return fig


def plot_bollinger(close: pd.Series) -> go.Figure:
    from ..core.analytics import bollinger_bands

    bb = bollinger_bands(close)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=close.index,
            y=close.values,
            mode="lines",
            name="Close",
            line=dict(color=_theme()["actual"]),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=bb.index,
            y=bb["Upper"].values,
            mode="lines",
            name="Upper",
            line=dict(color=COLOR_NAIVE, dash="dash"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=bb.index,
            y=bb["Lower"].values,
            mode="lines",
            name="Lower",
            line=dict(color=COLOR_NAIVE, dash="dash"),
            fill="tonexty",
            fillcolor="rgba(100,181,246,0.08)",
        )
    )
    fig.update_layout(**_layout("Bollinger bands (20, 2σ)", height=360))
    return fig


def plot_volume_analysis(data: pd.DataFrame) -> go.Figure:
    close = data["Close"]
    volume = data["Volume"]
    aggregated = False
    if len(data) > 260:
        # Long histories need aggregated bars or they become unreadable
        agg = (
            data[["Close", "Volume"]]
            .resample("W-FRI")
            .agg({"Close": "last", "Volume": "sum"})
            .dropna()
        )
        close, volume = agg["Close"], agg["Volume"]
        aggregated = True
    ret = close.pct_change().fillna(0)
    colors = [COLOR_PREDICTED if v >= 0 else "#EF5350" for v in ret]
    # Explicit bar width in milliseconds: one trading day, or five for weekly
    bar_width = 5 * 86_400_000 if aggregated else 86_400_000
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=volume.index,
            y=volume.values,
            width=bar_width,
            name="Volume",
            marker=dict(color=colors),
            opacity=0.7,
        )
    )
    title = "Volume by weekly move" if aggregated else "Volume coloured by daily move"
    fig.update_layout(**_layout(title, yaxis_title="Volume", height=340))
    return fig


def plot_volume_return_scatter(data: pd.DataFrame) -> go.Figure:
    close = data["Close"]
    volume = data["Volume"]
    ret = close.pct_change().fillna(0)
    fig = go.Figure(
        go.Scatter(
            x=volume.values,
            y=ret.values,
            mode="markers",
            name="Day",
            marker=dict(size=5, color="rgba(100,181,246,0.6)"),
            customdata=close.index.strftime("%Y-%m-%d"),
            hovertemplate="%{customdata}<br>vol %{x:,.0f}<br>return %{y:.2%}<extra></extra>",
        )
    )
    fig.update_layout(
        **_layout(
            "Volume vs daily return",
            xaxis_title="Volume",
            yaxis_title="Daily return",
            height=340,
        )
    )
    return fig


# --------------------------------------------------------------------------- #
# Compare page charts
# --------------------------------------------------------------------------- #
def plot_normalized_prices(series: dict[str, pd.Series]) -> go.Figure:
    from ..core.analytics import normalize_series

    fig = go.Figure()
    colors = ["#4FB477", "#64B5F6", "#FFA726", "#AB47BC", "#EF5350", "#26A69A"]
    for i, (name, close) in enumerate(series.items()):
        fig.add_trace(
            go.Scatter(
                x=close.index,
                y=normalize_series(close).values,
                mode="lines",
                name=name,
                line=dict(color=colors[i % len(colors)]),
            )
        )
    fig.update_layout(
        **_layout("Normalized price (start = 100)", yaxis_title="Indexed price")
    )
    return fig


def plot_cumulative_comparison(series: dict[str, pd.Series]) -> go.Figure:
    fig = go.Figure()
    colors = ["#4FB477", "#64B5F6", "#FFA726", "#AB47BC", "#EF5350", "#26A69A"]
    for i, (name, close) in enumerate(series.items()):
        cum = (1 + close.pct_change().fillna(0)).cumprod()
        fig.add_trace(
            go.Scatter(
                x=cum.index,
                y=cum.values,
                mode="lines",
                name=name,
                line=dict(color=colors[i % len(colors)]),
            )
        )
    fig.update_layout(**_layout("Cumulative returns", yaxis_title="Growth factor"))
    return fig


def plot_correlation_heatmap(corr: pd.DataFrame) -> go.Figure:
    fig = go.Figure(
        go.Heatmap(
            z=corr.values,
            x=list(corr.columns),
            y=list(corr.index),
            colorscale="RdBu",
            zmin=-1,
            zmax=1,
            text=corr.values.round(2).astype(str),
            texttemplate="%{text}",
        )
    )
    fig.update_layout(
        **_layout(
            "Correlation of daily returns", xaxis_title="", yaxis_title="", height=420
        )
    )
    return fig


def plot_drawdown_comparison(series: dict[str, pd.Series]) -> go.Figure:
    from ..core.analytics import drawdown_series

    fig = go.Figure()
    colors = ["#4FB477", "#64B5F6", "#FFA726", "#AB47BC", "#EF5350", "#26A69A"]
    for i, (name, close) in enumerate(series.items()):
        dd = drawdown_series(close)
        fig.add_trace(
            go.Scatter(
                x=dd.index,
                y=dd.values * 100,
                mode="lines",
                name=name,
                line=dict(color=colors[i % len(colors)]),
            )
        )
    fig.update_layout(
        **_layout("Drawdown comparison", yaxis_title="Drawdown (%)", height=380)
    )
    return fig


def plot_risk_return_scatter(table: pd.DataFrame) -> go.Figure:
    fig = go.Figure(
        go.Scatter(
            x=table["Ann. volatility"],
            y=table["CAGR"],
            mode="markers+text",
            text=table.index,
            textposition="top center",
            marker=dict(
                size=14,
                color=table["Sharpe"],
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Sharpe"),
            ),
            customdata=table["Max drawdown"],
            hovertemplate="%{text}<br>ann vol %{x:.1%}<br>CAGR %{y:.1%}<br>max DD %{customdata:.1%}<extra></extra>",
        )
    )
    fig.update_layout(
        **_layout(
            "Risk vs return (size = Sharpe, hover for details)",
            xaxis_title="Annualized volatility",
            yaxis_title="Annualized return (CAGR)",
            height=460,
        )
    )
    return fig
