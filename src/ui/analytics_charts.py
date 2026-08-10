"""Analytics page chart builders: returns, risk, seasonality, indicators, volume."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ..constants import COLOR_FUTURE, COLOR_MA, COLOR_NAIVE, COLOR_PREDICTED
from .chart_theme import DARK_THEME, MILLIS_PER_DAY, PALETTE, layout, with_alpha


def plot_cumulative_returns(close: pd.Series, log_scale: bool = True) -> go.Figure:
    from .chart_theme import cumulative_returns

    cum = cumulative_returns(close)
    fig = go.Figure(
        go.Scatter(x=cum.index, y=cum.values, mode="lines", name="Cumulative return")
    )
    fig.update_layout(
        **layout("Cumulative return (1 = start)", yaxis_title="Growth factor"),
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
        **layout(
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
    osm_arr = np.asarray(osm, dtype=float)
    osr_arr = np.asarray(osr, dtype=float)
    line_y = slope * osm_arr + intercept
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=osm_arr, y=osr_arr, mode="markers", name="Observed", marker=dict(size=4)
        )
    )
    fig.add_trace(
        go.Scatter(
            x=osm_arr,
            y=line_y,
            mode="lines",
            name="Normal line",
            line=dict(color=COLOR_PREDICTED, dash="dash"),
        )
    )
    fig.update_layout(
        **layout(
            "Q-Q plot vs normal distribution",
            xaxis_title="Theoretical quantiles",
            yaxis_title="Sample quantiles",
        )
    )
    return fig


def plot_rolling_volatility(returns: pd.Series) -> go.Figure:
    from ..core.returns import rolling_volatility

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
        **layout("Rolling 20-day volatility (annualized)", yaxis_title="Volatility")
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
        **layout(
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
                color=[COLOR_PREDICTED if v >= 0 else PALETTE[4] for v in table["mean"]]
            ),
        )
    )
    fig.update_layout(
        **layout(
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
        **layout("Monthly returns (%)", xaxis_title="", yaxis_title="Year", height=380)
    )
    return fig


def plot_underwater(close: pd.Series) -> go.Figure:
    from ..core.risk import drawdown_series

    dd = drawdown_series(close)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=dd.index,
            y=dd.to_numpy() * 100,
            mode="lines",
            name="Drawdown",
            fill="tozeroy",
        )
    )
    fig.update_layout(
        **layout("Drawdown (underwater curve)", yaxis_title="Drawdown (%)", height=340)
    )
    return fig


def plot_price_with_indicators(
    close: pd.Series, crosses: pd.DataFrame | None = None
) -> go.Figure:
    from ..core.indicators import ema, sma

    fig = go.Figure(
        go.Scatter(
            x=close.index,
            y=close.values,
            mode="lines",
            name="Close",
            line=dict(color=DARK_THEME["actual"]),
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
                    marker=dict(symbol="triangle-down", size=12, color=PALETTE[4]),
                )
            )
    fig.update_layout(**layout("Price with moving averages", height=460))
    return fig


def plot_rsi(close: pd.Series) -> go.Figure:
    from ..core.indicators import rsi

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
        y=70, line_dash="dash", line_color=PALETTE[4], annotation_text="Overbought"
    )
    fig.add_hline(
        y=30, line_dash="dash", line_color=PALETTE[5], annotation_text="Oversold"
    )
    fig.update_layout(
        **layout("Relative Strength Index (14)", yaxis_title="RSI", height=280)
    )
    return fig


def plot_macd(close: pd.Series) -> go.Figure:
    from ..core.indicators import macd

    m = macd(close).dropna()
    fig = go.Figure()
    colors = [COLOR_PREDICTED if v >= 0 else PALETTE[4] for v in m["Histogram"]]
    fig.add_trace(
        go.Bar(x=m.index, y=m["Histogram"], name="Histogram", marker=dict(color=colors))
    )
    fig.add_trace(
        go.Scatter(
            x=m.index,
            y=m["MACD"],
            mode="lines",
            name="MACD",
            line=dict(color=DARK_THEME["actual"]),
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
    fig.update_layout(**layout("MACD (12, 26, 9)", yaxis_title="", height=300))
    return fig


def plot_bollinger(close: pd.Series) -> go.Figure:
    from ..core.indicators import bollinger_bands

    bb = bollinger_bands(close)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=close.index,
            y=close.values,
            mode="lines",
            name="Close",
            line=dict(color=DARK_THEME["actual"]),
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
            fillcolor=with_alpha(PALETTE[1], 0.08),
        )
    )
    fig.update_layout(**layout("Bollinger bands (20, 2σ)", height=360))
    return fig


def plot_volume_analysis(data: pd.DataFrame) -> go.Figure:
    aggregated = len(data) > 260
    if aggregated:
        agg = (
            data[["Close", "Volume"]]
            .resample("W-FRI")
            .agg({"Close": "last", "Volume": "sum"})
            .dropna()
        )
        close = agg["Close"]
        volume = agg["Volume"]
    else:
        close = data["Close"]
        volume = data["Volume"]
    ret = close.pct_change().fillna(0)
    colors = [COLOR_PREDICTED if v >= 0 else PALETTE[4] for v in ret]
    bar_width = 5 * MILLIS_PER_DAY if aggregated else MILLIS_PER_DAY
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
    fig.update_layout(**layout(title, yaxis_title="Volume", height=340))
    return fig


def plot_volume_return_scatter(data: pd.DataFrame) -> go.Figure:
    close = data["Close"]
    volume = data["Volume"]
    ret = close.pct_change().fillna(0)
    fig = go.Figure(
        go.Scatter(
            x=volume.to_numpy(),
            y=ret.to_numpy(),
            mode="markers",
            name="Day",
            marker=dict(size=5, color=with_alpha(PALETTE[1], 0.6)),
            customdata=close.index.to_series().dt.strftime("%Y-%m-%d"),
            hovertemplate="%{customdata}<br>vol %{x:,.0f}<br>return %{y:.2%}<extra></extra>",
        )
    )
    fig.update_layout(
        **layout(
            "Volume vs daily return",
            xaxis_title="Volume",
            yaxis_title="Daily return",
            height=340,
        )
    )
    return fig
