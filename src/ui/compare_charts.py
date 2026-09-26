"""Compare page chart builders: multi-ticker visualizations."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from .chart_theme import PALETTE, layout


def _one_line_per_ticker(
    series: dict[str, pd.Series],
    transform,
    title: str,
    yaxis_title: str,
    height: int = 420,
) -> go.Figure:
    """One line per ticker, same colour assignment, same legend placement.

    Three of the charts on this page differ only in what they plot, and each
    carried its own copy of the loop — including the palette cycling, which is
    what makes the same ticker the same colour across all three. That is a
    property a reader relies on, so it lives in one place now.
    """
    fig = go.Figure()
    for i, (name, close) in enumerate(series.items()):
        plotted = transform(close)
        fig.add_trace(
            go.Scatter(
                x=plotted.index,
                y=plotted.values,
                mode="lines",
                name=name,
                line=dict(color=PALETTE[i % len(PALETTE)]),
            )
        )
    fig.update_layout(**layout(title, yaxis_title=yaxis_title, height=height))
    return fig


def plot_normalized_prices(series: dict[str, pd.Series]) -> go.Figure:
    from ..core.returns import normalize_series

    return _one_line_per_ticker(
        series,
        normalize_series,
        "Normalized price (start = 100)",
        "Indexed price",
    )


def plot_cumulative_comparison(series: dict[str, pd.Series]) -> go.Figure:
    from ..core.returns import cumulative_returns_series

    return _one_line_per_ticker(
        series, cumulative_returns_series, "Cumulative returns", "Growth factor"
    )


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
        **layout(
            "Correlation of daily returns", xaxis_title="", yaxis_title="", height=420
        )
    )
    return fig


def plot_drawdown_comparison(series: dict[str, pd.Series]) -> go.Figure:
    from ..core.risk import drawdown_series

    # Drawdown is a fraction; the chart is in percent, so the transform scales
    # here rather than every chart hard-coding a `* 100`.
    def as_percent(close: pd.Series) -> pd.Series:
        return drawdown_series(close) * 100

    return _one_line_per_ticker(
        series, as_percent, "Drawdown comparison", "Drawdown (%)", height=380
    )


def plot_risk_return_scatter(table: pd.DataFrame) -> go.Figure:
    """Risk against return, with Sharpe as the colour scale.

    Size is deliberately constant. Making it encode Sharpe as well would say
    the same thing twice and, worse, make a barely-different Sharpe look like
    a much bigger point — area reads as magnitude far faster than hue does.
    """
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
            hovertemplate="%{text}<br>ann vol %{x:.1%}<br>CAGR %{y:.1%}<br>Sharpe %{marker.color:.2f}<br>max DD %{customdata:.1%}<extra></extra>",
        )
    )
    fig.update_layout(
        **layout(
            "Risk vs return (colour = Sharpe, hover for details)",
            xaxis_title="Annualized volatility",
            yaxis_title="Annualized return (CAGR)",
            height=460,
        )
    )
    return fig
