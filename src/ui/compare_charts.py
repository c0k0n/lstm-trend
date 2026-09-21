"""Compare page chart builders: multi-ticker visualizations."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from .chart_theme import PALETTE, layout


def plot_normalized_prices(series: dict[str, pd.Series]) -> go.Figure:
    from ..core.returns import normalize_series

    fig = go.Figure()
    colors = PALETTE
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
        **layout("Normalized price (start = 100)", yaxis_title="Indexed price")
    )
    return fig


def plot_cumulative_comparison(series: dict[str, pd.Series]) -> go.Figure:
    from ..core.returns import cumulative_returns_series

    fig = go.Figure()
    colors = PALETTE
    for i, (name, close) in enumerate(series.items()):
        cum = cumulative_returns_series(close)
        fig.add_trace(
            go.Scatter(
                x=cum.index,
                y=cum.values,
                mode="lines",
                name=name,
                line=dict(color=colors[i % len(colors)]),
            )
        )
    fig.update_layout(**layout("Cumulative returns", yaxis_title="Growth factor"))
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
        **layout(
            "Correlation of daily returns", xaxis_title="", yaxis_title="", height=420
        )
    )
    return fig


def plot_drawdown_comparison(series: dict[str, pd.Series]) -> go.Figure:
    from ..core.risk import drawdown_series

    fig = go.Figure()
    colors = PALETTE
    for i, (name, close) in enumerate(series.items()):
        dd = drawdown_series(close)
        fig.add_trace(
            go.Scatter(
                x=dd.index,
                y=dd.to_numpy() * 100,
                mode="lines",
                name=name,
                line=dict(color=colors[i % len(colors)]),
            )
        )
    fig.update_layout(
        **layout("Drawdown comparison", yaxis_title="Drawdown (%)", height=380)
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
        **layout(
            "Risk vs return (size = Sharpe, hover for details)",
            xaxis_title="Annualized volatility",
            yaxis_title="Annualized return (CAGR)",
            height=460,
        )
    )
    return fig
