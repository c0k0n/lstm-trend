"""Evidence page charts: skill per model, and error at every walk-forward window."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from .chart_theme import COLOR_PREDICTED, PALETTE, layout

NEGATIVE_COLOR = "#EF5350"
BASELINE_COLOR = "#888780"


def plot_model_skill(summary: pd.DataFrame) -> go.Figure:
    """Horizontal skill bars. Right of zero means better than 'repeat yesterday'."""
    ordered = summary.iloc[::-1]  # plotly stacks upward, so flip for top-down reads
    colours = []
    for _, row in ordered.iterrows():
        if bool(row["is_baseline"]):
            colours.append(BASELINE_COLOR)
        elif row["skill"] > 0:
            colours.append(COLOR_PREDICTED)
        else:
            colours.append(NEGATIVE_COLOR)

    fig = go.Figure(
        go.Bar(
            x=ordered["skill"],
            y=ordered["model"],
            orientation="h",
            marker=dict(color=colours),
            text=[f"{v:+.1%}" for v in ordered["skill"]],
            textposition="outside",
            hovertemplate="%{y}<br>skill %{x:+.1%}<extra></extra>",
        )
    )
    fig.add_vline(x=0, line=dict(color="#EF9F27", width=1, dash="dash"))
    fig.update_layout(
        **layout(
            "Skill against 'repeat yesterday'",
            xaxis_title="Skill (positive beats the naive guess)",
            yaxis_title="",
            height=320,
        )
    )
    fig.update_xaxes(tickformat="+.0%")
    return fig


def plot_window_errors(frame: pd.DataFrame) -> go.Figure:
    """Absolute error at each origin, one line per model.

    The point of this chart is the *shape*: a model whose line swings wildly
    from window to window is not reliably better, even if its average looks
    good. Smooth and below the baseline is what a real edge looks like.
    """
    fig = go.Figure()
    models = [c for c in frame.columns if c not in ("origin", "target", "actual")]
    actual = frame["actual"].to_numpy(dtype=float)

    for i, model in enumerate(models):
        error = abs(actual - frame[model].to_numpy(dtype=float))
        fig.add_trace(
            go.Scatter(
                x=frame["origin"],
                y=error,
                mode="lines+markers",
                name=model,
                line=dict(color=PALETTE[i % len(PALETTE)], width=2),
                hovertemplate="%{x|%b %Y}<br>error $%{y:.2f}<extra>"
                + model
                + "</extra>",
            )
        )

    fig.update_layout(
        **layout(
            "Error at every window",
            xaxis_title="Forecast origin",
            yaxis_title="Absolute error (USD)",
            height=380,
        )
    )
    return fig


def plot_ticker_spread(frames: dict[str, pd.DataFrame], model: str) -> go.Figure:
    """Skill per ticker for one model, sorted worst to best.

    This is the chart that stops you quoting the best ticker. A pooled number
    hides the fact that the same model can be +18% on one name and −100% on
    another, and picking the winner after the fact is how a demo turns into a
    claim. Seeing the whole spread is the point.
    """
    rows = []
    for symbol, frame in frames.items():
        if model not in frame.columns:
            continue
        naive = float((frame["Naive (yesterday)"] - frame["actual"]).abs().mean())
        error = float((frame[model] - frame["actual"]).abs().mean())
        rows.append({"ticker": symbol, "skill": 1.0 - error / naive if naive else 0.0})

    if not rows:
        return go.Figure()

    ordered = sorted(rows, key=lambda row: row["skill"])
    skills = [row["skill"] for row in ordered]
    tickers = [row["ticker"] for row in ordered]

    # Monochrome on purpose: length carries the value, and a red/green ramp
    # would imply a verdict the sample cannot support anyway.
    fig = go.Figure(
        go.Bar(
            x=skills,
            y=tickers,
            orientation="h",
            marker=dict(
                color=skills,
                colorscale=[[0, "#F09595"], [0.5, "#D3D1C7"], [1, "#9FE1CB"]],
                cmid=0.0,
                line=dict(color="#888780", width=0.5),
            ),
            hovertemplate="%{y}<br>skill %{x:+.1%}<extra></extra>",
        )
    )
    fig.add_vline(x=0, line=dict(color="#EF9F27", width=1, dash="dash"))
    fig.update_layout(
        **layout(
            f"{model} — skill per ticker",
            xaxis_title="Skill (right of zero beats repeating yesterday)",
            yaxis_title="",
            height=max(280, 34 * len(tickers) + 120),
        )
    )
    fig.update_xaxes(tickformat="+.0%")
    return fig
