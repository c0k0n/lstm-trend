"""Shared chart theme constants and layout helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd

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

DARK_THEME: dict[str, str] = dict(
    template="plotly_dark",
    font=PLOT_FONT_COLOR,
    grid=PLOT_GRID_COLOR,
    actual=COLOR_ACTUAL,
)

PALETTE: tuple[str, ...] = (
    COLOR_PREDICTED,
    COLOR_NAIVE,
    COLOR_FUTURE,
    COLOR_MA,
    "#EF5350",
    "#26A69A",
)

MILLIS_PER_DAY: int = 86_400_000


def with_alpha(hex_color: str, alpha: float) -> str:
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    return f"rgba({r},{g},{b},{alpha})"


def cumulative_returns(close: "pd.Series") -> "pd.Series":
    from ..core.returns import cumulative_returns_series

    return cumulative_returns_series(close)


def layout(
    title: str,
    xaxis_title: str = "Date",
    yaxis_title: str = "Price (USD)",
    height: int = 420,
) -> dict[str, Any]:
    return dict(
        title=dict(text=title),
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        template=DARK_THEME["template"],
        paper_bgcolor=PLOT_BGCOLOR,
        plot_bgcolor=PLOT_BGCOLOR,
        font=dict(color=DARK_THEME["font"]),
        xaxis=dict(gridcolor=DARK_THEME["grid"]),
        yaxis=dict(gridcolor=DARK_THEME["grid"]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=height,
        margin=dict(t=60, b=40, l=50, r=20),
    )
