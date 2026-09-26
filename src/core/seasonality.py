"""Seasonality analysis: weekday effects, monthly returns, calendar patterns."""

from __future__ import annotations

import pandas as pd

MONTH_NAMES: tuple[str, ...] = (
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
)
"""January-first, matching `pd.Timestamp.month` and the 1-12 reindex below.

Defined once because two functions below both need it, and a hand-copied month
list is exactly the kind of thing that drifts: an earlier version had the two
copies one entry apart.
"""

WEEKDAY_NAMES: tuple[str, ...] = ("Mon", "Tue", "Wed", "Thu", "Fri")
"""Monday-first, matching `pd.Timestamp.dayofweek`."""


def _monthly_returns(close: pd.Series) -> pd.Series:
    """Month-end closes turned into month-on-month percentage returns.

    Shared by the heatmap and the month-effects table so the two cannot
    disagree about what "a month's return" means. `.dropna()` because the
    first month has no prior month to compare against.
    """
    return (close.resample("ME").last().pct_change() * 100).dropna()


def weekday_effects(returns: pd.Series) -> pd.DataFrame:
    """Mean return, hit rate, and count for each weekday (Mon-Fri)."""
    dt = pd.Series(pd.DatetimeIndex(returns.index))
    df = pd.DataFrame(
        {
            "return": returns.values,
            "weekday": dt.dt.dayofweek.values,
        }
    )
    grouped = df.groupby("weekday")["return"]
    table = pd.DataFrame(
        {
            "mean": grouped.mean(),
            "hit_rate": grouped.apply(lambda s: (s > 0).mean()),
            "count": grouped.count(),
        }
    )
    # A weekday can be missing entirely (e.g. a holiday week); reindex keeps
    # the table at five rows with NaN stats and a zero count for that day.
    table = table.reindex(range(5))
    table["count"] = table["count"].fillna(0)
    table.index = list(WEEKDAY_NAMES)
    return table.rename_axis("weekday").reset_index()


def monthly_returns_matrix(close: pd.Series) -> pd.DataFrame:
    """Year x month grid of monthly returns, for a heatmap."""
    df = _monthly_returns(close).to_frame("return")
    dt = pd.Series(pd.DatetimeIndex(df.index))
    df["year"] = dt.dt.year.values
    df["month"] = dt.dt.month.values
    pivot = df.pivot(index="year", columns="month", values="return")
    # Reindexed to all twelve so a range missing a month does not shift the
    # column labels, then named from the single shared list.
    pivot = pivot.reindex(columns=range(1, 13))
    pivot.columns = list(MONTH_NAMES)
    return pivot


def month_effects(close: pd.Series) -> pd.DataFrame:
    """Average return, hit rate and count per calendar month.

    Short ranges may not contain every month (and a missing month would
    otherwise shift the labels); the table is reindexed to all twelve
    months so each row always maps to the right calendar month.
    """
    monthly = _monthly_returns(close)
    months = pd.Series(monthly.index).dt.month.to_numpy()
    table = monthly.groupby(months).agg(
        mean="mean", hit_rate=lambda s: (s > 0).mean(), count="count"
    )
    table = table.reindex(range(1, 13))
    table["count"] = table["count"].fillna(0)
    table.index = list(MONTH_NAMES)
    return table.rename_axis("month").reset_index()
