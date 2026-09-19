"""Seasonality analysis: weekday effects, monthly returns, calendar patterns."""

from __future__ import annotations

import pandas as pd


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
    table.index = ["Mon", "Tue", "Wed", "Thu", "Fri"]
    return table.rename_axis("weekday").reset_index()


def monthly_returns_matrix(close: pd.Series) -> pd.DataFrame:
    """Year x month grid of monthly returns, for a heatmap."""
    monthly = close.resample("ME").last().pct_change() * 100
    df = monthly.dropna().to_frame("return")
    dt = pd.Series(pd.DatetimeIndex(df.index))
    df["year"] = dt.dt.year.values
    df["month"] = dt.dt.month.values
    pivot = df.pivot(index="year", columns="month", values="return")
    pivot = pivot.reindex(columns=range(1, 13))
    pivot.columns = [
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
    ]
    return pivot


def month_effects(close: pd.Series) -> pd.DataFrame:
    """Average return, hit rate and count per calendar month.

    Short ranges may not contain every month (and a missing month would
    otherwise shift the labels); the table is reindexed to all twelve
    months so each row always maps to the right calendar month.
    """
    monthly = close.resample("ME").last().pct_change().dropna()
    months = pd.Series(monthly.index).dt.month.to_numpy()
    table = monthly.groupby(months).agg(
        mean="mean", hit_rate=lambda s: (s > 0).mean(), count="count"
    )
    table = table.reindex(range(1, 13))
    table["count"] = table["count"].fillna(0)
    table.index = [
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
    ]
    return table.rename_axis("month").reset_index()
