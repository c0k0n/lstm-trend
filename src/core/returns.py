"""Return calculations, performance metrics, and comparison helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS: int = 252


def daily_returns(close: pd.Series) -> pd.Series:
    return close.pct_change().dropna()


def cumulative_return(close: pd.Series) -> float:
    return float(close.iloc[-1] / close.iloc[0] - 1) if len(close) > 1 else 0.0


def annualized_return(close: pd.Series) -> float:
    """Compound annual growth rate from first to last close."""
    if len(close) < 2:
        return 0.0
    first = close.index[0]
    last = close.index[-1]
    years = (last - first).days / 365.25
    if years <= 0:
        return 0.0
    return float((close.iloc[-1] / close.iloc[0]) ** (1 / years) - 1)


def annualized_volatility(returns: pd.Series, periods: int = TRADING_DAYS) -> float:
    return float(returns.std(ddof=1) * np.sqrt(periods)) if len(returns) > 1 else 0.0


def rolling_volatility(
    returns: pd.Series, window: int = 20, periods: int = TRADING_DAYS
) -> pd.Series:
    return returns.rolling(window).std(ddof=1) * np.sqrt(periods)


def period_returns(close: pd.Series) -> dict[str, float]:
    """Trailing returns over common horizons, in percent."""
    last_date = close.index[-1]
    today = close.iloc[-1]
    out: dict[str, float] = {}
    for label, bdays in (("1M", 21), ("3M", 63), ("6M", 126), ("1Y", 252)):
        past = close.asof(last_date - pd.offsets.BDay(bdays))
        out[label] = float((today / past - 1) * 100) if not np.isnan(past) else np.nan

    year_start = close.asof(pd.Timestamp(year=last_date.year, month=1, day=1))
    out["YTD"] = (
        float((today / year_start - 1) * 100) if not np.isnan(year_start) else np.nan
    )
    return out


def position_in_52w_range(close: pd.Series, window: int = 252) -> float:
    """Where today's price sits between the 52-week low and high, 0-1."""
    recent = close.iloc[-window:]
    lo, hi = recent.min(), recent.max()
    if lo == hi:
        return 1.0
    return float((close.iloc[-1] - lo) / (hi - lo))


def normalize_series(close: pd.Series, base: float = 100.0) -> pd.Series:
    return close / close.iloc[0] * base
