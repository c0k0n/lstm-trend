"""Feature matrix for the tabular challenger models.

Every column is built from trailing data only. Nothing here may look past the
day it describes — in a walk-forward test, a single forward-looking column
would show up as spectacular skill and mean nothing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .indicators import rsi
from .market_context import CONTEXT_COLUMNS, align_context
from .preprocessing import log_returns

PARKINSON_CONSTANT: float = 1.0 / (4.0 * np.log(2.0))
"""Parkinson's 1980 high-low variance estimator constant.

The daily high-low range carries far more information about that day's variance
than the close-to-close move does — the range estimator is roughly five times
more efficient, meaning a 20-day range-based volatility is about as accurate as
a 100-day close-based one. On a series as noisy as this, that difference is
worth more than any architecture change.
"""

FEATURE_COLUMNS: tuple[str, ...] = (
    "ret_1",
    "ret_5",
    "ret_10",
    "ret_20",
    "vol_20",
    "vol_60",
    "dist_ma20",
    "dist_ma50",
    "rsi_14",
    "vol_ratio",
    "dow",
    "month",
)

OHLCV_COLUMNS: tuple[str, ...] = (
    "gap",
    "range_pct",
    "close_pos",
    "parkinson_20",
    "garman_klass_20",
    "volume_trend",
)
"""Structure from the parts of a candle the close alone throws away.

Every one of these is known at the close of the day it describes, so none of
them leak. They are usable for direct-horizon forecasting; they are *not*
usable for recursive forecasting, because at day t+1 you would need that day's
high, low and volume, which do not exist yet.
"""


def forward_log_return(close: pd.Series, horizon: int) -> pd.Series:
    """The target: log return from today's close to `horizon` trading days ahead.

    Rows without a full `horizon` of future data are NaN and get dropped when a
    model is fitted, so the last `horizon` days of history can never be trained
    on — which is exactly what makes the walk-forward test honest.
    """
    return np.log(close.shift(-horizon) / close)


def _ohlcv_features(data: pd.DataFrame) -> pd.DataFrame:
    """Candle structure and range-based volatility, from OHLCV only.

    Returns an empty frame (no columns) when the frame has no High/Low, rather
    than a frame full of NaN — a NaN column silently empties every training row
    downstream and makes the model fall back to naive without saying so.
    """
    if not {"High", "Low"}.issubset(data.columns):
        return pd.DataFrame(index=data.index)

    high = data["High"].astype(float)
    low = data["Low"].astype(float)
    close = data["Close"].astype(float)
    open_ = data["Open"].astype(float) if "Open" in data.columns else close.shift(1)

    feats = pd.DataFrame(index=data.index)
    # Where the day actually started relative to where the last one ended.
    feats["gap"] = np.log(open_ / close.shift(1))

    span = (high - low).replace(0.0, np.nan)
    feats["range_pct"] = span / close
    feats["close_pos"] = (close - low) / span

    log_range = np.log(high / low).replace([np.inf, -np.inf], np.nan)
    feats["parkinson_20"] = np.sqrt(
        (PARKINSON_CONSTANT * log_range.pow(2)).rolling(20).mean()
    )

    # Garman-Klass adds the open and close to the range, which the pure
    # range estimator ignores.
    gk = 0.5 * np.log(high / low).pow(2) - (2 * np.log(2) - 1) * np.log(
        close / open_
    ).pow(2)
    feats["garman_klass_20"] = np.sqrt(gk.rolling(20).mean().clip(lower=0.0))

    if "Volume" in data.columns:
        volume = data["Volume"].astype(float).replace(0.0, np.nan)
        feats["volume_trend"] = volume.rolling(5).mean() / volume.rolling(20).mean()
    else:
        feats["volume_trend"] = np.nan

    return feats


def build_features(
    data: pd.DataFrame,
    context: pd.DataFrame | None = None,
    ohlcv: bool = True,
) -> pd.DataFrame:
    """One row per trading day, built strictly from that day and earlier.

    `context` is the market-wide frame from `market_context.py`. It is aligned
    onto this stock's calendar by forward-fill, so a day only ever receives
    market information that already existed when that day's close printed.

    `ohlcv` adds the candle-structure columns. Turn it off for a recursive
    forecast, where future highs, lows and volumes do not exist yet.
    """
    close = data["Close"].astype(float)
    daily = log_returns(close)

    feats = pd.DataFrame(index=close.index)
    for periods in (1, 5, 10, 20):
        feats[f"ret_{periods}"] = log_returns(close, periods)

    feats["vol_20"] = daily.rolling(20).std()
    feats["vol_60"] = daily.rolling(60).std()

    for window in (20, 50):
        feats[f"dist_ma{window}"] = close / close.rolling(window).mean() - 1.0

    feats["rsi_14"] = rsi(close, period=14)

    if "Volume" in data.columns:
        volume = data["Volume"].astype(float)
        rolling = volume.rolling(20).mean()
        feats["vol_ratio"] = (volume / rolling).replace([np.inf, -np.inf], np.nan)
    else:
        feats["vol_ratio"] = np.nan

    index = pd.Series(feats.index)
    feats["dow"] = index.dt.dayofweek.to_numpy()
    feats["month"] = index.dt.month.to_numpy()

    columns = list(FEATURE_COLUMNS)
    if ohlcv:
        candle = _ohlcv_features(data)
        if not candle.empty:
            feats = feats.join(candle, how="left")
            columns.extend(OHLCV_COLUMNS)
    if context is not None:
        feats = feats.join(align_context(context, close.index), how="left")
        columns.extend(CONTEXT_COLUMNS)

    return feats[columns]
