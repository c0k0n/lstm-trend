"""Market-wide context: what the whole market was doing on a given day.

`features.py` describes one stock in isolation — its own returns, its own
volatility, its own distance from its moving average. That is a description of
the patient with no information about the weather. These columns describe the
weather: implied volatility, the broad market's recent direction, the policy
rate, and the stock's own sector.

This is the largest untried lever in the upgrade path and the cheapest to add.
It is also the easiest place to introduce leakage, for a reason that has nothing
to do with the maths: market data is indexed on *its own* calendar, and joining
it onto a stock's calendar the obvious way pulls tomorrow's values backwards
into today's row. So the join here is a forward-fill, explicitly, and there is
a test that fails if it ever stops being one.

Columns are built from that day and earlier only. Nothing in this module may
read a date it does not yet have.
"""

from __future__ import annotations

import datetime
import logging

import numpy as np
import pandas as pd

from .data_loader import download_stock_data
from .preprocessing import log_returns

logger = logging.getLogger(__name__)

VIX = "^VIX"
MARKET = "SPY"
SHORT_RATE = "^IRX"
"""13-week Treasury bill rate, annualised percent.

Not a rate series you would model with, but it is the shortest, cleanest
proxy for "what does cash pay" that yfinance serves daily without an API key,
and the mechanism being tested is regime, not term structure.
"""

MARKET_COLUMNS: tuple[str, ...] = (
    "vix_level",
    "vix_rel_60",
    "vix_chg_5",
    "mkt_ret_5",
    "mkt_ret_20",
    "mkt_vol_20",
    "rate",
    "rate_chg_20",
)

SECTOR_COLUMNS: tuple[str, ...] = (
    "sec_ret_5",
    "sec_ret_20",
    "sec_dist_ma50",
)

CONTEXT_COLUMNS: tuple[str, ...] = MARKET_COLUMNS + SECTOR_COLUMNS

SECTOR_ETFS: dict[str, str] = {
    "AAPL": "XLK",
    "MSFT": "XLK",
    "NVDA": "XLK",
    "INTC": "XLK",
    "GOOGL": "XLC",
    "META": "XLC",
    "NFLX": "XLC",
    "DIS": "XLC",
    "AMZN": "XLY",
    "TSLA": "XLY",
    "JPM": "XLF",
    "BAC": "XLF",
    "V": "XLF",
    "XOM": "XLE",
    "CVX": "XLE",
    "JNJ": "XLV",
    "PG": "XLP",
    "KO": "XLP",
    "WMT": "XLP",
}
"""Which SPDR sector ETF stands in for which ticker.

A static GICS-style classification, not a fitted quantity — it carries no
information from the future. Tickers not listed here fall back to the broad
market, which makes their sector columns a duplicate of the market columns
rather than a guess.
"""


def sector_etf_for(symbol: str) -> str:
    """The sector ETF to use for `symbol`, defaulting to the broad market."""
    return SECTOR_ETFS.get(symbol.upper(), MARKET)


def _series(frame: pd.DataFrame | None, column: str = "Close") -> pd.Series | None:
    if frame is None or frame.empty:
        return None
    return frame[column].astype(float)


def download_market_context(
    start_date: datetime.date, end_date: datetime.date
) -> pd.DataFrame | None:
    """Volatility, broad market and short rate, as one row per market day.

    Returns None if any of the three downloads fails. A partial frame would be
    worse than none: a model would happily train on it and the missing column
    would read as a structural absence rather than a failed download.
    """
    vix = _series(download_stock_data(VIX, start_date, end_date))
    market = _series(download_stock_data(MARKET, start_date, end_date))
    rate = _series(download_stock_data(SHORT_RATE, start_date, end_date))

    if vix is None or market is None or rate is None:
        logger.warning("Market context unavailable; proceeding without it.")
        return None

    feats = pd.DataFrame(index=vix.index)
    feats["vix_level"] = vix
    feats["vix_rel_60"] = vix / vix.rolling(60).mean() - 1.0
    feats["vix_chg_5"] = log_returns(vix, 5)

    market_daily = log_returns(market, 1)
    feats["mkt_ret_5"] = log_returns(market, 5)
    feats["mkt_ret_20"] = log_returns(market, 20)
    feats["mkt_vol_20"] = market_daily.rolling(20).std()

    feats["rate"] = rate
    feats["rate_chg_20"] = rate.diff(20)

    # Every one of the three series trades on its own calendar. Intersecting
    # keeps a row only where all three were actually quoted, so no column is
    # silently stale.
    return feats.dropna()


def download_sector_context(
    symbol: str, start_date: datetime.date, end_date: datetime.date
) -> pd.DataFrame | None:
    """The sector ETF for `symbol`, as trailing-only columns."""
    close = _series(download_stock_data(sector_etf_for(symbol), start_date, end_date))
    if close is None:
        return None

    feats = pd.DataFrame(index=close.index)
    feats["sec_ret_5"] = log_returns(close, 5)
    feats["sec_ret_20"] = log_returns(close, 20)
    feats["sec_dist_ma50"] = close / close.rolling(50).mean() - 1.0
    return feats.dropna()


def build_context(
    symbol: str,
    start_date: datetime.date,
    end_date: datetime.date,
    market: pd.DataFrame | None = None,
) -> pd.DataFrame | None:
    """Full context frame for one ticker: market-wide plus its own sector.

    `market` may be passed in already downloaded so that a run over many
    tickers downloads the shared part once.
    """
    market = (
        market if market is not None else download_market_context(start_date, end_date)
    )
    if market is None:
        return None

    sector = download_sector_context(symbol, start_date, end_date)
    if sector is None:
        return None

    return market.join(sector, how="inner")[list(CONTEXT_COLUMNS)]


def align_context(context: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Put context onto a stock's own calendar without looking forward.

    A forward-fill is the whole point. The stock's calendar and the market's
    calendar disagree — half-days, index additions, a halt — and the naive
    join would either drop those rows or, worse, be filled later by something
    that reaches ahead. Filling from the most recent *earlier* market day is
    the only value that was knowable at the close being described.
    """
    ordered = context.sort_index()
    return ordered.reindex(index, method="ffill")
