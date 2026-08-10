"""Pure data-loading logic (no Streamlit imports)."""

import datetime
import logging

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


def download_stock_data(
    ticker: str, start_date: datetime.date, end_date: datetime.date
) -> pd.DataFrame | None:
    """Download daily OHLCV data for a ticker and return a flat-column DataFrame.

    yfinance occasionally returns MultiIndex columns (e.g. ("Close", "AAPL"));
    this normalises them to plain "Open", "High", "Low", "Close", "Volume"
    columns so the rest of the code never has to worry about it.

    Returns None when the download fails or contains no usable rows.
    """
    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if data is None or data.empty:
            return None

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        data = data[["Open", "High", "Low", "Close", "Volume"]]
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index)
        data = data.dropna()

        return data if not data.empty else None
    except Exception as exc:
        logger.warning(
            "Failed to download %s (%s -> %s): %s", ticker, start_date, end_date, exc
        )
        return None
