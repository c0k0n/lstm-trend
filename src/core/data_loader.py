"""Pure data-loading logic (no Streamlit imports)."""

import datetime
from typing import Optional

import pandas as pd
import yfinance as yf


def download_stock_data(
    ticker: str, start_date: datetime.date, end_date: datetime.date
) -> Optional[pd.DataFrame]:
    """Download daily OHLCV data for a ticker and return a flat-column DataFrame.

    yfinance occasionally returns MultiIndex columns (e.g. ("Close", "AAPL"));
    this normalises them to plain "Open", "High", "Low", "Close", "Volume"
    columns so the rest of the code never has to worry about it.

    Returns None when the download fails or contains no usable rows.
    """
    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if data.empty:
            return None

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        data = data[["Open", "High", "Low", "Close", "Volume"]]
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index)
        data = data.dropna()

        return data if not data.empty else None
    except Exception:
        return None
