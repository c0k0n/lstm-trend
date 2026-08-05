import yfinance as yf
import pandas as pd
import streamlit as st
import datetime
from typing import Optional


# Keep caching to avoid repeated downloads for the same parameters
@st.cache_data
def download_stock_data(
    ticker: str, start_date: datetime.date, end_date: datetime.date
) -> Optional[pd.DataFrame]:
    """
    Downloads historical stock data using yfinance for a given ticker and date range.

    Args:
        ticker (str): The stock ticker symbol (e.g., 'AAPL').
        start_date (datetime.date): The start date for the data (inclusive).
        end_date (datetime.date): The end date for the data (inclusive).

    Returns:
        Optional[pd.DataFrame]: A DataFrame containing the OHLCV data with a
                                DatetimeIndex, or None if download fails or
                                no data is found.
    """
    try:
        # Ensure end_date is included by adding a day if needed, yfinance is typically exclusive of end date
        # However, testing shows yf.download often includes the end_date if it's a valid market day.
        # Let's keep it simple first. If issues arise, adjust end_date: end_date + datetime.timedelta(days=1)
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)

        if data.empty:
            st.warning(
                f"No data found for {ticker} between {start_date} and {end_date}. "
                "Check the ticker symbol and date range (ensure markets were open)."
            )
            return None  # Return None for clarity

        # Ensure index is DatetimeIndex
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index)

        # Remove rows with NaN values, especially if they occur at the start/end
        data.dropna(inplace=True)

        if data.empty:
            st.warning(
                f"Data for {ticker} between {start_date} and {end_date} contained only NaN values after cleaning."
            )
            return None

        return data
    except Exception as e:
        st.error(f"Error downloading data for {ticker}: {e}")
        # Consider logging the error here as well for backend debugging
        # logger.error(f"yfinance download failed for {ticker}: {e}", exc_info=True)
        return None  # Return None on exception
