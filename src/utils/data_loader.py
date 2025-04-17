import streamlit as st
import pandas as pd
from datetime import date
import yfinance as yf
import logging

@st.cache_data # Cache data downloads
def download_stock_data(stock_symbol: str, start_date: date, end_date: date) -> pd.DataFrame:
    """Downloads stock data from Yahoo Finance."""
    try:
        stock_data = yf.download(stock_symbol, start=start_date, end=end_date)
        if stock_data.empty:
            st.error(f"No data found for symbol {stock_symbol}. Please check the symbol and date range.")
            return pd.DataFrame()
        logging.info(f"Successfully downloaded data for {stock_symbol}")
        return stock_data
    except Exception as e:
        st.error(f"Failed to download data for {stock_symbol}: {e}")
        logging.error(f"Failed to download data for {stock_symbol}: {e}")
        return pd.DataFrame()