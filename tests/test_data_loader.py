"""Unit tests for the yfinance download wrapper (network is mocked)."""

import datetime

import numpy as np
import pandas as pd
import pytest

import src.core.data_loader as data_loader

START = datetime.date(2024, 1, 1)
END = datetime.date(2024, 6, 1)


def _flat_frame(n=30) -> pd.DataFrame:
    idx = pd.bdate_range(START, periods=n)
    return pd.DataFrame(
        {
            "Open": 100.0,
            "High": 101.0,
            "Low": 99.0,
            "Close": 100.5,
            "Volume": 1_000_000,
        },
        index=idx,
    )


def test_flattens_multiindex_columns(monkeypatch):
    frame = _flat_frame()
    frame.columns = pd.MultiIndex.from_product(
        [["Open", "High", "Low", "Close", "Volume"], ["TEST"]]
    )
    monkeypatch.setattr(data_loader.yf, "download", lambda *a, **k: frame)

    result = data_loader.download_stock_data("TEST", START, END)
    assert result is not None
    assert list(result.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert isinstance(result.index, pd.DatetimeIndex)


def test_returns_none_on_empty_download(monkeypatch):
    monkeypatch.setattr(data_loader.yf, "download", lambda *a, **k: pd.DataFrame())

    assert data_loader.download_stock_data("TEST", START, END) is None


def test_returns_none_on_exception(monkeypatch):
    def boom(*a, **k):
        raise ConnectionError("rate limited")

    monkeypatch.setattr(data_loader.yf, "download", boom)

    assert data_loader.download_stock_data("TEST", START, END) is None


def test_drops_rows_with_missing_prices(monkeypatch):
    frame = _flat_frame()
    frame.loc[frame.index[3], "Close"] = None
    monkeypatch.setattr(data_loader.yf, "download", lambda *a, **k: frame)

    result = data_loader.download_stock_data("TEST", START, END)
    assert result is not None
    assert len(result) == len(frame) - 1
    assert result["Close"].notna().all()


def test_returns_none_when_all_rows_dropped(monkeypatch):
    frame = _flat_frame()
    frame.loc[:, "Close"] = np.nan
    monkeypatch.setattr(data_loader.yf, "download", lambda *a, **k: frame)

    assert data_loader.download_stock_data("TEST", START, END) is None
