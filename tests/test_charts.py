"""Unit tests for chart aggregation behaviour on long histories."""

import numpy as np
import pandas as pd

from src.constants import COLOR_ACTUAL
from src.ui import charts

DAYS = pd.bdate_range("2020-01-01", periods=400)


def make_data(n=400):
    rng = np.random.default_rng(3)
    prices = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, n)))
    return pd.DataFrame(
        {
            "Open": prices * 0.999,
            "High": prices * 1.01,
            "Low": prices * 0.99,
            "Close": prices,
            "Volume": rng.integers(1_000_000, 5_000_000, n).astype("int64"),
        },
        index=DAYS[:n],
    )


def test_volume_chart_aggregates_long_history():
    data = make_data(400)
    fig = charts.plot_volume_analysis(data)
    bars = fig.data[0].x
    assert len(bars) < 100  # weekly bars, not 400 daily ones
    assert "weekly" in fig.layout.title.text


def test_volume_chart_stays_daily_on_short_history():
    data = make_data(100)
    fig = charts.plot_volume_analysis(data)
    assert len(fig.data[0].x) == 100
    assert "daily" in fig.layout.title.text


def test_candlestick_aggregates_long_history():
    data = make_data(400)
    fig = charts.plot_candlestick(data)
    assert len(fig.data[0].x) < 100
    assert "weekly" in fig.layout.title.text


def test_candlestick_stays_daily_on_short_history():
    data = make_data(100)
    fig = charts.plot_candlestick(data)
    assert len(fig.data[0].x) == 100
    assert "weekly" not in fig.layout.title.text


def test_dark_theme_styling():
    data = make_data(100)
    fig = charts.plot_close(data["Close"])
    assert fig.layout.font.color == "#F0F2F6"
    predictions = pd.DataFrame(
        {"actual": data["Close"].values, "predicted": data["Close"].values}
    )
    fig = charts.plot_test_predictions(predictions, data["Close"], 50, 30)
    actual = next(tr for tr in fig.data if tr.name == "Actual (test)")
    assert actual.line.color == COLOR_ACTUAL
