"""Unit tests for every Plotly chart builder."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from src.constants import (
    COLOR_ACTUAL,
    COLOR_MA,
    COLOR_NAIVE,
    COLOR_PREDICTED,
)
from src.core.baselines import MOVING_AVERAGE, NAIVE
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


def make_result():
    """A minimal AnalysisResult-shaped object for result-driven charts."""
    from types import SimpleNamespace

    data = make_data(400)
    close = data["Close"]
    test_start = 300
    test_predictions = pd.DataFrame(
        {
            "actual": close.iloc[test_start:].values,
            "predicted": close.iloc[test_start:].values * 0.99,
        }
    )
    future = pd.DataFrame(
        {
            "date": pd.bdate_range(close.index[-1], periods=5)[1:],
            "predicted_close": [close.iloc[-1] * 1.01] * 4,
        }
    )
    baselines_frame = pd.DataFrame(
        {
            NAIVE: close.iloc[test_start:].shift(1).bfill().values,
            MOVING_AVERAGE: close.iloc[test_start:].rolling(3).mean().bfill().values,
        },
        index=close.index[test_start:],
    )
    return SimpleNamespace(
        symbol="TEST",
        close=close,
        test_start_index=test_start,
        test_predictions=test_predictions,
        future=future,
        lstm_metrics={"rmse": 2.5, "mae": 2.0, "mape": 0.02, "r2": 0.9, "mse": 6.25},
        baselines={NAIVE: {"rmse": 3.0}, MOVING_AVERAGE: {"rmse": 4.0}},
        baselines_frame=baselines_frame,
    )


@pytest.fixture()
def data() -> pd.DataFrame:
    return make_data(100)


@pytest.fixture()
def returns(data: pd.DataFrame) -> pd.Series:
    return data["Close"].pct_change().dropna()


@pytest.fixture()
def result():
    return make_result()


def test_volume_chart_aggregates_long_history():
    fig = charts.plot_volume_analysis(make_data(400))
    assert len(fig.data[0].x) < 100  # weekly bars, not 400 daily ones
    assert "weekly" in fig.layout.title.text


def test_volume_chart_stays_daily_on_short_history():
    fig = charts.plot_volume_analysis(make_data(100))
    assert len(fig.data[0].x) == 100
    assert "daily" in fig.layout.title.text


def test_candlestick_aggregates_long_history():
    fig = charts.plot_candlestick(make_data(400))
    assert len(fig.data[0].x) < 100
    assert "weekly" in fig.layout.title.text


def test_candlestick_stays_daily_on_short_history():
    fig = charts.plot_candlestick(make_data(100))
    assert len(fig.data[0].x) == 100
    assert "weekly" not in fig.layout.title.text


def test_dark_theme_styling(data):
    fig = charts.plot_cumulative_returns(data["Close"])
    assert fig.layout.font.color == "#F0F2F6"
    predictions = pd.DataFrame(
        {"actual": data["Close"].values, "predicted": data["Close"].values}
    )
    fig = charts.plot_test_predictions(predictions, data["Close"], 50, 30)
    actual = next(tr for tr in fig.data if tr.name == "Actual (test)")
    assert actual.line.color == COLOR_ACTUAL


# --------------------------------------------------------------------------- #
# Smoke tests: every builder returns a figure with the expected traces
# --------------------------------------------------------------------------- #
def test_plot_loss_history(data):
    class _History:
        history = {
            "loss": [1.0, 0.5],
            "val_loss": [1.2, 0.6],
        }

    fig = charts.plot_loss_history(_History())
    assert [t.name for t in fig.data] == ["Training loss (MAE)", "Validation loss"]


def test_plot_loss_history_empty(data):
    fig = charts.plot_loss_history(None)
    assert len(fig.data) == 0


def test_plot_test_predictions_trace_names(data):
    predictions = pd.DataFrame(
        {"actual": data["Close"].values[50:], "predicted": data["Close"].values[50:]}
    )
    fig = charts.plot_test_predictions(predictions, data["Close"], 50, 30)
    assert [t.name for t in fig.data] == [
        "Training history",
        "Actual (test)",
        "LSTM prediction",
    ]


def test_plot_forecast_trace_names(result):
    fig = charts.plot_forecast(result)
    assert [t.name for t in fig.data] == [
        "Close price",
        "LSTM on test window",
        f"Forecast ({len(result.future)} business days)",
    ]


def test_plot_baseline_comparison_trace_names(result):
    fig = charts.plot_baseline_comparison(result)
    assert [t.name for t in fig.data] == [
        "Actual (test)",
        "LSTM",
        NAIVE,
        MOVING_AVERAGE,
    ]


def test_plot_metric_bars_colors_follow_methods(result):
    fig = charts.plot_metric_bars(result)
    bars = fig.data[0]
    color_by_method = dict(zip(bars.y, bars.marker.color))
    assert color_by_method["LSTM"] == COLOR_PREDICTED
    assert color_by_method[NAIVE] == COLOR_NAIVE
    assert color_by_method[MOVING_AVERAGE] == COLOR_MA


def test_plot_metric_bars_sorted_by_rmse(result):
    fig = charts.plot_metric_bars(result)
    assert list(fig.data[0].y) == ["LSTM", NAIVE, MOVING_AVERAGE]


def test_plot_cumulative_returns_starts_at_one(data):
    fig = charts.plot_cumulative_returns(data["Close"])
    assert fig.data[0].y[0] == pytest.approx(1.0)


def test_plot_return_histogram(returns):
    fig = charts.plot_return_histogram(returns)
    assert len(fig.data) == 2
    assert fig.data[1].name == "KDE"


def test_plot_qq(returns):
    fig = charts.plot_qq(returns)
    assert [t.name for t in fig.data] == ["Observed", "Normal line"]


def test_plot_rolling_volatility(returns):
    fig = charts.plot_rolling_volatility(returns)
    assert len(fig.data) == 1
    assert fig.data[0].y is not None and len(fig.data[0].y) == len(returns)


def test_plot_acf(returns):
    fig = charts.plot_acf(returns)
    assert fig.data[0].name == "Autocorrelation"


def test_plot_weekday_effects(data):
    from src.core import analytics

    table = analytics.weekday_effects(data["Close"].pct_change().dropna())
    fig = charts.plot_weekday_effects(table)
    assert len(fig.data[0].x) == 5


def test_plot_monthly_heatmap(data):
    from src.core import analytics

    matrix = analytics.monthly_returns_matrix(data["Close"])
    fig = charts.plot_monthly_heatmap(matrix)
    assert len(fig.data[0].z) == matrix.shape[0]


def test_plot_underwater(data):
    fig = charts.plot_underwater(data["Close"])
    assert fig.data[0].y is not None


def test_plot_price_with_indicators(data):
    from src.core import analytics

    crosses = analytics.crossover_dates(
        analytics.sma(data["Close"], 20), analytics.sma(data["Close"], 200)
    )
    fig = charts.plot_price_with_indicators(data["Close"], crosses)
    assert [t.name for t in fig.data] == [
        "Close",
        "SMA 20",
        "SMA 50",
        "SMA 200",
        "EMA 50",
    ]


def test_plot_price_with_indicators_no_crosses(data):
    fig = charts.plot_price_with_indicators(data["Close"])
    assert len(fig.data) == 5


def test_plot_rsi(data):
    fig = charts.plot_rsi(data["Close"])
    assert fig.data[0].name == "RSI 14"
    assert len(fig.data) == 1  # hlines are layout, not traces


def test_plot_macd(data):
    fig = charts.plot_macd(data["Close"])
    assert [t.name for t in fig.data] == ["Histogram", "MACD", "Signal"]


def test_plot_bollinger(data):
    fig = charts.plot_bollinger(data["Close"])
    assert [t.name for t in fig.data] == ["Close", "Upper", "Lower"]


def test_plot_volume_return_scatter(data):
    fig = charts.plot_volume_return_scatter(data)
    assert fig.data[0].name == "Day"
    assert fig.data[0].hovertemplate is not None


def test_plot_normalized_prices(data):
    series = {"A": data["Close"], "B": data["Close"] * 2}
    fig = charts.plot_normalized_prices(series)
    assert [t.name for t in fig.data] == ["A", "B"]


def test_plot_cumulative_comparison(data):
    series = {"A": data["Close"], "B": data["Close"] * 2}
    fig = charts.plot_cumulative_comparison(series)
    assert [t.name for t in fig.data] == ["A", "B"]


def test_plot_correlation_heatmap(data):
    corr = data[["Open", "Close", "Volume"]].corr()
    fig = charts.plot_correlation_heatmap(corr)
    assert len(fig.data[0].z) == 3


def test_plot_drawdown_comparison(data):
    series = {"A": data["Close"], "B": data["Close"] * 2}
    fig = charts.plot_drawdown_comparison(series)
    assert [t.name for t in fig.data] == ["A", "B"]


def test_plot_risk_return_scatter():
    table = pd.DataFrame(
        {
            "Ann. volatility": [0.2, 0.3],
            "CAGR": [0.1, 0.15],
            "Sharpe": [0.5, 0.5],
            "Max drawdown": [-0.2, -0.3],
        },
        index=["AAPL", "MSFT"],
    )
    fig = charts.plot_risk_return_scatter(table)
    assert len(fig.data) == 1


def _chart_cases(data: pd.DataFrame, returns: pd.Series, result):
    """(builder, args) pairs exercising every public chart builder."""
    from src.core import analytics

    close = data["Close"]
    return [
        (charts.plot_candlestick, (data,)),
        (charts.plot_loss_history, (None,)),
        (charts.plot_test_predictions, (result.test_predictions, close, 300, 30)),
        (charts.plot_forecast, (result,)),
        (charts.plot_baseline_comparison, (result,)),
        (charts.plot_metric_bars, (result,)),
        (charts.plot_cumulative_returns, (close,)),
        (charts.plot_return_histogram, (returns,)),
        (charts.plot_qq, (returns,)),
        (charts.plot_rolling_volatility, (returns,)),
        (charts.plot_acf, (returns,)),
        (charts.plot_weekday_effects, (analytics.weekday_effects(returns),)),
        (charts.plot_monthly_heatmap, (analytics.monthly_returns_matrix(close),)),
        (charts.plot_underwater, (close,)),
        (charts.plot_price_with_indicators, (close,)),
        (charts.plot_rsi, (close,)),
        (charts.plot_macd, (close,)),
        (charts.plot_bollinger, (close,)),
        (charts.plot_volume_analysis, (data,)),
        (charts.plot_volume_return_scatter, (data,)),
        (charts.plot_normalized_prices, ({"A": close},)),
        (charts.plot_cumulative_comparison, ({"A": close},)),
        (charts.plot_correlation_heatmap, (data[["Open", "Close"]].corr(),)),
        (charts.plot_drawdown_comparison, ({"A": close},)),
        (charts.plot_risk_return_scatter, (_risk_table(),)),
    ]


def _risk_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Ann. volatility": [0.2],
            "CAGR": [0.1],
            "Sharpe": [0.5],
            "Max drawdown": [-0.2],
        },
        index=["AAPL"],
    )


@pytest.mark.parametrize("builder_idx", range(25), ids=lambda i: f"builder_{i}")
def test_every_builder_returns_figure(data, returns, result, builder_idx):
    builder, args = _chart_cases(data, returns, result)[builder_idx]
    fig = builder(*args)
    assert isinstance(fig, go.Figure)
