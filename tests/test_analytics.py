"""Unit tests for the analytics module."""

import numpy as np
import pandas as pd
import pytest

from src.core import analytics

DAYS = pd.bdate_range("2024-01-01", periods=400)


@pytest.fixture()
def close() -> pd.Series:
    rng = np.random.default_rng(7)
    prices = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.02, len(DAYS))))
    return pd.Series(prices, index=DAYS)


def test_daily_returns_shape_and_positive_ratio(close):
    returns = analytics.daily_returns(close)
    assert len(returns) == len(close) - 1
    assert 0.0 <= analytics.positive_day_ratio(returns) <= 1.0


def test_cumulative_return_matches_manual(close):
    assert analytics.cumulative_return(close) == pytest.approx(
        close.iloc[-1] / close.iloc[0] - 1
    )


def test_sharpe_of_constant_returns_is_zero():
    flat = pd.Series(0.05, index=DAYS[:100])
    assert analytics.sharpe_ratio(flat) == 0.0


def test_max_drawdown_non_positive(close):
    assert analytics.max_drawdown(close) <= 0.0


def test_drawdown_series_bounds(close):
    dd = analytics.drawdown_series(close)
    assert dd.max() <= 1e-12
    assert (dd >= -1.0 - 1e-12).all()


def test_drawdown_events_structure(close):
    events = analytics.drawdown_events(close, min_depth=0.02)
    if len(events):
        assert {"Start", "Trough", "End", "Depth", "Duration (days)"} <= set(
            events.columns
        )
        assert (events["Depth"] <= -0.02).all()


def test_period_returns_has_all_keys(close):
    pr = analytics.period_returns(close)
    assert set(pr) == {"1M", "3M", "6M", "1Y", "YTD"}


def test_position_in_52w_range_bounds(close):
    assert 0.0 <= analytics.position_in_52w_range(close) <= 1.0


def test_value_at_risk_and_cvar(close):
    returns = analytics.daily_returns(close)
    var = analytics.value_at_risk(returns, alpha=0.95)
    cvar = analytics.conditional_var(returns, alpha=0.95)
    assert var < 0
    assert cvar <= var


def test_adf_rejects_random_walk(close):
    summary = analytics.adf_summary(close)
    # A random walk should NOT be stationary at 5%
    assert summary["stationary"] == False


def test_adf_accepts_stationary_series():
    stationary = pd.Series(0.5 + 0.1 * np.sin(np.arange(300)), index=DAYS[:300])
    assert analytics.adf_summary(stationary)["stationary"] is True


def test_weekday_effects_has_five_rows(close):
    table = analytics.weekday_effects(analytics.daily_returns(close))
    assert list(table["weekday"]) == ["Mon", "Tue", "Wed", "Thu", "Fri"]
    assert set(table.columns) == {"weekday", "mean", "hit_rate", "count"}


def test_weekday_effects_with_missing_weekday():
    idx = pd.bdate_range("2024-01-01", periods=200)
    idx = idx[pd.Series(idx).map(lambda ts: ts.dayofweek) != 3]  # no Thursdays at all
    returns = pd.Series(np.linspace(0.001, 0.01, len(idx)), index=idx)
    table = analytics.weekday_effects(returns)
    assert list(table["weekday"]) == ["Mon", "Tue", "Wed", "Thu", "Fri"]
    assert table["count"].iloc[3] == 0
    assert table["mean"].isna().iloc[3]


def test_monthly_matrix_shape(close):
    matrix = analytics.monthly_returns_matrix(close)
    assert matrix.shape[1] == 12
    assert list(matrix.columns)[0] == "Jan"


def test_acf_first_lag_is_one(close):
    acf = analytics.acf(analytics.daily_returns(close), nlags=5)
    assert acf.iloc[0] == pytest.approx(1.0)
    assert len(acf) == 6


def test_rsi_bounds(close):
    r = analytics.rsi(close)
    r = r.dropna()
    assert ((r >= 0) & (r <= 100)).all()


def test_rsi_extreme_direction():
    up = pd.Series(np.linspace(100, 200, 100), index=DAYS[:100])
    down = pd.Series(np.linspace(200, 100, 100), index=DAYS[:100])
    assert analytics.rsi(up).dropna().iloc[-1] > 90
    assert analytics.rsi(down).dropna().iloc[-1] < 10


def test_macd_columns(close):
    m = analytics.macd(close)
    assert set(m.columns) == {"MACD", "Signal", "Histogram"}


def test_crossover_detects_golden_cross():
    idx = DAYS[:100]
    fast = pd.Series(np.linspace(0, 10, 100), index=idx)
    slow = pd.Series(np.linspace(10, 0, 100), index=idx)
    crosses = analytics.crossover_dates(fast, slow)
    assert len(crosses) == 1
    assert crosses.iloc[0]["Direction"] == "Golden"


def test_latest_signals_keys(close):
    signals = analytics.latest_signals(close)
    assert len(signals) >= 3


def test_normalize_series_starts_at_base(close):
    norm = analytics.normalize_series(close, base=100)
    assert norm.iloc[0] == pytest.approx(100)


def test_comparison_frame_keys(close):
    frame = analytics.comparison_frame(close)
    assert set(frame) == {
        "Total return",
        "CAGR",
        "Ann. volatility",
        "Sharpe",
        "Sortino",
        "Max drawdown",
        "VaR 95% (daily)",
        "CVaR 95% (daily)",
        "Positive days",
    }
