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


def test_drawdown_events_captures_multi_day_drawdown():
    """A clear 10-day dip must be reported as ONE event with the right depth."""
    values = np.concatenate(
        [np.linspace(100, 120, 15), np.linspace(120, 60, 10), np.linspace(60, 125, 15)]
    )
    close = pd.Series(values, index=pd.bdate_range("2024-01-01", periods=40))
    events = analytics.drawdown_events(close, min_depth=0.05)

    assert len(events) == 1
    row = events.iloc[0]
    assert row["Depth"] == pytest.approx(-0.5, rel=1e-2)
    # Calendar days between the start and the recovery back above the peak
    assert 20 <= row["Duration (days)"] <= 35
    assert row["Trough"] > row["Start"]


def test_drawdown_events_splits_separate_episodes():
    """Two distinct dips must produce two events, most recent first."""
    values = np.concatenate(
        [
            np.linspace(100, 60, 10),
            np.linspace(60, 110, 10),
            np.linspace(110, 70, 10),
            np.linspace(70, 100, 10),
        ]
    )
    close = pd.Series(values, index=pd.bdate_range("2024-01-01", periods=40))
    events = analytics.drawdown_events(close, min_depth=0.05)

    assert len(events) == 2
    assert events.iloc[0]["Start"] > events.iloc[1]["Start"]
    assert events["Depth"].le(-0.05).all()


def test_drawdown_events_empty_when_no_drawdown():
    monotonic = pd.Series(np.linspace(100, 200, 50), index=DAYS[:50])
    assert analytics.drawdown_events(monotonic, min_depth=0.05).empty


def test_annualized_volatility_matches_manual(close):
    returns = analytics.daily_returns(close)
    assert analytics.annualized_volatility(returns) == pytest.approx(
        returns.std(ddof=1) * np.sqrt(252)
    )


def test_sortino_ignores_upside():
    idx = DAYS[:100]
    up = pd.Series(np.linspace(100, 200, 100), index=idx)
    down = pd.Series(np.linspace(200, 100, 100), index=idx)
    returns_up = analytics.daily_returns(up)
    returns_down = analytics.daily_returns(down)
    assert analytics.sortino_ratio(returns_up) > analytics.sortino_ratio(returns_down)


def test_conditional_var_below_var(close):
    returns = analytics.daily_returns(close)
    assert analytics.conditional_var(returns) <= analytics.value_at_risk(returns)


def test_rolling_volatility_length(close):
    rv = analytics.rolling_volatility(analytics.daily_returns(close), window=20)
    assert len(rv) == len(close) - 1
    assert rv.iloc[:19].isna().all()
    assert rv.iloc[19:].notna().all()


def test_ema_follows_ewm(close):
    np.testing.assert_allclose(
        analytics.ema(close, span=50), close.ewm(span=50, adjust=False).mean()
    )


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
