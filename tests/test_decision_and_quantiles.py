"""Tests for the decision metrics and the quantile model.

These guard the failure modes that matter here, which are all *silent* ones: a
metric that reports noise as skill, a cost model that flatters a strategy, a
quantile grid that is missing its own median, and a loss that punishes the
wrong side of a tail.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.core import decision as dec
from src.core import quantile_model as qm
from src.core.skill import BASELINE


# --- decision: direction --------------------------------------------------


def test_a_perfect_forecast_scores_perfectly():
    call = np.array([1.0, -1.0, 1.0, -1.0, 1.0])
    move = np.array([0.02, -0.01, 0.03, -0.02, 0.01])
    result = dec.directional_accuracy(call, move)
    assert result["accuracy"] == pytest.approx(1.0)
    assert result["beats_coin_flip"] is True


def test_a_coin_flip_is_not_reported_as_skill():
    """The bug this project has already made twice: calling noise an edge."""
    rng = np.random.default_rng(0)
    call = rng.choice([-1.0, 1.0], 3000)
    move = rng.choice([-1.0, 1.0], 3000)
    result = dec.directional_accuracy(call, move)
    assert result["accuracy"] < 0.55
    assert result["beats_coin_flip"] is False


def test_an_unchanging_day_is_a_miss_not_a_half_win():
    """Zero-change days are genuinely not predicted. Generosity here is a lie."""
    call = np.array([1.0, 1.0, 1.0])
    move = np.array([0.01, 0.0, 0.02])
    result = dec.directional_accuracy(call, move)
    assert result["accuracy"] == pytest.approx(2 / 3)


def test_non_finite_inputs_are_dropped_not_propagated():
    call = np.array([1.0, np.nan, -1.0, np.inf])
    move = np.array([0.01, 0.02, -0.01, 0.03])
    result = dec.directional_accuracy(call, move)
    assert result["n"] == 2


def test_empty_input_scores_zero_rather_than_raising():
    result = dec.directional_accuracy(np.array([]), np.array([]))
    assert result["n"] == 0
    assert result["accuracy"] == 0.0


# --- decision: costs and utility ------------------------------------------


def test_costs_turn_a_coin_flip_into_a_loss():
    rng = np.random.default_rng(1)
    move = rng.normal(0.0005, 0.01, 500)
    call = rng.choice([-1.0, 1.0], 500)
    free = dec.signal_utility(call, move, cost_bps=0.0)
    paid = dec.signal_utility(call, move, cost_bps=10.0)
    assert paid["net"] < free["net"]
    assert paid["net"] < 0.0, "noise must not survive costs"


def test_an_oracle_makes_money_after_costs():
    rng = np.random.default_rng(2)
    move = rng.normal(0.001, 0.01, 500)
    oracle = np.sign(move)
    result = dec.signal_utility(oracle, move, cost_bps=10.0)
    assert result["net"] > 0.0
    assert dec.sharpe_of_signal(oracle, move) > 0.0


def test_sharpe_penalises_lumpy_gains():
    """Winning often and small, losing rarely and large, must score lower.

    Both arms have the same *expected* daily move and the same win rate; only
    the size of the losing days differs, so the gentle arm has a higher ratio
    of mean to spread.

    Two earlier versions of this test were vacuous and are worth recording. One
    compared a series whose gains were entirely consumed by the 10 bps cost, so
    both arms scored zero. The other used a perfectly uniform series — every
    winning day exactly +3% — whose standard deviation is zero, so the sd guard
    returned 0.0 for the good arm and the comparison proved nothing. Real
    forecasts vary; a constant is not a harder case, it is a degenerate one.
    """
    rng = np.random.default_rng(12)
    up = 0.030 + rng.normal(0, 0.004, 95)
    gentle = np.concatenate([up, np.full(5, -0.030)])
    brutal = np.concatenate([up, np.full(5, -0.900)])

    gentle_sharpe = dec.sharpe_of_signal(np.sign(gentle), gentle)
    brutal_sharpe = dec.sharpe_of_signal(np.sign(brutal), brutal)

    assert gentle_sharpe > 0.0, "a steady, profitable arm must score well"
    assert brutal_sharpe < gentle_sharpe, (
        "a heavier tail at the same win rate must not score higher"
    )


def test_costs_can_consume_an_edge_entirely():
    """A strategy that wins 95% of days at +0.1% is worthless at 10 bps a trade.

    This is the number that stops a 52%-accurate forecaster from looking like a
    money machine, so it is worth pinning explicitly rather than leaving to a
    comment. The winning days are jittered so the series is not degenerate —
    a perfectly uniform series has zero spread and the Sharpe guard returns
    0.0 regardless, which would make the assertion meaningless.
    """
    rng = np.random.default_rng(13)
    thin = np.concatenate([0.0008 + rng.normal(0, 0.00015, 95), np.full(5, -0.0008)])
    # At zero cost the edge is thin but positive; at 10 bps it is gone.
    assert dec.signal_utility(np.sign(thin), thin, cost_bps=0.0)["net"] > 0.0
    assert dec.signal_utility(np.sign(thin), thin, cost_bps=10.0)["net"] < 0.0
    # And the cost parameter is monotone: more friction, worse score.
    assert dec.sharpe_of_signal(np.sign(thin), thin, cost_bps=0.0) > (
        dec.sharpe_of_signal(np.sign(thin), thin, cost_bps=50.0)
    )


def test_higher_costs_always_reduce_a_strategy_score():
    """The cost parameter has to be interrogable, so monotonicity is the test."""
    rng = np.random.default_rng(11)
    move = rng.normal(0.002, 0.01, 300)
    call = np.sign(move)
    scores = [
        dec.sharpe_of_signal(call, move, cost_bps=bps) for bps in (0.0, 5.0, 10.0, 25.0)
    ]
    assert scores == sorted(scores, reverse=True), (
        f"Sharpe must fall as costs rise, got {scores}"
    )


# --- decision: bands -------------------------------------------------------


def test_band_coverage_uses_one_shared_mask():
    """A NaN in only one bound must not silently misalign the two."""
    actual = np.array([0.0, 0.0, 0.0, 0.0])
    lower = np.array([-1.0, np.nan, -1.0, -1.0])
    upper = np.array([1.0, 1.0, 1.0, 1.0])
    result = dec.band_calibration(actual, lower, upper)
    # Row 1 is dropped entirely, not scored against a mismatched pair.
    assert result["n"] == 3
    assert result["coverage"] == pytest.approx(1.0)


def test_a_wide_band_containing_everything_reports_full_coverage():
    actual = np.array([0.0, 0.5, -0.5, 2.0])
    result = dec.band_calibration(actual, np.full(4, -3.0), np.full(4, 3.0))
    assert result["coverage"] == pytest.approx(1.0)


def test_a_too_narrow_band_reports_low_coverage():
    # 0.0 is inside [-0.1, 0.1]; +2, -2, +1, -1 are not. So exactly one of five.
    actual = np.array([0.0, 2.0, -2.0, 1.0, -1.0])
    result = dec.band_calibration(actual, np.full(5, -0.1), np.full(5, 0.1))
    assert result["coverage"] == pytest.approx(0.2)


# --- decision: cross-sectional control ------------------------------------


def test_picking_the_best_of_a_random_ranking_beats_equal_weight_by_chance():
    """The control the strategy is measured against must itself be attainable."""
    rng = np.random.default_rng(3)
    scores = dict(zip("ABCDE", rng.normal(size=5), strict=True))
    realised = {k: scores[k] + rng.normal(0, 0.01) for k in scores}
    result = dec.rank_and_go_long_top(scores, realised, k=2)
    # With a perfect score, the top-2 basket beats the equal-weight control.
    assert result["vs_equal_weight"] > 0


def test_top_k_is_winsorised_to_half_the_universe():
    """Asking for the top 10 of 12 names would trivially be everything."""
    scores = {chr(65 + i): float(i) for i in range(4)}
    realised = {k: 0.0 for k in scores}
    result = dec.rank_and_go_long_top(scores, realised, k=99)
    assert result["n"] == 4


# --- quantile model --------------------------------------------------------


def test_the_quantile_grid_contains_exact_medians():
    """The bug that made this module unrunnable.

    `np.arange(0.01, 1.0, 0.05)` produces 0.060000000000000005 and friends, so
    the grid had no 0.5 in it and every median lookup would raise.
    """
    assert 0.5 in qm.ALL_QUANTILES
    assert 0.05 in qm.ALL_QUANTILES
    assert 0.95 in qm.ALL_QUANTILES
    assert 0.01 in qm.ALL_QUANTILES
    assert 0.99 in qm.ALL_QUANTILES
    assert list(qm.ALL_QUANTILES) == sorted(qm.ALL_QUANTILES)
    assert len(set(qm.ALL_QUANTILES)) == len(qm.ALL_QUANTILES)


def test_pinball_is_zero_only_at_the_truth():
    truth = np.zeros((1, 1))
    perfect = np.zeros((1, len(qm.ALL_QUANTILES)))
    assert qm.pinball_loss(truth, perfect) == pytest.approx(0.0)
    off = np.full((1, len(qm.ALL_QUANTILES)), 0.05)
    assert qm.pinball_loss(truth, off) > 0.0


def test_pinball_punishes_over_forecasting_a_low_quantile_harder():
    """At the 5% level, too high is the expensive error.

    A 5th-percentile number is what a risk budget reads to decide how much
    exposure is survivable, so understating the downside is the costly mistake.
    """
    truth = np.zeros((1, 1))
    over = np.zeros((1, len(qm.ALL_QUANTILES)))
    over[0, qm.ALL_QUANTILES.index(0.05)] = 0.10
    under = np.zeros((1, len(qm.ALL_QUANTILES)))
    under[0, qm.ALL_QUANTILES.index(0.05)] = -0.10
    assert qm.pinball_loss(truth, over) > qm.pinball_loss(truth, under)


def test_pinball_rejects_a_wrong_number_of_outputs():
    truth = np.zeros((1, 1))
    with pytest.raises(ValueError, match="quantile outputs"):
        qm.pinball_loss(truth, np.zeros((1, 3)))


def test_sorting_fixes_crossing_and_is_idempotent():
    crossing = np.array([[0.05, -0.02, 0.01, 0.03]])
    ordered = qm.sort_quantiles(crossing)
    assert np.all(np.diff(ordered[0]) >= 0)
    assert np.allclose(qm.sort_quantiles(ordered), ordered)


def test_the_lower_tail_is_allowed_to_be_negative():
    """Clamping at zero would delete the entire downside this model exists for."""
    sweep = np.linspace(-0.05, 0.08, len(qm.ALL_QUANTILES))[None, :]
    _, lower, _ = qm.band_from_quantiles(sweep, 0.05, 0.95)
    assert lower[0] < 0
    prices = qm.quantile_returns_to_prices(100.0, sweep)
    assert prices[0, qm.ALL_QUANTILES.index(0.05)] < 100.0


def test_band_is_ordered_around_the_median():
    sweep = np.linspace(-0.05, 0.08, len(qm.ALL_QUANTILES))[None, :]
    median, lower, upper = qm.band_from_quantiles(sweep, 0.05, 0.95)
    assert lower[0] < median[0] < upper[0]


def test_a_known_distribution_calibrates():
    """The control that makes the calibration score meaningful."""
    from scipy import stats as sps

    rng = np.random.default_rng(4)
    realised = sps.t.rvs(df=4, scale=0.02, size=4000, random_state=1)
    true_levels = sps.t.ppf(np.array(qm.ALL_QUANTILES), df=4, scale=0.02)
    coverage = qm.quantile_coverage(realised, np.tile(true_levels, (4000, 1)))
    assert coverage["mean_abs_error"] < 0.06


def test_a_too_tight_distribution_calibrates_worse():
    from scipy import stats as sps

    realised = sps.t.rvs(df=4, scale=0.02, size=4000, random_state=1)
    true_levels = sps.t.ppf(np.array(qm.ALL_QUANTILES), df=4, scale=0.02)
    good = qm.quantile_coverage(realised, np.tile(true_levels, (4000, 1)))
    bad = qm.quantile_coverage(realised, np.tile(true_levels * 0.5, (4000, 1)))
    assert bad["mean_abs_error"] > good["mean_abs_error"]


def test_window_building_aligns_window_end_to_target_row():
    features = np.arange(500, dtype=float).reshape(500, 1)
    target = np.arange(500, dtype=float)
    x, y = qm.fit_quantile_windows(features, target, 60)
    assert x.shape == (440, 60, 1)
    assert y.shape == (440, 1)
    # Window j ends at row 59+j and predicts row 60+j.
    assert x[0, -1, 0] == 59
    assert y[0, 0] == 60
    assert x[-1, -1, 0] == 498
    assert y[-1, 0] == 499


def test_too_short_an_input_yields_an_empty_result_not_an_error():
    features = np.arange(30, dtype=float).reshape(30, 1)
    x, y = qm.fit_quantile_windows(features, np.arange(30, dtype=float), 60)
    assert x.shape[0] == 0
    assert y.shape[0] == 0


# --- the recursive arm, on synthetic data ---------------------------------


def test_recursive_backtest_scores_both_arms_finitely():
    """The off-by-one that made the direct arm return NaN for every window.

    Row 0's return is NaN by construction; a window reaching back to it
    poisoned the fit. Guarded here with synthetic data so the bug cannot come
    back quietly.
    """
    from src.core import recursive_path as rp

    rng = np.random.default_rng(5)
    n = 700
    index = pd.bdate_range("2021-01-01", periods=n)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    frame = pd.DataFrame(
        {
            "Open": close,
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": rng.integers(1_000_000, 5_000_000, n).astype(float),
        },
        index=index,
    )
    result = rp.recursive_backtest(frame, horizon=3, n_origins=2, epochs=1)
    assert np.isfinite(result[rp.DIRECT].to_numpy(dtype=float)).all()
    assert np.isfinite(result[rp.RECURSIVE].to_numpy(dtype=float)).all()
    assert result[rp.RECURSIVE].notna().all()


# --- the units bug: dollars reported as returns ---------------------------


def _price_frame(price_scale: float, skill: float = 0.5) -> pd.DataFrame:
    """A walk-forward frame where the model recovers `skill` of each move."""
    anchor = np.array([100.0, 200.0, 150.0, 120.0]) * price_scale
    moves = np.array([0.01, -0.02, 0.03, -0.01])
    return pd.DataFrame(
        {
            "origin": pd.bdate_range("2025-01-01", periods=4),
            "target": pd.bdate_range("2025-01-08", periods=4),
            "actual": anchor * (1 + moves),
            BASELINE: anchor,
            "Model": anchor * (1 + moves * skill),
        }
    )


def test_decision_metrics_are_scale_free():
    """A 50x share price must not change a single reported figure.

    The bug this guards: `decision_summary` computed the move as
    `actual - anchor`, which is *dollars*. For a $300 stock that is a number
    like 4.20, and the utility function reported it as a "340% return" and
    annualised it to +85909%. Direction was unaffected because it only reads
    signs, so the error hid behind a plausible-looking percentage column.
    """
    small = dec.decision_summary(_price_frame(1.0), "Model")
    large = dec.decision_summary(_price_frame(50.0), "Model")

    for key in ("direction", "information", "utility", "sharpe"):
        assert small[key] == large[key], f"{key} is not scale-free"


def test_reported_returns_are_plausible_for_a_short_horizon():
    """A five-day move cannot be 340% per window.

    Catches the second half of the same bug: dividing the *dollar* difference
    by the anchor is not a fix, it just rescales the nonsense. The correct
    form is `actual / anchor - 1`, which is ~0.01 for a 1% day.
    """
    summary = dec.decision_summary(_price_frame(1.0), "Model", period_days=5)
    utility = summary["utility"]
    assert abs(utility["net"]) < 0.05, f"net {utility['net']} is not a 5-day return"
    assert abs(utility["annualized"]) < 15, "annualised figure is implausible"


def test_annualisation_respects_the_window_length():
    """A 5-day return must not be annualised as though it were a 1-day one.

    `net * 252` on a 5-day signal reports roughly five times the honest
    figure, and the error grows with how good the signal looks — so the
    better a model appears, the more wrong the number. Compounded
    `(1 + net) ** (252 / period_days) - 1` is the standard form.
    """
    frame = _price_frame(1.0)
    daily = dec.decision_summary(frame, "Model", period_days=1)["utility"]
    five_day = dec.decision_summary(frame, "Model", period_days=5)["utility"]

    assert five_day["net"] == pytest.approx(daily["net"]), (
        "the per-window return does not depend on the horizon"
    )
    # The same return, annualised correctly, is a much smaller number than
    # treating each window as a day.
    assert abs(five_day["annualized"]) < abs(daily["annualized"])

    expected = (1.0 + five_day["net"]) ** (252 / 5) - 1.0
    assert five_day["annualized"] == pytest.approx(expected)


def test_annualisation_compounds_rather_than_scales():
    """Compounding is not the same as multiplying by the window count."""
    net = 0.0134
    simple = net * (252 / 5)
    compounded = (1.0 + net) ** (252 / 5) - 1.0
    assert not math.isclose(simple, compounded, rel_tol=0.01)
    result = dec.signal_utility(
        np.array([1.0] * 50),
        np.full(50, net),
        cost_bps=0.0,
        period_days=5,
    )
    assert result["annualized"] == pytest.approx(compounded, rel=1e-6)


def test_a_forecast_equal_to_naive_earns_nothing():
    """The null case: a model that says exactly what naive says must score ~0.

    Without this, any function that reports a large positive number for a
    useless forecast still passes every "is it positive" assertion.
    """
    summary = dec.decision_summary(_price_frame(1.0, skill=0.0), "Model")
    assert abs(summary["utility"]["net"]) < 0.01
    assert summary["sharpe"] == pytest.approx(0.0)
