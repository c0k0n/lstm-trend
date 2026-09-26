"""Tests for the scoring core.

These guard the two mistakes this module has actually made in production:
calling noise an edge, and aggregating pooled skill in the wrong order.
"""

import numpy as np
import pandas as pd
import pytest

from src.core.skill import (
    BASELINE,
    model_summary,
    pooled_summary,
    significance,
    skill_from_mae,
    verdict_for,
)


def test_skill_from_mae_is_the_shared_definition():
    """Findings and Evidence both score from here, so it is pinned directly."""
    assert skill_from_mae(0.8, 1.0) == pytest.approx(0.2)
    assert skill_from_mae(1.2, 1.0) == pytest.approx(-0.2)
    assert skill_from_mae(1.0, 1.0) == pytest.approx(0.0)
    # A perfect baseline cannot be divided by; guard rather than explode.
    assert skill_from_mae(1.0, 0.0) == 0.0


def _frame(actual, **models) -> pd.DataFrame:
    frame = pd.DataFrame({"actual": actual})
    for name, values in models.items():
        frame[name] = values
    return frame


def test_baseline_scores_zero_against_itself():
    frame = _frame(actual=[100.0, 101.0, 102.0], **{BASELINE: [100.0, 100.0, 101.0]})
    row = model_summary(frame).iloc[0]
    assert row["skill"] == pytest.approx(0.0)
    assert bool(row["is_baseline"])


def test_perfect_model_scores_one():
    # Baseline is wrong on every window, so a perfect model wins all three
    # rather than tying on a window where naive happened to be exactly right.
    actual = [100.0, 101.0, 102.0]
    frame = _frame(
        actual=actual, **{BASELINE: [101.0, 100.0, 101.0], "Perfect": actual}
    )
    row = model_summary(frame).set_index("model").loc["Perfect"]
    assert row["skill"] == pytest.approx(1.0)
    assert row["wins"] == 3


def test_worse_than_baseline_scores_negative():
    actual = [100.0, 101.0, 102.0]
    frame = _frame(
        actual=actual,
        **{BASELINE: [100.0, 100.0, 101.0], "Bad": [110.0, 90.0, 80.0]},
    )
    row = model_summary(frame).set_index("model").loc["Bad"]
    assert row["skill"] < 0


def test_missing_baseline_column_raises():
    with pytest.raises(ValueError, match="baseline"):
        model_summary(_frame(actual=[1.0, 2.0], Model=[1.0, 2.0]))


# --- the pooling-order bug -------------------------------------------------


def test_pooled_skill_survives_a_near_perfect_baseline_window():
    """A window where naive is almost exactly right must not swamp the average.

    Averaging per-window ratios puts `1 - error/baseline_error` from that one
    window into the mean directly, and with baseline_error near zero the term
    runs to tens of thousands. This is what produced -161% skill.
    """
    rng = np.random.default_rng(0)
    n = 12
    actual = 100.0 + rng.normal(0, 2.0, n)
    baseline = actual + rng.normal(0, 0.5, n)
    model = actual + rng.normal(0, 0.7, n)

    # One quiet window: naive is essentially exactly right, model is off by a cent.
    baseline[3] = actual[3] + 1e-7
    model[3] = actual[3] + 0.01

    frame = _frame(actual=actual, **{BASELINE: baseline, "Model": model})
    skill = float(pooled_summary({"X": frame}).set_index("model").loc["Model", "skill"])

    assert -5.0 < skill < 1.0, f"pooled skill exploded to {skill}"


def test_pooled_averages_across_tickers_not_windows():
    """Each ticker gets one vote, however many windows it contributed."""
    quiet = _frame(actual=[100.0] * 8, **{BASELINE: [100.0] * 8, "Model": [100.5] * 8})
    noisy = _frame(
        actual=[100.0, 110.0] * 4,
        **{BASELINE: [100.0, 100.0] * 4, "Model": [101.0, 108.0] * 4},
    )

    summary = pooled_summary({"quiet": quiet, "noisy": noisy}).set_index("model")
    row = summary.loc["Model"]

    # quiet: baseline error 0 -> skill 0.0. noisy: baseline mean 5.0, model
    # mean 1.5 -> skill 0.7. Average of the two tickers is 0.35.
    assert row["skill"] == pytest.approx(0.35)
    assert row["windows"] == 16
    assert row["tickers"] == 2


def test_pooled_requires_the_baseline_column():
    with pytest.raises(ValueError, match="baseline"):
        pooled_summary({"X": _frame(actual=[1.0], Model=[1.0])})


# --- the verdict -----------------------------------------------------------


def _row(skill: float, wins: int, windows: int, is_baseline: bool = False) -> pd.Series:
    return pd.Series(
        {
            "skill": skill,
            "wins": wins,
            "windows": windows,
            "win_rate": wins / windows if windows else 0.0,
            "is_baseline": is_baseline,
        }
    )


def test_noise_is_not_called_an_edge():
    """5 wins in 8 with a real margin is still luck, and must not be an edge."""
    verdict = verdict_for(_row(skill=0.114, wins=5, windows=8))
    assert verdict["edge"] is False
    assert verdict["label"] == "No reliable edge"
    assert "coin flip" in verdict["detail"]


def test_a_well_powered_record_is_called_an_edge():
    verdict = verdict_for(_row(skill=0.20, wins=90, windows=100))
    assert verdict["edge"] is True
    assert verdict["label"] == "Edge suggested"


def test_worse_than_naive_is_no_edge():
    verdict = verdict_for(_row(skill=-0.10, wins=30, windows=100))
    assert verdict["label"] == "No edge"
    assert verdict["edge"] is False


def test_losing_record_is_no_edge_even_when_wins_are_significant():
    """43% of windows won is significantly *worse* than a coin flip."""
    verdict = verdict_for(_row(skill=-0.059, wins=130, windows=300))
    assert verdict["edge"] is False
    assert significance(_row(skill=-0.059, wins=130, windows=300)) < 0.05


def test_baseline_row_is_described_as_the_baseline():
    verdict = verdict_for(_row(skill=0.0, wins=0, windows=300, is_baseline=True))
    assert "baseline" in verdict["note"]


def test_significance_of_a_coin_flip_is_not_significant():
    assert significance(_row(skill=0.0, wins=50, windows=100)) > 0.05


def test_significance_with_no_windows_is_one():
    assert significance(_row(skill=0.0, wins=0, windows=0)) == 1.0
