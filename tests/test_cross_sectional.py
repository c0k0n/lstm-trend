"""Tests for the cross-sectional ranking controls. Deleted after running."""

import numpy as np
import pandas as pd
import pytest

from src.core.cross_sectional import (
    cross_sectional_backtest,
    rank_within_date,
    ranked_relative_frame,
    sector_relative_returns,
    top_k_long_short,
)


def _prices(seed: int, n: int = 400, drift: float = 0.0004) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return 100 * np.exp(np.cumsum(rng.normal(drift, 0.012, n)))


def _frame(series: np.ndarray):
    import pandas as pd

    index = pd.bdate_range("2022-01-03", periods=len(series))
    close = pd.Series(series, index=index, name="Close")
    return pd.DataFrame(
        {
            "Open": close,
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": np.full(len(series), 1_000_000.0),
        },
        index=index,
    )


def test_ranking_is_a_percentile_between_zero_and_one():
    values = pd.Series({"A": 3.0, "B": 1.0, "C": 2.0})
    ranked = rank_within_date(values)
    assert ranked.min() >= 0.0
    assert ranked.max() <= 1.0
    assert ranked["B"] < ranked["C"] < ranked["A"]


def test_ranking_gives_ties_the_same_score():
    values = pd.Series({"A": 1.0, "B": 1.0, "C": 5.0})
    ranked = rank_within_date(values)
    assert ranked["A"] == ranked["B"], "ties must not be broken arbitrarily"


def test_a_perfect_score_selects_the_winners():
    """The control must be attainable, or every result is vacuous.

    Twelve names rather than six: with only 6 there are just 20 distinct
    top-3 subsets, so a perfect selection still lands mid-distribution and the
    p-value cannot get near 0.05 however good it is. The universe has to be
    large enough for a real selection to be distinguishable from chance.
    """
    names = [chr(ord("A") + i) for i in range(12)]
    scores = {n: float(11 - i) for i, n in enumerate(names)}
    # The top 6 by score all go up; the bottom 6 all go down.
    realised = {n: (0.05 if i < 6 else -0.05) for i, n in enumerate(names)}
    result = top_k_long_short(scores, realised, k=3)
    assert result["picked"] == names[:3]
    assert result["avoided"] == names[-3:]
    assert result["spread"] > 0
    assert result["p_value"] < 0.05, (
        f"a perfect ranking must beat the random control, got p={result['p_value']}"
    )


def test_random_picks_score_near_the_p_value_one():
    """A ranking that cannot beat a random pick must not be reported as edge."""
    rng = np.random.default_rng(7)
    scores = dict(zip("ABCDEFGH", rng.normal(size=8), strict=True))
    realised = {k: float(rng.normal(0, 0.02)) for k in scores}
    result = top_k_long_short(scores, realised, k=3)
    # No relationship between score and outcome, so the observed spread sits in
    # the bulk of the random distribution.
    assert result["p_value"] > 0.05
    assert abs(result["spread"]) < 0.05


def test_k_is_capped_at_half_the_universe():
    """Asking for the top 3 of 4 names would trivially cover everything."""
    scores = {c: float(i) for i, c in enumerate("ABCD")}
    realised = {c: 0.0 for c in scores}
    result = top_k_long_short(scores, realised, k=99)
    assert result["k"] == 2
    assert len(result["picked"]) == 2
    assert len(result["avoided"]) == 2


def test_the_p_value_is_reproducible():
    """A reported p-value must not move when the page reloads."""
    scores = dict(zip("ABCDE", [3.0, 2.0, 1.0, 0.0, -1.0], strict=True))
    realised = dict(zip("ABCDE", [0.03, 0.02, 0.01, -0.01, -0.02], strict=True))
    first = top_k_long_short(scores, realised, k=2, seed=42)
    second = top_k_long_short(scores, realised, k=2, seed=42)
    assert first["p_value"] == second["p_value"]


def test_a_single_name_cannot_be_ranked():
    result = top_k_long_short({"A": 1.0}, {"A": 0.1}, k=1)
    assert result["n"] == 1
    assert result["spread"] == 0.0


def test_sector_relative_returns_subtract_a_control():
    import pandas as pd

    frames = {"AAA": _frame(_prices(1)), "XLK": _frame(_prices(2))}
    relative = sector_relative_returns(frames, horizon=5)
    assert set(relative.columns) >= {"AAA"}
    assert "sector" in relative.columns
    # AAA's sector ETF is XLK, so the relative column must be a difference of
    # two forward returns rather than the raw one.
    assert relative["AAA"].notna().any()
    assert not np.allclose(relative["AAA"].dropna().to_numpy(), 0.0, atol=1e-12), (
        "a relative return that is identically zero means nothing was subtracted"
    )


def test_ranked_frame_preserves_the_column_set():
    import pandas as pd

    frame = pd.DataFrame(
        {"AAA": [0.01, 0.02, 0.03], "BBB": [-0.01, 0.0, 0.04]},
        index=pd.bdate_range("2024-01-01", periods=3),
    )
    ranked = ranked_relative_frame(frame)
    assert set(ranked.columns) == {"AAA", "BBB"}
    assert ranked["AAA"].between(0, 1).all()


def test_cross_sectional_needs_more_than_one_ticker():
    with pytest.raises(ValueError, match="at least two"):
        cross_sectional_backtest({"AAA": _frame(_prices(1))}, horizon=5, n_origins=2)
