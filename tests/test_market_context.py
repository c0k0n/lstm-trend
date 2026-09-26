"""Tests for the cross-asset context.

The join is where this goes wrong. Market data arrives on its own calendar,
and a stock's calendar disagrees with it in ways that only show up on a few
days a year — which is exactly the kind of bug that survives a glance and then
reads as skill. So the tests here are about the join, not the arithmetic.
"""

import numpy as np
import pandas as pd
import pytest

from src.core.features import FEATURE_COLUMNS, build_features
from src.core.market_context import (
    CONTEXT_COLUMNS,
    MARKET_COLUMNS,
    SECTOR_COLUMNS,
    align_context,
    sector_etf_for,
)


def _context(n: int = 300, seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2022-01-03", periods=n)
    return pd.DataFrame(
        {column: rng.normal(size=n) for column in CONTEXT_COLUMNS}, index=index
    )


def _prices(n: int = 300, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    volume = rng.integers(1_000_000, 5_000_000, n).astype(float)
    index = pd.bdate_range("2022-01-03", periods=n)
    return pd.DataFrame({"Close": close, "Volume": volume}, index=index)


def test_context_columns_are_market_plus_sector():
    assert CONTEXT_COLUMNS == MARKET_COLUMNS + SECTOR_COLUMNS
    assert len(set(CONTEXT_COLUMNS)) == len(CONTEXT_COLUMNS)


def test_sector_etf_falls_back_to_the_broad_market():
    assert sector_etf_for("AAPL") == "XLK"
    assert sector_etf_for("JPM") == "XLF"
    assert sector_etf_for("AAPL") == sector_etf_for("aapl")
    assert sector_etf_for("SOMETHING-NOT-LISTED") == "SPY"


def test_alignment_takes_the_most_recent_earlier_day():
    context = pd.DataFrame(
        {"vix_level": [10.0, 20.0, 30.0]},
        index=pd.to_datetime(["2022-01-03", "2022-01-04", "2022-01-05"]),
    )
    aligned = align_context(context, pd.to_datetime(["2022-01-05", "2022-01-06"]))
    # The 5th exists, so it is taken exactly. The 6th has no market row, so it
    # receives the 5th — never a later one, because there is no later one.
    assert aligned["vix_level"].tolist() == [30.0, 30.0]


def test_alignment_leaves_the_unknowable_nan_rather_than_reaching_back():
    context = pd.DataFrame(
        {"vix_level": [10.0, 20.0]},
        index=pd.to_datetime(["2022-01-04", "2022-01-05"]),
    )
    aligned = align_context(context, pd.to_datetime(["2022-01-03", "2022-01-04"]))
    assert np.isnan(aligned["vix_level"].iloc[0])
    assert aligned["vix_level"].iloc[1] == 10.0


def test_alignment_does_not_reach_forward_across_a_gap():
    """A stock that missed a market day must not be handed that day's value."""
    context = pd.DataFrame(
        {"vix_level": [10.0, 99.0, 30.0]},
        index=pd.to_datetime(["2022-01-03", "2022-01-04", "2022-01-05"]),
    )
    # The stock did not trade on the 4th. Whatever it gets on the 5th must be
    # the 5th's own value, not the 4th's, and never anything later.
    aligned = align_context(context, pd.to_datetime(["2022-01-05"]))
    assert aligned["vix_level"].iloc[0] == 30.0


def test_no_context_column_looks_forward():
    """Mutating future market data must not change any earlier row."""
    data = _prices(n=300)
    context = _context(n=300)
    cutoff = 200

    before = build_features(data, context)

    mutated = context.copy()
    mutated.iloc[cutoff:] += 50.0
    after = build_features(data, mutated)

    pd.testing.assert_frame_equal(before.iloc[:cutoff], after.iloc[:cutoff])
    assert (
        (
            after.iloc[cutoff:][list(CONTEXT_COLUMNS)]
            != before.iloc[cutoff:][list(CONTEXT_COLUMNS)]
        )
        .any()
        .any()
    )


def test_mutating_the_stock_alone_does_not_move_context():
    data = _prices(n=300)
    context = _context(n=300)

    before = build_features(data, context)
    mutated = data.copy()
    mutated["Close"] *= 1.5
    after = build_features(mutated, context)

    pd.testing.assert_frame_equal(
        before[list(CONTEXT_COLUMNS)], after[list(CONTEXT_COLUMNS)]
    )


def test_with_context_the_columns_are_the_declared_ones():
    feats = build_features(_prices(), _context())
    assert list(feats.columns) == list(FEATURE_COLUMNS) + list(CONTEXT_COLUMNS)


def test_without_context_nothing_changes():
    pd.testing.assert_frame_equal(build_features(_prices()), build_features(_prices()))


def test_a_short_context_run_yields_nan_not_a_crash():
    """Context that starts late must read as missing, not silently shift."""
    data = _prices(n=300)
    late = _context(n=300).iloc[250:].copy()
    feats = build_features(data, late)
    assert feats[list(CONTEXT_COLUMNS)].iloc[:250].isna().all().all()
    assert feats[list(CONTEXT_COLUMNS)].iloc[250:].notna().all().all()


@pytest.mark.parametrize("column", CONTEXT_COLUMNS)
def test_every_column_is_actually_populated(column: str) -> None:
    feats = build_features(_prices(), _context())
    assert feats[column].iloc[100:].notna().all()
