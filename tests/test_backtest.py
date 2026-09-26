"""Tests for the walk-forward harness.

The property worth testing is not that the numbers come out right — that is
what the real-data runs are for — but that context cannot leak sideways. Two
models here are deliberately blind to context, and if they ever stop being
blind, every covariate result in the project is void.
"""

import numpy as np
import pandas as pd
import pytest

from src.core.backtest import (
    MOVING_AVERAGE,
    BacktestConfig,
    _panel_factory,
    walk_forward,
)
from src.core.market_context import CONTEXT_COLUMNS
from src.core.skill import BASELINE

TICKERS = ("AAA", "BBB")


def _prices(n: int = 300, seed: int = 7) -> pd.DataFrame:
    """Volume included because real data always has it — without it every
    feature row is incomplete and LightGBM silently falls back to naive."""
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    volume = rng.integers(1_000_000, 5_000_000, n).astype(float)
    index = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame({"Close": close, "Volume": volume}, index=index)


def _context(n: int = 300, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame(
        {column: rng.normal(size=n) for column in CONTEXT_COLUMNS}, index=index
    )


def _both() -> dict[str, pd.DataFrame]:
    return {name: _prices(seed=i + 1) for i, name in enumerate(TICKERS)}


def test_panel_refuses_context_for_only_some_tickers():
    """Half-applied context would stack frames of different widths."""
    with pytest.raises(ValueError, match="differ in width"):
        _panel_factory(_both(), horizon=5, contexts={"AAA": _context()})


def test_panel_accepts_context_for_every_ticker():
    contexts = {name: _context(seed=i + 21) for i, name in enumerate(TICKERS)}
    # Constructing the factory is the part that would raise; no fitting needed.
    assert callable(_panel_factory(_both(), horizon=5, contexts=contexts))


def test_naive_and_moving_average_cannot_see_context():
    """The control that makes every covariate comparison mean anything."""
    config = BacktestConfig(horizon=5, n_origins=3, include_panel=False)
    without = walk_forward(_prices(), config)
    with_context = walk_forward(
        _prices(), config, symbol="AAA", contexts={"AAA": _context()}
    )

    for column in (BASELINE, MOVING_AVERAGE, "actual"):
        pd.testing.assert_series_equal(without[column], with_context[column])


def test_context_changes_what_the_learner_predicts():
    """And the converse: the model that can see context must react to it."""
    config = BacktestConfig(horizon=5, n_origins=3, include_panel=False)
    without = walk_forward(_prices(), config)
    with_context = walk_forward(
        _prices(), config, symbol="AAA", contexts={"AAA": _context()}
    )

    difference = float((without["LightGBM"] - with_context["LightGBM"]).abs().max())
    assert difference > 0.0, "context columns were not actually fed to the model"


def test_context_without_a_matching_symbol_is_an_error():
    """A silent fall back to the control arm would invalidate the run."""
    config = BacktestConfig(horizon=5, n_origins=2, include_panel=False)
    with pytest.raises(ValueError, match="No market context"):
        walk_forward(_prices(), config, contexts={"BBB": _context()})


def test_panel_refuses_context_missing_one_ticker():
    config = BacktestConfig(horizon=5, n_origins=2, include_panel=True)
    contexts = {"AAA": _context()}
    with pytest.raises(ValueError, match="differ in width"):
        walk_forward(
            _prices(),
            config,
            symbol="AAA",
            panel_data=_both(),
            contexts=contexts,
        )


def test_too_little_history_is_an_error_not_a_silent_empty():
    with pytest.raises(ValueError, match="Not enough history"):
        walk_forward(
            _prices(n=40),
            BacktestConfig(horizon=5, n_origins=3),
        )


def test_origins_leave_a_full_horizon_ahead():
    result = walk_forward(
        _prices(),
        BacktestConfig(horizon=5, n_origins=4),
    )
    assert len(result) == 4
    # Every window must have an actual price to be scored against.
    assert np.isfinite(result["actual"].to_numpy()).all()
