"""Tests for the UI-layer logic that can be reached without a Streamlit runtime.

Two things here have already gone wrong once, and both failed *quietly* — a
cached band drawn over the wrong forecast, and a column called `mae` holding a
ratio. Neither would show up as an exception, so they are pinned directly.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.core.skill import BASELINE, model_summary, pooled_summary
from src.ui.result_rendering import _band_key


class _Result:
    """The minimum shape `_band_key` reads, so the test needs no pipeline run."""

    def __init__(self, symbol, start, end, steps=15, **params):
        self.symbol = symbol
        self.data = pd.DataFrame(
            {"Close": [1.0, 2.0]},
            index=pd.DatetimeIndex([start, end], name="Date"),
        )
        self.future = pd.DataFrame({"predicted_close": range(steps)})
        self.params = params or {"sequence_length": 60, "epochs": 50, "batch_size": 32}


def test_band_key_separates_different_date_ranges():
    """The bug: two runs of one ticker over different history shared a key.

    The second run then drew the first run's calibrated range over its own
    forecast — a number that looks right and describes the wrong dataset.
    """
    narrow = _band_key(_Result("AAPL", "2020-01-01", "2026-09-01"))
    wide = _band_key(_Result("AAPL", "2016-01-01", "2026-09-01"))
    assert narrow != wide


def test_band_key_separates_different_forecast_horizons():
    short = _band_key(_Result("AAPL", "2020-01-01", "2026-09-01", steps=15))
    long = _band_key(_Result("AAPL", "2020-01-01", "2026-09-01", steps=30))
    assert short != long


def test_band_key_separates_symbols_and_settings():
    base = _Result("AAPL", "2020-01-01", "2026-09-01")
    assert _band_key(_Result("MSFT", "2020-01-01", "2026-09-01")) != _band_key(base)
    assert _band_key(
        _Result("AAPL", "2020-01-01", "2026-09-01", epochs=20)
    ) != _band_key(base)


def test_band_key_is_stable_for_the_same_run():
    """A cache key that changes on every rerun would never hit the cache."""
    first = _band_key(_Result("AAPL", "2020-01-01", "2026-09-01", steps=15))
    second = _band_key(_Result("AAPL", "2020-01-01", "2026-09-01", steps=15))
    assert first == second


# --- the pooled summary schema --------------------------------------------


def _frame(actual, **models) -> pd.DataFrame:
    frame = pd.DataFrame({"actual": actual})
    for name, values in models.items():
        frame[name] = values
    return frame


def test_pooled_summary_reports_no_dollar_mae():
    """Dollar errors do not add across tickers; the column must not pretend.

    `model_summary` fills `mae` with dollars and `pooled_summary` used to fill
    the same column with `1 - skill`, so the two summaries shared a name and
    disagreed about what it held.
    """
    frame = _frame(
        actual=[100.0, 101.0, 99.0],
        **{BASELINE: [99.0, 100.0, 101.0], "Moving average": [98.0, 99.0, 100.0]},
    )
    pooled = pooled_summary({"AAA": frame})
    assert "mae" not in pooled.columns
    assert "mae_baseline" not in pooled.columns

    # And the honest quantity is still there, under an honest name.
    assert "skill" in pooled.columns
    assert pooled["skill"].between(-1.0, 1.0).all()


def test_model_summary_keeps_its_dollar_mae():
    """The per-ticker path is unchanged: there, dollars are meaningful."""
    frame = _frame(actual=[100.0, 101.0], **{BASELINE: [99.0, 100.0]})
    summary = model_summary(frame)
    assert "mae" in summary.columns
    naive = summary.set_index("model").loc[BASELINE]
    # |100-99| and |101-100| -> mean absolute error of one dollar.
    assert naive["mae"] == pytest.approx(1.0)
