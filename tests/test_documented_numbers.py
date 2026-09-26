"""Pin the headline numbers to the raw backtest data they came from.

The project's credibility rests on a handful of figures quoted in the README,
the upgrade plan and the About page. They were transcribed by hand, and two of
them had drifted from the CSVs they claim to summarise — 122/300 became
117/300, and p = 0.0015 became p = 0.00017.

These tests recompute from `.workbuddy-ai/*.csv` using the same aggregation
`core.skill.pooled_summary` uses (skill within a ticker, then averaged across
tickers, because dollar errors do not add across stocks). If a figure in the
docs is wrong, this fails rather than letting it be re-transcribed wrong again.

Skipped when the CSVs are absent — they are gitignored working data, not
source, so a fresh clone legitimately has nothing to check against.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from src.core.skill import BASELINE

CSV_DIR = pathlib.Path(__file__).resolve().parent.parent / ".workbuddy-ai"

pytestmark = pytest.mark.skipif(
    not CSV_DIR.is_dir(),
    reason="raw backtest CSVs are gitignored working data, not present in a clone",
)


def _frames(prefix: str) -> dict[str, pd.DataFrame]:
    frames = {
        p.stem.replace(f"{prefix}_bt_", ""): pd.read_csv(p)
        for p in sorted(CSV_DIR.glob(f"{prefix}_bt_*.csv"))
    }
    if not frames:
        pytest.skip(f"no {prefix}_bt_*.csv in {CSV_DIR}")
    return frames


def _pooled(
    frames: dict[str, pd.DataFrame], model: str
) -> tuple[float, int, int, float]:
    """Skill averaged within each ticker then across tickers; wins summed."""
    skills: list[float] = []
    wins = windows = 0
    for frame in frames.values():
        if model not in frame.columns:
            pytest.skip(f"{model!r} not in this CSV family")
        actual = frame["actual"].to_numpy(dtype=float)
        base_err = np.abs(actual - frame[BASELINE].to_numpy(dtype=float))
        model_err = np.abs(actual - frame[model].to_numpy(dtype=float))
        skills.append(1.0 - model_err.mean() / base_err.mean())
        wins += int((model_err < base_err).sum())
        windows += len(model_err)
    pvalue = float(stats.binomtest(wins, windows, 0.5, alternative="two-sided").pvalue)
    return float(np.mean(skills)), wins, windows, pvalue


# --- the numbers the docs quote -------------------------------------------


def test_lstm_300_window_record_matches_the_docs():
    """README/upgrade-path/About all quote this row of the results table."""
    frames = _frames("lstm300")
    assert len(frames) == 20, f"expected 20 tickers, got {len(frames)}"

    skill, wins, windows, pvalue = _pooled(frames, "LSTM + features + context")

    # Quoted in README.md and docs/upgrade-path-2026.md.
    assert skill == pytest.approx(-0.150, abs=0.001)
    assert (wins, windows) == (117, 300)
    assert pvalue == pytest.approx(0.00017, abs=5e-6)

    # And the qualitative claim: reliably worse than doing nothing.
    assert pvalue < 0.05
    assert skill < 0


def test_no_model_in_the_300_window_run_beats_naive():
    """The project's headline: nothing beat 'repeat yesterday'."""
    frames = _frames("lstm300")
    sample = next(iter(frames.values()))
    for model in sample.columns:
        if model in ("origin", "target", "actual", BASELINE):
            continue
        skill, _, _, _ = _pooled(frames, model)
        assert skill < 0, f"{model} scored {skill:+.1%} — the docs claim none did"


def test_lstm_beat_naive_on_only_three_of_twenty_tickers():
    """Quoted as '3 of 20 tickers' — the per-ticker spread behind the mean."""
    frames = _frames("lstm300")
    better = 0
    for frame in frames.values():
        actual = frame["actual"].to_numpy(dtype=float)
        base_err = np.abs(actual - frame[BASELINE].to_numpy(dtype=float))
        model_err = np.abs(
            actual - frame["LSTM + features + context"].to_numpy(dtype=float)
        )
        if 1.0 - model_err.mean() / base_err.mean() > 0:
            better += 1
    assert better == 3


def test_moving_average_is_significantly_worse_at_300_windows():
    frames = _frames("lstm300")
    skill, wins, windows, pvalue = _pooled(frames, "Moving average")
    assert skill == pytest.approx(-0.337, abs=0.002)
    assert (wins, windows) == (120, 300)
    assert pvalue < 0.001


def test_lstm_upgrade_is_a_real_improvement_over_the_old_network():
    """The 200-window duel that decided the old network was deleted."""
    frames = _frames("duel")
    old_skill, _, _, _ = _pooled(frames, "LSTM")
    new_skill, _, _, _ = _pooled(frames, "LSTM + features")

    assert old_skill == pytest.approx(-0.425, abs=0.002)
    assert new_skill == pytest.approx(-0.165, abs=0.002)
    # The claim is "closed most of the gap", not "created an edge".
    assert new_skill > old_skill
    assert new_skill < 0, "the upgrade closed the gap; it did not beat naive"


def test_ohlcv_hurt_the_point_forecast_which_is_why_it_is_off():
    """`BacktestConfig.use_ohlcv` defaults to False because of this."""
    frames = _frames("ohlcv")
    close_only, _, _, _ = _pooled(frames, "LSTM close-only")
    with_ohlcv, _, _, _ = _pooled(frames, "LSTM + OHLCV")

    assert close_only == pytest.approx(-0.115, abs=0.002)
    assert with_ohlcv == pytest.approx(-0.189, abs=0.002)
    assert with_ohlcv < close_only


def test_panel_training_advantage_collapsed_at_scale():
    """+2.1% at 80 windows became -4.2% at 300. The recurring lesson."""
    frames = _frames("panel")
    panel, _, _, pvalue = _pooled(frames, "LightGBM (panel)")
    per_ticker, _, _, _ = _pooled(frames, "LightGBM")

    assert panel == pytest.approx(-0.042, abs=0.002)
    # Only a 1.7-point edge over per-ticker training, not the 13 the
    # 80-window sample suggested. Panel is better (less negative), but barely.
    assert 0.01 < (panel - per_ticker) < 0.025
    # And not significant.
    assert pvalue > 0.05
