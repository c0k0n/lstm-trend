"""Scale-free scoring: is a forecast better than 'repeat yesterday'?

A dollar error like "$2.14" cannot be compared across tickers, windows or
years. Dividing by what the naive guess would have cost turns it into a number
that means the same thing everywhere: 0.93 means "7% better than repeating
yesterday's price", 1.08 means 8% worse.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypedDict

import numpy as np
import pandas as pd
from scipy import stats

BASELINE = "Naive (yesterday)"
"""Every model is scored against this one."""

SKILL_MARGIN = 0.05
"""How much better than naive a model must be before we call it an edge."""

WIN_RATE_MARGIN = 0.6
"""Share of windows a model must win before the edge is called consistent."""

SIGNIFICANCE_LEVEL = 0.05
"""A win rate this unlikely under a coin flip is the bar for calling it real."""


def skill_from_mae(mae_model: float, mae_baseline: float) -> float:
    """Turn two mean absolute errors into one skill number.

    The single definition of the formula, so the Findings page and the
    walk-forward test cannot drift apart. Guards a zero baseline: when the
    naive guess is exactly right there is nothing to divide by, and the honest
    answer is "no information", not infinity.
    """
    return 1.0 - mae_model / mae_baseline if mae_baseline else 0.0


def model_summary(frame: pd.DataFrame, baseline: str = BASELINE) -> pd.DataFrame:
    """Per-model error, skill and win rate across all walk-forward windows.

    `frame` holds one row per origin with an `actual` column and one column of
    predicted prices per model. Skill is 1 - MAE(model) / MAE(baseline), so it
    is positive when the model beat the baseline and negative when it did not.
    """
    if baseline not in frame.columns:
        raise ValueError(f"frame has no baseline column {baseline!r}")

    actual = frame["actual"].to_numpy(dtype=float)
    baseline_error = abs(actual - frame[baseline].to_numpy(dtype=float))

    rows = []
    for model in frame.columns:
        if model in ("actual", "origin", "target"):
            continue
        error = abs(actual - frame[model].to_numpy(dtype=float))
        mae_model = float(error.mean())
        mae_baseline = float(baseline_error.mean())
        wins = int((error < baseline_error).sum())

        rows.append(
            {
                "model": model,
                "is_baseline": model == baseline,
                "windows": int(len(error)),
                "mae": mae_model,
                "mae_baseline": mae_baseline,
                "skill": 1.0 - mae_model / mae_baseline if mae_baseline else 0.0,
                "wins": wins,
                "win_rate": wins / len(error) if len(error) else 0.0,
            }
        )

    summary = pd.DataFrame(rows)
    return summary.sort_values("skill", ascending=False).reset_index(drop=True)


@dataclass
class _PoolRecord:
    """Running totals for one model across every ticker's windows."""

    skills: list[float] = field(default_factory=list)
    wins: int = 0
    windows: int = 0
    tickers: set[str] = field(default_factory=set)


def pooled_summary(
    frames: dict[str, pd.DataFrame], baseline: str = BASELINE
) -> pd.DataFrame:
    """Skill pooled across several tickers, so the sample is windows not tickers.

    Ten windows on one stock cannot resolve anything — even seven wins out of
    ten gives p=0.34. Pooling a hundred and twenty windows from ten stocks can.

    Dollar errors do not add across tickers: two dollars on a forty-dollar
    stock is not the same mistake as two dollars on a three-hundred-dollar one.
    So skill is computed *within* each ticker first and then averaged across
    tickers, one vote per ticker.

    Averaging per-window ratios instead would look equivalent and is not: on a
    flat day the baseline is nearly perfect, its error approaches zero, and the
    ratio explodes. A single quiet window then swamps the entire average —
    this produced a nonsensical −161% skill before the order was fixed.
    """
    records: dict[str, _PoolRecord] = {}

    for symbol, frame in frames.items():
        if baseline not in frame.columns:
            raise ValueError(f"{symbol}: no baseline column {baseline!r}")

        actual = frame["actual"].to_numpy(dtype=float)
        baseline_error = np.abs(actual - frame[baseline].to_numpy(dtype=float))

        for model in frame.columns:
            if model in ("actual", "origin", "target"):
                continue
            error = np.abs(actual - frame[model].to_numpy(dtype=float))

            record = records.setdefault(model, _PoolRecord())
            mae_baseline = float(baseline_error.mean())
            record.skills.append(skill_from_mae(float(error.mean()), mae_baseline))
            record.wins += int((error < baseline_error).sum())
            record.windows += len(error)
            record.tickers.add(symbol)

    rows = []
    for model, record in records.items():
        windows = record.windows
        skill = float(np.mean(record.skills))
        rows.append(
            {
                "model": model,
                "is_baseline": model == baseline,
                "windows": windows,
                "tickers": len(record.tickers),
                # Deliberately absent: `mae` and `mae_baseline`. `model_summary`
                # fills those with dollars, and a pooled dollar error is not a
                # thing — dollar errors do not add across tickers, which is the
                # whole reason this function exists. They used to be filled
                # with 1 - skill and 1.0, which shared a column name with real
                # dollars and held a ratio. Skill is the pooled quantity; it
                # is the only honest one to report here.
                "skill": skill,
                "wins": record.wins,
                "win_rate": record.wins / windows if windows else 0.0,
            }
        )

    summary = pd.DataFrame(rows)
    return summary.sort_values("skill", ascending=False).reset_index(drop=True)


def significance(summary_row: pd.Series) -> float:
    """Two-sided sign-test p-value for that model's win rate.

    With a handful of windows, "won 5 of 8" is not obviously different from a
    coin flip. This says how often a coin flip would produce a record at least
    that lopsided. High p-value means the win rate is not evidence of anything.
    """
    windows = int(summary_row["windows"])
    wins = int(summary_row["wins"])
    if windows == 0:
        return 1.0
    result = stats.binomtest(wins, windows, p=0.5, alternative="two-sided")
    return float(result.pvalue)


class Verdict(TypedDict):
    """The plain-language reading of one model's record."""

    label: str
    edge: bool
    skill: float
    win_rate: float
    pvalue: float
    windows: int
    detail: str
    note: str


def verdict_for(summary_row: pd.Series) -> Verdict:
    """A plain-language reading of one model's record.

    Deliberately conservative. Beating the baseline on a handful of windows is
    easy to do by luck, so an edge is only *suggested* when the margin and the
    consistency are both there — and even then the wording stays hedged.
    """
    skill = float(summary_row["skill"])
    win_rate = float(summary_row["win_rate"])
    pvalue = significance(summary_row)
    windows = int(summary_row["windows"])
    wins = int(summary_row["wins"])

    margin_ok = skill >= SKILL_MARGIN
    consistency_ok = win_rate >= WIN_RATE_MARGIN
    significant = pvalue < SIGNIFICANCE_LEVEL

    edge = margin_ok and consistency_ok and significant

    if edge:
        label = "Edge suggested"
        detail = (
            f"Beat the naive guess by about {skill:.0%} on average, won "
            f"{wins} of {windows} windows, and a coin flip would produce a "
            f"record that lopsided only about {pvalue:.0%} of the time."
        )
    elif skill <= 0:
        label = "No edge"
        detail = (
            f"No better than repeating yesterday's price ({skill:+.0%}). "
            "On this ticker and horizon the model is not adding information."
        )
    else:
        label = "No reliable edge"
        failed = []
        if not margin_ok:
            failed.append(f"the average improvement was only {skill:.0%}")
        if not consistency_ok:
            failed.append(f"it won just {wins} of {windows} windows")
        if not significant:
            failed.append(
                f"a coin flip would produce a win rate this lopsided about "
                f"{pvalue:.0%} of the time"
            )
        detail = (
            f"Better than naive on average, but not established: {'; '.join(failed)}."
        )

    note = f"Scored on {windows} windows."
    if windows < 12:
        note += " That is few — treat this as indicative and raise the count for a sharper read."
    if bool(summary_row.get("is_baseline", False)):
        note = "This is the baseline every other model is measured against."

    return {
        "label": label,
        "edge": edge,
        "skill": skill,
        "win_rate": win_rate,
        "pvalue": pvalue,
        "windows": windows,
        "detail": detail,
        "note": note,
    }
