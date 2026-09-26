"""Is a forecast *useful*, as opposed to accurate?

Every number in this project so far is skill: `1 - MAE(model) / MAE(naive)`.
That is the right yardstick for one question — "did it predict the price
better?" — and the wrong yardstick for the question a user actually has, which
is "should I act on this?"

The gap is not academic. A model with zero point skill and a well-calibrated
volatility forecast is *useful*: it tells you when to size down, when a range
is wide, when a stop is more likely to be hit. Judged only on MAE, that same
model scores exactly zero and reads as worthless. Meanwhile a model with
slightly positive point skill and no calibration is untradeable in practice,
because you cannot tell which of its forecasts to trust.

So this module scores the properties that survive contact with a decision:

- **direction** — did the sign of the move come out right, against the sign a
  coin flip would have got. Ties are not wins.
- **information ratio** — the return of the *ranked* forecast against the
  benchmark, i.e. can you tell the good days from the bad ones at all.
- **calibration of the band** — when the model says 90%, how often is the
  outcome inside? This is the property that is actually achievable here.
- **utility** — the return of acting on the signal, with a cost, over the same
  windows. The only number here that is measured in the units a user cares
  about.

None of this rescues a point forecaster that has no edge. It describes the
shapes of usefulness that survive once the edge is gone, which is the honest
remaining territory — and it is the territory this project should be working in.
"""

from __future__ import annotations

from typing import NotRequired, TypedDict

import numpy as np
import pandas as pd

TRADING_DAYS = 252
"""Trading days in a year, for annualising a daily-return ratio."""


def _finite(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Drop pairs where either side is not a finite number.

    `reshape(-1)` first, because a caller may hand in a scalar — `np.isfinite`
    on a 0-d array yields a 0-d boolean, which cannot be used to index, and the
    failure would surface as an IndexError a long way from the cause. Truncating
    to the shorter side means a length mismatch degrades to "use what both
    have" rather than raising, which keeps a partially-populated frame from
    taking down a whole page.
    """
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    size = min(a.size, b.size)
    a, b = a[:size], b[:size]
    keep = np.isfinite(a) & np.isfinite(b)
    return a[keep], b[keep]


def directional_accuracy(
    predicted_move: np.ndarray, actual_move: np.ndarray
) -> DirectionScore:
    """How often the sign was right, and against what baseline.

    `predicted_move` is any signed quantity whose sign is the call — a price
    delta, a predicted return, a rank. Magnitude is ignored on purpose: the
    question is "up or down", and folding in the size of the call would smuggle
    the point-forecast question back in.

    A no-change day is a genuine miss. Treating a zero move as half a win is
    the kind of generosity that turns a coin flip into a claim, and this project
    has been bitten by exactly that twice already.
    """
    p, a = _finite(predicted_move, actual_move)
    if p.size == 0:
        return {"accuracy": 0.0, "n": 0, "edge": 0.0, "beats_coin_flip": False}

    correct = np.sign(p) == np.sign(a)
    accuracy = float(correct.mean())
    coin_flip = float(max((np.sign(a) == 1).mean(), (np.sign(a) == -1).mean()))
    edge = accuracy - coin_flip
    return {
        "accuracy": accuracy,
        "n": int(p.size),
        "edge": float(edge),
        "beats_coin_flip": bool(edge > 0),
    }


def information_ratio(
    predicted_move: np.ndarray,
    actual_move: np.ndarray,
    baseline_move: np.ndarray,
) -> InformationScore:
    """Can the forecast *rank* days — separate good from bad — better than the baseline?

    This is the property that survives when point accuracy does not. A forecast
    need not be right often to be useful; it needs to put the days it is right
    on in front of the days it is wrong on. That is exactly what a trader does
    with a score, and it is invisible to MAE.

    Measured as the mean of `sign(predicted) * sign(actual)` for the model,
    divided by the same quantity for the baseline. Both are in [-1, 1] and the
    ratio says how much more of the spread the model can see.
    """
    p, a = _finite(predicted_move, actual_move)
    _, b = _finite(predicted_move, baseline_move)
    if p.size == 0 or b.size != p.size or a.size != p.size:
        return {"model": 0.0, "baseline": 0.0, "ratio": 0.0, "n": int(p.size)}

    model_score = float(np.mean(np.sign(p) * np.sign(a)))
    baseline_score = float(np.mean(np.sign(b) * np.sign(a)))
    if abs(baseline_score) < 1e-12:
        ratio = float("inf") if abs(model_score) > 1e-12 else 1.0
    else:
        ratio = model_score / baseline_score
    return {
        "model": model_score,
        "baseline": baseline_score,
        "ratio": float(ratio),
        "n": int(p.size),
    }


def band_calibration(
    actual: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> dict[str, float]:
    """Did the promised range contain the outcome as often as it claimed?

    `coverage` is what the band actually achieved; `width` is how much room it
    bought for that. Together they answer the only question a band has to pass:
    *can you rely on this?* A band that is 80% wide and 90% accurate is honest;
    one that is 20% wide and 90% accurate is a lie.
    """
    # One shared mask across all three series. Filtering each pair separately
    # would let a NaN in `lower` and a different NaN in `upper` shift the two
    # bounds relative to each other, and the result would be a coverage number
    # computed against mismatched rows — plausible, plausible-looking, and
    # meaningless.
    a = np.asarray(actual, dtype=float).reshape(-1)
    lo = np.asarray(lower, dtype=float).reshape(-1)
    hi = np.asarray(upper, dtype=float).reshape(-1)
    size = min(a.size, lo.size, hi.size)
    a, lo, hi = a[:size], lo[:size], hi[:size]
    keep = np.isfinite(a) & np.isfinite(lo) & np.isfinite(hi)
    a, lo, hi = a[keep], lo[keep], hi[keep]
    if a.size == 0:
        return {"coverage": 0.0, "width": 0.0, "n": 0}
    inside = (a >= lo) & (a <= hi)
    span = np.abs(hi - lo)
    return {
        "coverage": float(inside.mean()),
        "width": float(span.mean()),
        "n": int(a.size),
    }


def signal_utility(
    predicted_move: np.ndarray,
    actual_move: np.ndarray,
    cost_bps: float = 10.0,
    period_days: int = 1,
    periods: int = TRADING_DAYS,
) -> UtilityScore:
    """The return of simply acting on the sign, after costs.

    Deliberately the most obvious possible rule: go long when the forecast says
    up, flat when it says down, and pay `cost_bps` on every position. No
    position sizing, no stop, no cleverness — because a complicated strategy
    fitted on the same windows it is scored on is how a backtest lies to you.

    Costs are on by default. A strategy that only works at zero cost is not a
    strategy. At 10 bps per side on a daily flip, the friction is real, and it
    is the number that stops a 52%-accurate forecaster from looking like a
    money machine.

    `period_days` is how many trading days one window spans — 5 for this
    project's walk-forward test. It matters only for `annualized`, and getting
    it wrong is not a rounding error: scaling a 5-day return by 252 instead of
    by 252/5 reports +339% where compounding says +96%. A "return" that
    depends on how often you sampled it is not a return.
    """
    p, a = _finite(predicted_move, actual_move)
    if p.size == 0:
        return UtilityScore(
            net=0.0,
            gross=0.0,
            hit_rate=0.0,
            cost=0.0,
            annualized=0.0,
            period_days=1,
            n=0,
        )

    position = np.sign(p)
    # A flat day costs nothing, matching "do not open a position".
    traded = position != 0.0
    cost = float(traded.sum()) * (cost_bps / 10_000.0)

    gross = float(np.mean(position * a))
    net = gross - cost / p.size
    windows_per_year = periods / max(1, period_days)
    return {
        "net": net,
        "gross": gross,
        "hit_rate": float((position == np.sign(a)).mean()),
        "cost": cost,
        # Compounded, not scaled. `net * windows_per_year` would be a linear
        # extrapolation that overstates a good run and understates a bad one;
        # (1 + net) ** n - 1 is the standard annualisation and is the honest
        # one to put in front of a reader.
        "annualized": float((1.0 + net) ** windows_per_year - 1.0),
        "period_days": int(period_days),
        "n": int(p.size),
    }


def sharpe_of_signal(
    predicted_move: np.ndarray,
    actual_move: np.ndarray,
    cost_bps: float = 10.0,
    period_days: int = 1,
    periods: int = TRADING_DAYS,
) -> float:
    """Annualised Sharpe of the long/flat rule, costs included.

    Uses the *net* per-period returns rather than the mean, so the
    annualisation reflects variability and not just the average day. A rule
    that wins a little every day and occasionally loses a lot scores low here,
    which is the correct verdict.

    `cost_bps` is the same knob `signal_utility` exposes, and the two functions
    share the identical cost convention: charged once per day a position is
    open, at the full quoted size. It is a parameter rather than a constant
    because a cost assumption baked into a scoring function cannot be
    interrogated, and "what if it were 5 bps" is the first question anyone asks
    of a strategy number.
    """
    p, a = _finite(predicted_move, actual_move)
    if p.size < 3:
        return 0.0
    position = np.sign(p)
    cost_rate = cost_bps / 10_000.0
    per_period = position * a - np.where(position != 0.0, cost_rate, 0.0)
    sd = float(np.std(per_period, ddof=1))
    if sd <= 1e-15:
        return 0.0
    # Annualised by the square root of the windows per year, so a 5-day
    # signal is not scaled as though it were a daily one.
    windows_per_year = periods / max(1, period_days)
    return float(np.mean(per_period) / sd * np.sqrt(windows_per_year))


def decision_summary(
    frame: pd.DataFrame,
    model: str,
    baseline: str = "Naive (yesterday)",
    band: tuple[np.ndarray, np.ndarray] | None = None,
    reference: str | None = None,
    cost_bps: float = 10.0,
    period_days: int = 1,
) -> DecisionSummary:
    """Everything decision-relevant about one model, on one walk-forward frame.

    The `frame` is the same one `skill.model_summary` consumes: one row per
    window, an `actual` price, and one column of predicted prices per model.

    The framing throughout is *relative to the naive guess*. A walk-forward
    window asks "from yesterday's close, which way?", so the realised move is
    `actual - baseline` and the model's call is `predicted - baseline`. Scoring
    against the naive anchor rather than against zero is what makes a
    long/flat rule meaningful: holding the stock is the zero-information
    alternative, and a forecast is only worth acting on if it beats that.

    `reference` names a second model column to rank against, for the
    information ratio. Leave it unset when there is nothing to compare to; the
    ratio's own baseline component still reports.

    Point accuracy is deliberately *not* included — `skill.py` owns that, and
    a number that lives in two places is a number that will drift.
    """
    if model not in frame.columns:
        raise ValueError(f"frame has no model column {model!r}")
    if baseline not in frame.columns:
        raise ValueError(f"frame has no baseline column {baseline!r}")

    actual = frame["actual"].to_numpy(dtype=float)
    predicted = frame[model].to_numpy(dtype=float)
    anchor = frame[baseline].to_numpy(dtype=float)

    # **Fractions, and relative to the naive anchor.** Two mistakes live here
    # and both produced numbers that looked plausible:
    #
    # 1. Using `actual - anchor` gives *dollars*. For a $300 stock that is a
    #    number like 4.20, which the utility function then reported as a "340%
    #    return" and annualised to +85909%.
    # 2. Dividing that dollar difference by the anchor is worse, not better:
    #    for a stock that moved 1%, `(actual - anchor) / anchor` is about
    #    −0.99, because `actual - anchor` is already the *whole* naive
    #    forecast expressed as a gap from an anchor of 1.0, not a small move.
    #
    # The correct form is `actual / anchor - 1`: the return from yesterday's
    # close, which is ~0.01 for a 1% day and identical whatever the share
    # price. Holding the stock is the zero-information alternative, and this
    # is what makes a long/flat rule interpretable and comparable across a
    # $20 name and a $600 one.
    anchor_safe = np.where(anchor != 0, anchor, np.nan)
    realised_move = actual / anchor_safe - 1.0
    model_call = predicted / anchor_safe - 1.0

    reference_call: np.ndarray | None = None
    if reference is not None:
        if reference not in frame.columns:
            raise ValueError(f"frame has no reference column {reference!r}")
        reference_call = frame[reference].to_numpy(dtype=float) / anchor_safe - 1.0

    # Without a second model to rank against, the naive move is its own
    # reference, so the ratio degrades to the model's own directional score
    # rather than dividing by zero.
    comparison = realised_move if reference_call is None else reference_call

    summary: DecisionSummary = {
        "model": model,
        "windows": int(len(actual)),
        "direction": directional_accuracy(model_call, realised_move),
        "information": information_ratio(model_call, realised_move, comparison),
        "utility": signal_utility(
            model_call, realised_move, cost_bps, period_days=period_days
        ),
        "sharpe": sharpe_of_signal(
            model_call, realised_move, cost_bps, period_days=period_days
        ),
    }
    if band is not None:
        lower, upper = band
        summary["band"] = band_calibration(actual, lower, upper)
    return summary


class DirectionScore(TypedDict):
    """How often the sign was right, against a coin flip."""

    accuracy: float
    n: int
    edge: float
    beats_coin_flip: bool


class InformationScore(TypedDict):
    """Whether the forecast can rank days, relative to a reference."""

    model: float
    baseline: float
    ratio: float
    n: int


class UtilityScore(TypedDict):
    """The long/flat rule after costs."""

    net: float
    gross: float
    hit_rate: float
    cost: float
    annualized: float
    period_days: int
    n: int


class DecisionSummary(TypedDict):
    """Everything decision-relevant about one model on one walk-forward frame.

    Typed all the way down so a consumer never has to cast. Under
    `dict[str, object]` every call site needs a `float(...)`, and a cast is
    exactly where a wrong assumption stops being caught.
    """

    model: str
    windows: int
    direction: DirectionScore
    information: InformationScore
    utility: UtilityScore
    sharpe: float
    band: NotRequired[dict[str, float]]


class LongShortResult(TypedDict):
    """The outcome of one long/short cross-sectional decision.

    A TypedDict rather than `dict[str, object]` so the numeric fields keep
    their types at the call site. Under `object` every consumer needs a cast,
    and a cast is exactly where a wrong assumption stops being caught.
    """

    spread: float
    equal_weight: float
    vs_equal_weight: float
    p_value: float
    n: int
    picked: list[str]
    avoided: list[str]
    k: int
    random_spread_mean: float


def rank_and_go_long_top(
    predictions: dict[str, float],
    actuals: dict[str, float],
    k: int = 5,
    seed: int = 42,
    n_random: int = 2000,
) -> LongShortResult:
    """Long the `k` names the model likes most, short the worst `k`.

    The equity is `top_k_mean - bottom_k_mean`, a dollar-neutral construction:
    it earns the *spread*, not a market bet, so a period where every stock
    falls is not automatically a loss. That matters because the alternative
    framing — "did the basket go up" — is answered yes most of the time and so
    proves nothing.

    Three controls, because two is not enough:

    - **equal weight**: long the whole universe. "Did picking beat holding" is
      the comparison a user actually cares about.
    - **random pick of the same size**: with a small universe the top-k can look
      excellent on luck alone. The p-value is where the observed spread sits in
      the distribution of random k-name spreads. If it is not near the top, the
      ranking is not real. A six-name universe has only 20 distinct top-3
      subsets, so no result can reach p < 0.05 there however good it is — the
      universe has to be large enough for a real selection to be
      distinguishable from chance, and `n_random` sets the resolution of the
      check itself.
    - **the k cap**: `k` is clamped to half the universe, because asking for
      the top 10 of 12 names would be asking for everything and the short side
      would overlap the long side.

    The random control uses a fixed seed so a reported p-value is reproducible
    rather than a number that moves when the page reloads.
    """
    common = sorted(set(predictions) & set(actuals))
    n = len(common)
    if n < 2:
        return LongShortResult(
            spread=0.0,
            equal_weight=0.0,
            vs_equal_weight=0.0,
            # No ranking was possible, so the random control is undefined. 1.0
            # is the honest value: "no evidence", not "no effect".
            p_value=1.0,
            n=n,
            picked=[],
            avoided=[],
            k=0,
            random_spread_mean=0.0,
        )

    ordered = sorted(common, key=lambda s: predictions[s], reverse=True)
    k = max(1, min(k, n // 2))
    top = ordered[:k]
    bottom = ordered[-k:]

    spread = float(
        np.mean([actuals[s] for s in top]) - np.mean([actuals[s] for s in bottom])
    )
    equal_weight = float(np.mean([actuals[s] for s in common]))

    values = np.array([actuals[s] for s in common], dtype=float)
    rng = np.random.default_rng(seed)
    random_spreads = np.empty(n_random, dtype=float)
    for i in range(n_random):
        pick = rng.permutation(n)
        random_spreads[i] = values[pick[:k]].mean() - values[pick[-k:]].mean()
    p_value = float((random_spreads >= spread).mean())

    return LongShortResult(
        spread=spread,
        equal_weight=equal_weight,
        vs_equal_weight=spread - equal_weight,
        p_value=p_value,
        n=n,
        k=k,
        picked=top,
        avoided=bottom,
        random_spread_mean=float(random_spreads.mean()),
    )
