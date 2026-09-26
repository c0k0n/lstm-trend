"""Prediction intervals by split-conformal calibration.

A point forecast without a range cannot be usefully wrong — it is just wrong,
with no way to tell a near miss from a disaster. This turns any of the
forecasters into a band using the errors they actually made out of sample,
rather than an assumed error distribution.

The calibration is scale-free: it works on `|error| / |actual|`, so a band
calibrated across many tickers still means something on any single one. Two
dollars on a forty-dollar stock and two dollars on a three-hundred-dollar one
are different mistakes, and a dollar-denominated band would blur them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_COVERAGE = 0.9
"""Fraction of outcomes the band aims to contain."""

_MIN_RESIDUALS = 5
"""Below this, a band is arithmetic on nothing and is not reported."""


def relative_residuals(frame: pd.DataFrame, model: str) -> np.ndarray:
    """Out-of-sample absolute error as a fraction of the realised price."""
    if model not in frame.columns:
        raise ValueError(f"frame has no model column {model!r}")

    actual = frame["actual"].to_numpy(dtype=float)
    predicted = frame[model].to_numpy(dtype=float)
    usable = np.isfinite(actual) & np.isfinite(predicted) & (actual != 0)
    return np.abs(actual[usable] - predicted[usable]) / np.abs(actual[usable])


def conformal_width(residuals: np.ndarray, coverage: float = DEFAULT_COVERAGE) -> float:
    """Half-width of the band, as a fraction of the forecast value.

    Split conformal with the usual finite-sample correction: the quantile level
    is `ceil((n + 1) * coverage) / n`, not `coverage`. Without it, a 90% band
    calibrated on ten residuals would silently contain rather less than 90%.

    Returns 0.0 when there are too few residuals to say anything, which callers
    must treat as "no band available" rather than "no uncertainty".
    """
    values = np.asarray(residuals, dtype=float)
    values = values[np.isfinite(values)]
    count = len(values)
    if count < _MIN_RESIDUALS:
        return 0.0

    level = min(1.0, np.ceil((count + 1) * coverage) / count)
    return float(np.quantile(values, level))


def parkinson_volatility(data: pd.DataFrame, window: int = 20) -> pd.Series:
    """Rolling volatility estimated from the daily high-low range.

    This is where the candle earns its place. Parkinson's estimator reads the
    whole day's range instead of only close-to-close, and is roughly five times
    more efficient — a 20-day range-based figure is about as accurate as a
    100-day close-based one. It was measured and *hurt* the point forecast, so
    it is not used there; here it is exactly what is needed.
    """
    if not {"High", "Low"}.issubset(data.columns):
        close = data["Close"].astype(float)
        return close.pct_change().rolling(window).std()

    high = data["High"].astype(float)
    low = data["Low"].astype(float)
    log_range = np.log(high / low).replace([np.inf, -np.inf], np.nan)
    variance = (log_range.pow(2) / (4.0 * np.log(2.0))).rolling(window).mean()
    return np.sqrt(variance.clip(lower=0.0))


def volatility_scaled_width(
    residuals: np.ndarray,
    volatility: np.ndarray,
    current_volatility: float,
    coverage: float = DEFAULT_COVERAGE,
    floor: float = 0.5,
    ceiling: float = 3.0,
) -> float:
    """A width that widens when the market is moving and narrows when it is not.

    A single constant width is calibrated on every window at once, so it is
    really the *average* mistake — too wide in calm periods, too narrow when
    volatility spikes. Errors are strongly proportional to volatility, so
    dividing each residual by the volatility that produced it gives a
    volatility-neutral error, and multiplying the calibrated quantile of those
    by today's volatility puts the scale back.

    The ratio is clamped. A single freak window in the calibration set could
    otherwise produce an absurd narrow or wide band, and the clamp keeps the
    adjustment honest rather than dramatic.
    """
    values = np.asarray(residuals, dtype=float)
    scale = np.asarray(volatility, dtype=float)

    usable = np.isfinite(values) & np.isfinite(scale) & (scale > 0)
    if usable.sum() < _MIN_RESIDUALS:
        return conformal_width(residuals, coverage)
    if not np.isfinite(current_volatility) or current_volatility <= 0:
        return conformal_width(residuals, coverage)

    neutral = values[usable] / scale[usable]
    baseline = float(np.mean(scale[usable]))
    if baseline <= 0:
        return conformal_width(residuals, coverage)

    ratio = float(np.clip(current_volatility / baseline, floor, ceiling))
    return float(conformal_width(neutral, coverage) * baseline * ratio)


def pooled_width(
    frames: dict[str, pd.DataFrame], model: str, coverage: float = DEFAULT_COVERAGE
) -> float:
    """Calibrate on every ticker's windows at once.

    One ticker gives about a dozen residuals, which is too few to locate a 90%
    quantile. Pooling ten tickers gives over a hundred.
    """
    batches = [
        relative_residuals(frame, model)
        for frame in frames.values()
        if model in frame.columns
    ]
    if not batches:
        return 0.0
    return conformal_width(np.concatenate(batches), coverage)


def band(point: float, width: float) -> tuple[float, float]:
    """Lower and upper bounds around one point forecast."""
    return point * (1.0 - width), point * (1.0 + width)


def horizon_band(
    points: np.ndarray, width: float, calibration_horizon: int
) -> tuple[np.ndarray, np.ndarray]:
    """Spread a single calibrated width across a multi-day forecast path.

    One calibration gives the spread at one horizon, but a forecast path needs a
    band at every step. Volatility grows roughly with the square root of time,
    so step `h` gets `width * sqrt(h / calibration_horizon)`.

    That scaling is an assumption, not a measurement. It is the reason realised
    coverage is reported next to every band in this app: an assumption you can
    check is fine, one presented as fact is not.
    """
    steps = np.arange(1, len(points) + 1, dtype=float)
    scaled = width * np.sqrt(steps / max(calibration_horizon, 1))
    return points * (1.0 - scaled), points * (1.0 + scaled)


def coverage_of(frame: pd.DataFrame, model: str, width: float) -> float:
    """Share of windows whose realised price fell inside the band.

    The check that makes the band honest. A model claiming 90% whose realised
    coverage is 60% is overconfident, and this is how you find out.
    """
    actual = frame["actual"].to_numpy(dtype=float)
    predicted = frame[model].to_numpy(dtype=float)
    usable = np.isfinite(actual) & np.isfinite(predicted)
    if not usable.any():
        return 0.0

    realised = actual[usable]
    points = predicted[usable]
    inside = (realised >= points * (1.0 - width)) & (realised <= points * (1.0 + width))
    return float(inside.mean())


def pooled_coverage(frames: dict[str, pd.DataFrame], model: str, width: float) -> float:
    """Realised coverage pooled across every ticker's windows."""
    counts: list[int] = []
    hits: list[int] = []
    for frame in frames.values():
        if model not in frame.columns:
            continue
        actual = frame["actual"].to_numpy(dtype=float)
        predicted = frame[model].to_numpy(dtype=float)
        usable = np.isfinite(actual) & np.isfinite(predicted)
        realised, points = actual[usable], predicted[usable]
        inside = (realised >= points * (1 - width)) & (realised <= points * (1 + width))
        counts.append(len(realised))
        hits.append(int(inside.sum()))

    total = sum(counts)
    return sum(hits) / total if total else 0.0
