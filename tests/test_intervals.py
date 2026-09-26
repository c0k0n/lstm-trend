"""Tests for the conformal bands."""

import numpy as np
import pandas as pd
import pytest

from src.core.intervals import (
    parkinson_volatility,
    volatility_scaled_width,
    DEFAULT_COVERAGE,
    band,
    conformal_width,
    coverage_of,
    horizon_band,
    pooled_coverage,
    pooled_width,
    relative_residuals,
)


def test_too_few_residuals_gives_no_band():
    """A band on four residuals is arithmetic on nothing."""
    assert conformal_width(np.array([0.01, 0.02, 0.03, 0.04])) == 0.0


def test_finite_sample_correction_is_applied():
    """Ten residuals at 90% must use ceil(11 * 0.9) / 10 = 1.0, not 0.9."""
    values = np.arange(1, 101, dtype=float) / 100
    width = conformal_width(values, DEFAULT_COVERAGE)
    assert width == pytest.approx(0.9109, abs=1e-4)
    assert width > np.quantile(values, 0.9)


def test_non_finite_residuals_are_dropped():
    values = np.array([0.01, 0.02, np.nan, 0.03, 0.04, np.inf, 0.05, 0.06])
    assert conformal_width(values) == pytest.approx(
        conformal_width(values[:2].tolist() + [0.03, 0.04, 0.05, 0.06])
    )


def test_relative_residuals_drops_zero_actuals():
    frame = pd.DataFrame({"actual": [100.0, 0.0, 200.0], "Model": [90.0, 1.0, 180.0]})
    residuals = relative_residuals(frame, "Model")
    assert len(residuals) == 2
    assert residuals[0] == pytest.approx(0.10)
    assert residuals[1] == pytest.approx(0.10)


def test_relative_residuals_requires_the_model_column():
    with pytest.raises(ValueError, match="model"):
        relative_residuals(pd.DataFrame({"actual": [1.0]}), "Nope")


def test_pooled_width_pools_every_ticker():
    frames = {
        "A": pd.DataFrame({"actual": [100.0] * 6, "Model": [90.0] * 6}),
        "B": pd.DataFrame({"actual": [100.0] * 6, "Model": [110.0] * 6}),
    }
    assert pooled_width(frames, "Model") == pytest.approx(0.10)
    assert pooled_width(frames, "Absent") == 0.0


def test_band_is_symmetric_in_relative_terms():
    low, high = band(100.0, 0.10)
    assert low == pytest.approx(90.0)
    assert high == pytest.approx(110.0)


def test_horizon_band_scales_with_square_root_of_time():
    """At the calibration horizon the width is exactly the calibrated width."""
    points = np.array([100.0] * 5)
    width = 0.10
    low, high = horizon_band(points, width, calibration_horizon=5)

    assert (points - low)[-1] == pytest.approx(width * points[-1])
    assert (high - points)[-1] == pytest.approx(width * points[-1])
    # Step 1 is narrower than step 5 by sqrt(1/5).
    assert (points - low)[0] == pytest.approx(points[0] * width * np.sqrt(1 / 5))


def test_horizon_band_grows_monotonically():
    _, high = horizon_band(np.array([100.0] * 10), 0.10, calibration_horizon=5)
    assert np.all(np.diff(high) >= -1e-12)


def test_coverage_counts_what_fell_inside():
    frame = pd.DataFrame(
        {"actual": [100.0, 100.0, 100.0], "Model": [100.0, 92.0, 50.0]}
    )
    assert coverage_of(frame, "Model", 0.10) == pytest.approx(2 / 3)
    assert coverage_of(frame, "Model", 1.00) == pytest.approx(1.0)


def test_coverage_with_nothing_usable_is_zero():
    frame = pd.DataFrame({"actual": [np.nan], "Model": [np.nan]})
    assert coverage_of(frame, "Model", 0.10) == 0.0


def test_pooled_coverage_is_a_single_fraction_over_every_window():
    frames = {
        "A": pd.DataFrame({"actual": [100.0], "Model": [100.0]}),
        "B": pd.DataFrame({"actual": [100.0], "Model": [200.0]}),
    }
    assert pooled_coverage(frames, "Model", 0.10) == pytest.approx(0.5)
    assert pooled_coverage({}, "Model", 0.10) == 0.0


def test_a_calibrated_band_covers_roughly_what_it_claims():
    """End to end: calibrate on residuals, then check coverage on new data."""
    rng = np.random.default_rng(7)
    actual = 100.0 + rng.normal(0, 5.0, 400)
    predicted = actual + rng.normal(0, 5.0, 400)

    frame = pd.DataFrame({"actual": actual, "Model": predicted})
    width = pooled_width({"X": frame}, "Model", DEFAULT_COVERAGE)
    realised = coverage_of(frame, "Model", width)

    assert width > 0
    assert abs(realised - DEFAULT_COVERAGE) < 0.05


def _candle_frame(n: int = 300, seed: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    spread = np.abs(rng.normal(0, 0.008, n))
    index = pd.bdate_range("2022-01-03", periods=n)
    return pd.DataFrame(
        {
            "Open": close * (1 + rng.normal(0, 0.003, n)),
            "High": close * (1 + spread),
            "Low": close * (1 - spread),
            "Close": close,
            "Volume": rng.integers(1_000_000, 5_000_000, n).astype(float),
        },
        index=index,
    )


def test_parkinson_uses_the_range_not_just_the_close():
    data = _candle_frame()
    wide = data.copy()
    width = (wide["High"] - wide["Low"]) * 3.0
    mid = (wide["High"] + wide["Low"]) / 2.0
    wide["High"] = mid + width / 2.0
    wide["Low"] = mid - width / 2.0

    assert parkinson_volatility(wide).iloc[-1] > parkinson_volatility(data).iloc[-1]


def test_parkinson_falls_back_without_a_candle():
    """Close-only data must still produce something, not a crash."""
    series = parkinson_volatility(pd.DataFrame({"Close": _candle_frame()["Close"]}))
    assert series.dropna().gt(0).all()


def test_volatility_scaled_width_widens_when_volatility_rises():
    residuals = np.full(20, 0.02)
    volatility = np.full(20, 0.01)
    calm = volatility_scaled_width(residuals, volatility, 0.01)
    stormy = volatility_scaled_width(residuals, volatility, 0.03)
    assert stormy > calm


def test_volatility_scaled_width_falls_back_without_volatility():
    residuals = np.full(20, 0.02)
    constant = conformal_width(residuals)
    assert volatility_scaled_width(residuals, np.full(20, np.nan), 0.01) == constant
    assert volatility_scaled_width(residuals, np.full(20, 0.01), np.nan) == constant


def test_volatility_scaling_is_clamped():
    """One freak window must not produce an absurd band."""
    residuals = np.full(20, 0.02)
    volatility = np.full(20, 0.01)
    extreme = volatility_scaled_width(residuals, volatility, 10.0)
    unclamped = extreme
    assert unclamped <= conformal_width(residuals / 0.01) * 0.01 * 3.0 + 1e-12
