"""Distribution analysis, stationarity tests, and autocorrelation."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import stats

_NULL_SIMS = 5_000
"""Monte Carlo draws behind the null distribution of the Dickey-Fuller statistic."""

_NULL_CHUNK = 500
"""Draws per batch, so memory stays flat no matter how long the history is."""

_NULL_SEED = 20260920
"""Fixed seed, so the same series always yields the same p-value."""


def _df_t_statistics(walks: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """t-statistic on the slope of ``dy = a + b * y_{-1}``, one per row.

    The same regression :func:`df_summary` runs, written with sufficient
    statistics so thousands of simulated series each cost a single pass.
    """
    y = walks[:, :-1]
    dy = np.diff(walks, axis=1)

    n = y.shape[1]
    sum_y = y.sum(axis=1)
    sum_yy = np.einsum("mk,mk->m", y, y)
    sum_dy = dy.sum(axis=1)
    sum_ydy = np.einsum("mk,mk->m", y, dy)
    sum_dydy = np.einsum("mk,mk->m", dy, dy)

    det = n * sum_yy - sum_y * sum_y
    slope = (n * sum_ydy - sum_y * sum_dy) / det
    intercept = (sum_yy * sum_dy - sum_y * sum_ydy) / det

    # RSS = S(dy^2) - a*S(dy) - b*S(y*dy);  (X'X)^-1_[2,2] = n / det
    rss = sum_dydy - intercept * sum_dy - slope * sum_ydy
    se = np.sqrt(rss / (n - 2) * (n / det))
    return slope / se


@lru_cache(maxsize=16)
def _null_t_statistics(n: int) -> npt.NDArray[np.float64]:
    """Null distribution of the Dickey-Fuller t-statistic at sample size ``n``.

    Under H0 the series is a driftless random walk, so we simulate the walk and
    run the very same regression on it. Deriving both the p-value and the
    critical values from this one sample keeps them in step with each other at
    the sample size we actually have — an asymptotic table would be a second,
    unchecked source of truth, and the response-surface coefficients this used
    to hardcode were inverted (p rose as evidence against the unit root grew).
    """
    rng = np.random.default_rng(_NULL_SEED)
    out = np.empty(_NULL_SIMS)
    for start in range(0, _NULL_SIMS, _NULL_CHUNK):
        stop = min(start + _NULL_CHUNK, _NULL_SIMS)
        walks = np.cumsum(rng.standard_normal((stop - start, n)), axis=1)
        out[start:stop] = _df_t_statistics(walks)
    return out


def return_moments(returns: pd.Series) -> dict[str, float]:
    return {
        "skewness": float(stats.skew(returns)) if len(returns) > 2 else 0.0,
        "kurtosis": float(stats.kurtosis(returns)) if len(returns) > 2 else 0.0,
        "std": float(returns.std(ddof=1)) if len(returns) > 1 else 0.0,
    }


def df_summary(close: pd.Series) -> dict[str, object]:
    """Dickey-Fuller test on the log price level.

    Runs Δy_t = α + β * y_{t-1} + ε_t with y the log price; H0 is β = 0, a unit
    root. Note this is the plain Dickey-Fuller regression — no lagged
    differences are included, so residual serial correlation is not corrected
    for. It is therefore *not* the augmented variant, and neither this function
    nor anything that reads it should be labelled "ADF".

    The p-value and the critical values both come from a simulated null
    distribution at this exact sample size, so they cannot contradict each
    other the way a response surface and a hardcoded table can.

    Raises:
        ValueError: if any non-null price is <= 0. The log transform is
            undefined there, and dropping those rows would tear the series out
            of time order and bias the test toward "stationary".
    """
    levels = close.dropna().to_numpy(dtype=float)
    if len(levels) < 5:
        return {
            "statistic": 0.0,
            "pvalue": 1.0,
            "critical_values": {},
            "stationary": False,
        }
    if levels.min() <= 0:
        raise ValueError(
            "df_summary needs strictly positive prices: the log transform is "
            "undefined at zero or below, and dropping those rows would break "
            f"the series' time ordering. Got minimum {levels.min()}."
        )

    y = np.log(levels)
    dy = np.diff(y)
    y_lag = y[:-1]

    # OLS: dy = α + β * y_lag + ε
    X = np.column_stack([np.ones(len(y_lag)), y_lag])
    beta_hat = np.linalg.lstsq(X, dy, rcond=None)[0]
    residuals = dy - X @ beta_hat
    n, k = X.shape
    sigma2 = np.dot(residuals, residuals) / (n - k)
    se_beta = np.sqrt((sigma2 * np.linalg.inv(X.T @ X))[1, 1])
    df_stat = beta_hat[1] / se_beta if se_beta > 0 else 0.0

    null = _null_t_statistics(len(y))
    pvalue = float(np.mean(null <= df_stat))

    return {
        "statistic": float(df_stat),
        "pvalue": pvalue,
        "critical_values": {
            "1%": float(np.quantile(null, 0.01)),
            "5%": float(np.quantile(null, 0.05)),
            "10%": float(np.quantile(null, 0.10)),
        },
        "stationary": pvalue < 0.05,
    }


def acf(returns: pd.Series, nlags: int = 20) -> pd.Series:
    """Autocorrelation of daily returns for lags 0..nlags."""
    x = returns.to_numpy(dtype=float)
    x = x - x.mean()
    var = np.dot(x, x)
    out = np.ones(nlags + 1)
    for k in range(1, nlags + 1):
        out[k] = np.dot(x[k:], x[:-k]) / var if var else 0.0
    return pd.Series(out, index=range(nlags + 1), name="ACF")
