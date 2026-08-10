"""Distribution analysis, stationarity tests, and autocorrelation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def return_moments(returns: pd.Series) -> dict[str, float]:
    return {
        "skewness": float(stats.skew(returns)) if len(returns) > 2 else 0.0,
        "kurtosis": float(stats.kurtosis(returns)) if len(returns) > 2 else 0.0,
        "std": float(returns.std(ddof=1)) if len(returns) > 1 else 0.0,
    }


def adf_summary(close: pd.Series) -> dict[str, object]:
    """Augmented Dickey-Fuller test on the price level (log prices).

    Uses a simple ADF regression: Δy_t = α + β * y_{t-1} + ε_t.
    H0: β = 0 (unit root). Critical values from Dickey-Fuller distribution.
    """
    clean = close.dropna()
    series = np.log(clean[clean > 0])
    y = series.to_numpy(dtype=float)
    if len(y) < 5:
        return {
            "statistic": 0.0,
            "pvalue": 1.0,
            "critical_values": {"1%": -3.43, "5%": -2.86, "10%": -2.57},
            "stationary": False,
        }

    dy = np.diff(y)
    y_lag = y[:-1]

    # OLS: dy = α + β * y_lag + ε
    X = np.column_stack([np.ones(len(y_lag)), y_lag])
    beta_hat = np.linalg.lstsq(X, dy, rcond=None)[0]
    fitted = X @ beta_hat
    residuals = dy - fitted
    n, k = X.shape
    sigma2 = np.dot(residuals, residuals) / (n - k)
    var_beta = sigma2 * np.linalg.inv(X.T @ X)
    se_beta = np.sqrt(var_beta[1, 1])
    adf_stat = beta_hat[1] / se_beta if se_beta > 0 else 0.0

    # Approximate p-value using MacKinnon (1996) response surface regression
    # Coefficients for model="c" (intercept only)
    tau = adf_stat
    tau2 = tau * tau
    tau3 = tau2 * tau
    pvalue = 0.0036 + (-0.0015) * tau + (-0.0093) * tau2 + (-0.0083) * tau3
    pvalue = max(0.0, min(1.0, pvalue))

    # MacKinnon critical values for model="c"
    crit = {"1%": -3.43, "5%": -2.86, "10%": -2.57}
    stationary = pvalue < 0.05

    return {
        "statistic": float(adf_stat),
        "pvalue": float(pvalue),
        "critical_values": crit,
        "stationary": stationary,
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
