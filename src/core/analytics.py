"""Exploratory data analysis helpers: returns, risk metrics, drawdowns,
seasonality, stationarity tests, and technical indicators. All pure pandas /
numpy / scipy — no Streamlit imports.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd
from scipy import stats

TRADING_DAYS: int = 252


# --------------------------------------------------------------------------- #
# Returns and performance
# --------------------------------------------------------------------------- #
def daily_returns(close: pd.Series) -> pd.Series:
    return close.pct_change().dropna()


def cumulative_return(close: pd.Series) -> float:
    return float(close.iloc[-1] / close.iloc[0] - 1) if len(close) > 1 else 0.0


def annualized_return(close: pd.Series) -> float:
    """Compound annual growth rate from first to last close."""
    if len(close) < 2:
        return 0.0
    first = cast(pd.Timestamp, close.index[0])
    last = cast(pd.Timestamp, close.index[-1])
    years = (last - first).days / 365.25
    if years <= 0:
        return 0.0
    return float((close.iloc[-1] / close.iloc[0]) ** (1 / years) - 1)


def annualized_volatility(returns: pd.Series, periods: int = TRADING_DAYS) -> float:
    return float(returns.std(ddof=1) * np.sqrt(periods)) if len(returns) > 1 else 0.0


def sharpe_ratio(
    returns: pd.Series, risk_free_rate: float = 0.0, periods: int = TRADING_DAYS
) -> float:
    if len(returns) < 2 or returns.std(ddof=1) <= 1e-12:
        return 0.0
    excess = returns.mean() - risk_free_rate / periods
    return float(excess / returns.std(ddof=1) * np.sqrt(periods))


def sortino_ratio(
    returns: pd.Series, risk_free_rate: float = 0.0, periods: int = TRADING_DAYS
) -> float:
    downside = returns[returns < risk_free_rate / periods]
    if len(downside) < 2 or downside.std(ddof=1) == 0:
        return 0.0
    excess = returns.mean() - risk_free_rate / periods
    return float(excess / downside.std(ddof=1) * np.sqrt(periods))


def max_drawdown(close: pd.Series) -> float:
    running_max = close.cummax()
    return float((close / running_max - 1).min()) if len(close) > 1 else 0.0


def drawdown_series(close: pd.Series) -> pd.Series:
    return close / close.cummax() - 1


def value_at_risk(returns: pd.Series, alpha: float = 0.95) -> float:
    """Historical VaR: the alpha-quantile loss, as a negative number."""
    if len(returns) == 0:
        return 0.0
    return float(np.percentile(returns, (1 - alpha) * 100))


def conditional_var(returns: pd.Series, alpha: float = 0.95) -> float:
    """Expected shortfall / CVaR: mean loss beyond the VaR cutoff."""
    var = value_at_risk(returns, alpha)
    tail = returns[returns <= var]
    return float(tail.mean()) if len(tail) else var


def positive_day_ratio(returns: pd.Series) -> float:
    return float((returns > 0).mean()) if len(returns) else 0.0


def rolling_volatility(
    returns: pd.Series, window: int = 20, periods: int = TRADING_DAYS
) -> pd.Series:
    return returns.rolling(window).std(ddof=1) * np.sqrt(periods)


def period_returns(close: pd.Series) -> dict[str, float]:
    """Trailing returns over common horizons, in percent."""
    last_date = cast(pd.Timestamp, close.index[-1])
    today = close.iloc[-1]
    out: dict[str, float] = {}
    for label, bdays in (("1M", 21), ("3M", 63), ("6M", 126), ("1Y", 252)):
        past = cast(float, close.asof(last_date - pd.offsets.BDay(bdays)))
        out[label] = float((today / past - 1) * 100) if not np.isnan(past) else np.nan

    year_start = cast(
        float, close.asof(pd.Timestamp(year=last_date.year, month=1, day=1))
    )
    out["YTD"] = (
        float((today / year_start - 1) * 100) if not np.isnan(year_start) else np.nan
    )
    return out


def position_in_52w_range(close: pd.Series, window: int = 252) -> float:
    """Where today's price sits between the 52-week low and high, 0-1."""
    recent = close.iloc[-window:]
    lo, hi = recent.min(), recent.max()
    if lo == hi:
        return 1.0
    return float((close.iloc[-1] - lo) / (hi - lo))


def drawdown_events(close: pd.Series, min_depth: float = 0.05) -> pd.DataFrame:
    """List the worst drawdown episodes (depth >= min_depth), most recent first."""
    dd = drawdown_series(close)
    events: list[dict[str, object]] = []
    start: pd.Timestamp | None = None
    trough: pd.Timestamp | None = None
    trough_val = 0.0
    in_dd = False
    for raw_idx, val in zip(dd.index, dd.values):
        idx = cast(pd.Timestamp, pd.Timestamp(raw_idx))
        if not in_dd and val < 0:
            in_dd, start, trough, trough_val = True, idx, idx, val
        elif in_dd:
            if val < trough_val:
                trough, trough_val = idx, val
            elif val >= 0 and start is not None and trough is not None:
                events.append(
                    {
                        "Start": start,
                        "Trough": trough,
                        "End": idx,
                        "Depth": trough_val,
                        "Duration (days)": (idx - start).days,
                    }
                )
                in_dd = False
                start, trough, trough_val = None, None, 0.0
    if in_dd and start is not None and trough is not None:
        last_index = cast(pd.Timestamp, close.index[-1])
        events.append(
            {
                "Start": start,
                "Trough": trough,
                "End": last_index,
                "Depth": trough_val,
                "Duration (days)": (last_index - start).days,
            }
        )
    df = pd.DataFrame(events)
    if len(df):
        df = df[df["Depth"] <= -min_depth].sort_values(by="Start", ascending=False)
    return df


# --------------------------------------------------------------------------- #
# Distribution and stationarity
# --------------------------------------------------------------------------- #
def return_moments(returns: pd.Series) -> dict[str, float]:
    return {
        "skewness": float(stats.skew(returns)) if len(returns) > 2 else 0.0,
        "kurtosis": float(stats.kurtosis(returns)) if len(returns) > 2 else 0.0,
        "std": float(cast(float, returns.std(ddof=1))) if len(returns) > 1 else 0.0,
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


# --------------------------------------------------------------------------- #
# Seasonality
# --------------------------------------------------------------------------- #
def weekday_effects(returns: pd.Series) -> pd.DataFrame:
    dt = pd.Series(pd.DatetimeIndex(returns.index))
    df = pd.DataFrame(
        {
            "return": returns.values,
            "weekday": dt.dt.dayofweek.values,
        }
    )
    grouped = df.groupby("weekday")["return"]
    table = pd.DataFrame(
        {
            "mean": grouped.mean(),
            "hit_rate": grouped.apply(lambda s: (s > 0).mean()),
            "count": grouped.count(),
        }
    )
    # A weekday can be missing entirely (e.g. a holiday week); reindex keeps
    # the table at five rows with NaN stats and a zero count for that day.
    table = table.reindex(range(5))
    table["count"] = table["count"].fillna(0)
    table.index = ["Mon", "Tue", "Wed", "Thu", "Fri"]
    return table.rename_axis("weekday").reset_index()


def monthly_returns_matrix(close: pd.Series) -> pd.DataFrame:
    """Year x month grid of monthly returns, for a heatmap."""
    monthly = close.resample("ME").last().pct_change() * 100
    df = monthly.dropna().to_frame("return")
    dt = pd.Series(pd.DatetimeIndex(df.index))
    df["year"] = dt.dt.year.values
    df["month"] = dt.dt.month.values
    pivot = df.pivot(index="year", columns="month", values="return")
    pivot = pivot.reindex(columns=range(1, 13))
    pivot.columns = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]
    return pivot


def month_effects(close: pd.Series) -> pd.DataFrame:
    """Average return, hit rate and count per calendar month.

    Short ranges may not contain every month (and a missing month would
    otherwise shift the labels); the table is reindexed to all twelve
    months so each row always maps to the right calendar month.
    """
    monthly = close.resample("ME").last().pct_change().dropna()
    table = monthly.groupby(monthly.index.month).agg(
        mean="mean", hit_rate=lambda s: (s > 0).mean(), count="count"
    )
    table = table.reindex(range(1, 13))
    table["count"] = table["count"].fillna(0)
    table.index = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]
    return table.rename_axis("month").reset_index()


# --------------------------------------------------------------------------- #
# Technical indicators
# --------------------------------------------------------------------------- #
def sma(close: pd.Series, window: int) -> pd.Series:
    return cast(pd.Series, close.rolling(window).mean())


def ema(close: pd.Series, span: int) -> pd.Series:
    return cast(pd.Series, close.ewm(span=span, adjust=False).mean())


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    # avg_loss == 0 -> rs = inf -> RSI 100; 0/0 on the first row -> NaN -> 50
    rs = avg_gain / avg_loss
    out = cast(pd.Series, 100 - 100 / (1 + rs))
    return out.fillna(50)


def macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.DataFrame:
    line = ema(close, fast) - ema(close, slow)
    return pd.DataFrame(
        {"MACD": line, "Signal": line.ewm(span=signal, adjust=False).mean()}
    ).assign(Histogram=lambda d: d["MACD"] - d["Signal"])


def bollinger_bands(
    close: pd.Series, window: int = 20, num_std: float = 2.0
) -> pd.DataFrame:
    mid = close.rolling(window).mean()
    std = close.rolling(window).std(ddof=0)
    return pd.DataFrame(
        {"Mid": mid, "Upper": mid + num_std * std, "Lower": mid - num_std * std}
    )


def crossover_dates(fast: pd.Series, slow: pd.Series) -> pd.DataFrame:
    """Dates where the fast line crosses the slow line, with direction."""
    diff = fast - slow
    sign = np.sign(diff)
    change = sign.diff().fillna(0) != 0
    out = pd.DataFrame(
        {
            "Date": diff.index[change],
            "Direction": np.where(sign[change] > 0, "Golden", "Death"),
        }
    )
    return out


def latest_signals(close: pd.Series) -> dict[str, str]:
    """Human-readable snapshot of the indicator state."""
    out: dict[str, str] = {}
    r = rsi(close)
    if not r.dropna().empty:
        val = r.dropna().iloc[-1]
        state = "overbought" if val >= 70 else "oversold" if val <= 30 else "neutral"
        out["RSI (14)"] = f"{val:.1f} — {state}"
    bb = bollinger_bands(close).dropna()
    if len(bb):
        price = close.iloc[-1]
        if price >= bb["Upper"].iloc[-1]:
            band = "touching the upper band"
        elif price <= bb["Lower"].iloc[-1]:
            band = "touching the lower band"
        else:
            band = "between the bands"
        out["Bollinger (20, 2σ)"] = band
    m = macd(close).dropna()
    if len(m):
        side = "bullish" if m["MACD"].iloc[-1] > m["Signal"].iloc[-1] else "bearish"
        out["MACD"] = (
            f"{side} (MACD {'above' if side == 'bullish' else 'below'} signal)"
        )
    s20, s200 = sma(close, 20), sma(close, 200)
    if not s200.dropna().empty:
        trend = "above" if close.iloc[-1] > s200.iloc[-1] else "below"
        out["SMA 200"] = (
            f"price {trend} the 200-day average (long-term {'uptrend' if trend == 'above' else 'downtrend'})"
        )
    crosses = crossover_dates(s20, s200)
    cutoff = cast(pd.Timestamp, close.index[-1]) - pd.Timedelta(days=365)
    recent = crosses[crosses["Date"] > cutoff]
    if len(recent):
        last = recent.iloc[-1]
        out["SMA 20/200 cross"] = (
            f"{last['Direction']} cross on {last['Date'].date()} (within the last year)"
        )
    return out


# --------------------------------------------------------------------------- #
# Comparison
# --------------------------------------------------------------------------- #
def normalize_series(close: pd.Series, base: float = 100.0) -> pd.Series:
    return close / close.iloc[0] * base


def comparison_frame(close: pd.Series) -> dict[str, float]:
    returns = daily_returns(close)
    ann_ret = annualized_return(close)
    ann_vol = annualized_volatility(returns)
    sharpe = sharpe_ratio(returns)
    sortino = sortino_ratio(returns)
    return {
        "Total return": cumulative_return(close),
        "CAGR": ann_ret,
        "Ann. volatility": ann_vol,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Max drawdown": max_drawdown(close),
        "VaR 95% (daily)": value_at_risk(returns),
        "CVaR 95% (daily)": conditional_var(returns),
        "Positive days": positive_day_ratio(returns),
    }
