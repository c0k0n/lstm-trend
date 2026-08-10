"""Risk metrics: Sharpe, Sortino, drawdowns, VaR, CVaR."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .returns import (
    TRADING_DAYS,
    daily_returns,
    cumulative_return,
    annualized_return,
    annualized_volatility,
)


def sharpe_ratio(
    returns: pd.Series, risk_free_rate: float = 0.0, periods: int = TRADING_DAYS
) -> float:
    """Annualized Sharpe ratio (excess return / volatility)."""
    if len(returns) < 2 or returns.std(ddof=1) <= 1e-12:
        return 0.0
    excess = returns.mean() - risk_free_rate / periods
    return float(excess / returns.std(ddof=1) * np.sqrt(periods))


def sortino_ratio(
    returns: pd.Series, risk_free_rate: float = 0.0, periods: int = TRADING_DAYS
) -> float:
    """Annualized Sortino ratio (excess return / downside deviation)."""
    downside = returns[returns < risk_free_rate / periods]
    if len(downside) < 2 or downside.std(ddof=1) == 0:
        return 0.0
    excess = returns.mean() - risk_free_rate / periods
    return float(excess / downside.std(ddof=1) * np.sqrt(periods))


def max_drawdown(close: pd.Series) -> float:
    """Largest peak-to-trough decline as a negative fraction."""
    running_max = close.cummax()
    return float((close / running_max - 1).min()) if len(close) > 1 else 0.0


def drawdown_series(close: pd.Series) -> pd.Series:
    """Drawdown at each point relative to the running maximum."""
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
    """Fraction of trading days with a positive return."""
    return float((returns > 0).mean()) if len(returns) else 0.0


def drawdown_events(close: pd.Series, min_depth: float = 0.05) -> pd.DataFrame:
    """List the worst drawdown episodes (depth >= min_depth), most recent first."""
    dd = drawdown_series(close)
    events: list[dict[str, object]] = []
    start: pd.Timestamp | None = None
    trough: pd.Timestamp | None = None
    trough_val = 0.0
    in_dd = False
    for idx, val in zip(dd.index, dd.values):
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
        last_index = close.index[-1]
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


def comparison_frame(close: pd.Series) -> dict[str, float]:
    """Aggregate comparison metrics (return, risk, drawdown) for a single ticker."""
    returns = daily_returns(close)
    return {
        "Total return": cumulative_return(close),
        "CAGR": annualized_return(close),
        "Ann. volatility": annualized_volatility(returns),
        "Sharpe": sharpe_ratio(returns),
        "Sortino": sortino_ratio(returns),
        "Max drawdown": max_drawdown(close),
        "VaR 95% (daily)": value_at_risk(returns),
        "CVaR 95% (daily)": conditional_var(returns),
        "Positive days": positive_day_ratio(returns),
    }
