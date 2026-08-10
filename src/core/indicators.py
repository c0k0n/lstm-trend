"""Technical indicators: SMA, EMA, RSI, MACD, Bollinger bands, crossovers."""

from __future__ import annotations

import numpy as np
import pandas as pd


def sma(close: pd.Series, window: int) -> pd.Series:
    return close.rolling(window).mean()


def ema(close: pd.Series, span: int) -> pd.Series:
    return close.ewm(span=span, adjust=False).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    # avg_loss == 0 -> rs = inf -> RSI 100; 0/0 on the first row -> NaN -> 50
    rs = avg_gain / avg_loss
    out = 100 - 100 / (1 + rs)
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
    cutoff = close.index[-1] - pd.Timedelta(days=365)
    recent = crosses[crosses["Date"] > cutoff]
    if len(recent):
        last = recent.iloc[-1]
        out["SMA 20/200 cross"] = (
            f"{last['Direction']} cross on {last['Date'].date()} (within the last year)"
        )
    return out
