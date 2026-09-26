"""Measure the dashboard's own forecast path, which was never scored.

Every number this project reports comes from `backtest.walk_forward`, and every
number in it is a *direct* forecast: one network pass produces the day-H return
and that is the prediction. The dashboard does something different and, by the
project's own description, weaker — `lstm_model.forecast_path` rolls forward one
business day at a time, feeding each predicted return into the window that
produces the next one.

So the number the app displays has never been the number the app was measured
on. That is the gap this module closes, and the honest expectation is that the
recursive path scores *worse*, because each step's error becomes the next
step's input.

Both arms run on identical windows, with the identical leakage rule and the
identical model. The only difference is direct-versus-recursive, which is the
only variable that makes this a measurement of anything.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np
import pandas as pd

from .features import forward_log_return
from .lstm_model import create_lstm_model, train_model

logger = logging.getLogger(__name__)

RECURSIVE = "LSTM + features (recursive)"
DIRECT = "LSTM + features (direct)"


def _factory(
    data: pd.DataFrame,
    horizon: int,
    length: int,
    epochs: int,
    random_seed: int,
    recursive: bool,
) -> Callable[[int, pd.Series], float]:
    """Build one forecaster: direct, or rolling forward one day at a time.

    Both arms use **return + calendar only**, matching the dashboard's own
    `forecast_path` input. That is the whole point of this module: the number
    that matters is the one the app draws, so the arm under test has to see the
    same columns the app feeds it.

    Using the full trailing-feature matrix here would be a different and easier
    experiment — and unmeasurable, because those features do not exist for a
    future day. A 60-day volatility at t+3 needs a close at t+3. So the
    recursive arm is restricted to the columns that *are* knowable ahead, and
    the direct arm is restricted identically so the two remain comparable.
    """
    from sklearn.preprocessing import MinMaxScaler

    from .preprocessing import calendar_frame

    close = data["Close"].astype(float)
    returns = np.log(close / close.shift(1)).to_numpy(dtype=float)
    calendar = calendar_frame(data.index)
    target = forward_log_return(close, horizon).to_numpy(dtype=float)

    # Column 0 is the return itself; the rest are the cyclic calendar. Row 0
    # has no prior close, so its return is NaN by construction — `complete`
    # has to test the *features*, not just the target, or that first row is
    # treated as usable and poisons the scaler fit, which then returns NaN for
    # every window downstream. That bug produced an all-NaN direct arm and was
    # invisible until the recursive and direct columns were compared.
    features = np.column_stack([returns, calendar])
    complete = (
        ~np.isnan(target)
        & ~np.isnan(features).any(axis=1)
        & (np.arange(len(features)) > 0)
    )
    if not complete.any():
        raise ValueError("no complete rows")

    n_features = features.shape[1]

    def predict(origin: int, train_close: pd.Series) -> float:
        if origin < length - 1:
            logger.warning(
                "Recursive path: origin %d is inside the %d-day lookback, so "
                "no complete window exists; falling back to naive.",
                origin,
                length,
            )
            return float(train_close.iloc[-1])
        if origin + horizon >= len(features):
            logger.warning(
                "Recursive path: origin %d plus a %d-day horizon runs past the "
                "end of the series; falling back to naive.",
                origin,
                horizon,
            )
            return float(train_close.iloc[-1])

        # Same leakage rule as every other arm: a row is only trainable once
        # its horizon-day outcome was known at the origin.
        allowed = np.zeros(len(features), dtype=bool)
        allowed[: origin - horizon + 1] = True
        rows = np.flatnonzero(complete & allowed)
        # The window ending at row p covers p-length+1 .. p, so the earliest
        # usable *end* row is `length`, not `length - 1`. Off by one here puts
        # row 0 inside a training window, and row 0's return is NaN by
        # construction — which poisons the whole fit and returns NaN for every
        # prediction. `x` having exactly one NaN is the fingerprint.
        rows = rows[rows >= length]
        if len(rows) < 40:
            logger.warning(
                "Recursive path: only %d usable training rows at origin %d, "
                "need 40; falling back to naive.",
                len(rows),
                origin,
            )
            return float(train_close.iloc[-1])

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(features[rows])
        scaled = scaler.transform(features)
        if np.isnan(scaled[rows]).any():
            logger.warning(
                "Recursive path: scaled training rows contain NaN at origin %d; "
                "falling back to naive.",
                origin,
            )
            return float(train_close.iloc[-1])

        x = np.stack([scaled[p - length + 1 : p + 1] for p in rows])
        y = target[rows].reshape(-1, 1)

        import keras

        keras.utils.set_random_seed(random_seed)
        model = create_lstm_model(
            input_shape=(length, n_features), units=64, dropout_rate=0.2, dense_units=25
        )
        train_model(
            model,
            x,
            y,
            epochs=epochs,
            batch_size=32,
            validation_split=0.1,
            patience=5,
        )

        if not recursive:
            window = scaled[origin - length + 1 : origin + 1].reshape(
                1, length, n_features
            )
            if np.isnan(window).any():
                logger.warning(
                    "Direct arm: prediction window at origin %d contains NaN; "
                    "falling back to naive.",
                    origin,
                )
                return float(train_close.iloc[-1])
            return float(
                train_close.iloc[-1]
                * np.exp(float(model.predict(window, verbose="0")[0, 0]))
            )

        # Recursive: roll forward one business day at a time, exactly as the
        # dashboard does. Each predicted return is appended to the window and
        # becomes tomorrow's input, so error compounds.
        current = scaled[origin - length + 1 : origin + 1].copy()
        last = float(train_close.iloc[-1])
        for step in range(horizon):
            batch = current.reshape(1, length, n_features)
            if np.isnan(batch).any():
                logger.warning(
                    "Recursive path: step %d window at origin %d contains NaN; "
                    "abandoning the roll and falling back to naive.",
                    step,
                    origin,
                )
                return float(train_close.iloc[-1])
            step_return = float(model.predict(batch, verbose="0")[0, 0])
            last *= np.exp(step_return)
            # The only unknown at t+1 is the return; the calendar is known.
            # That is exactly why the dashboard's path is restricted to these
            # columns, and why this arm can be measured at all.
            new_row = np.empty(n_features, dtype=float)
            new_row[0] = step_return
            new_row[1:] = calendar[origin + 1 + step]
            current = np.vstack([current[1:], new_row])
        return last

    return predict


def recursive_backtest(
    data: pd.DataFrame,
    horizon: int = 5,
    n_origins: int = 10,
    length: int = 60,
    epochs: int = 15,
    random_seed: int = 42,
) -> pd.DataFrame:
    """Score both arms on the same windows and return one row per origin.

    The direct arm is included in the same frame so the comparison is visible
    in one place. A recursive number on its own is uninterpretable — you cannot
    tell whether it is bad in absolute terms or bad *relative to the thing you
    should have compared it to*.
    """
    from .skill import BASELINE
    from .backtest import _origins, BacktestConfig

    close = data["Close"].astype(float)
    origins = _origins(len(close), BacktestConfig(horizon=horizon, n_origins=n_origins))
    if not origins:
        raise ValueError("not enough history to score either arm")

    direct = _factory(data, horizon, length, epochs, random_seed, recursive=False)
    recursive = _factory(data, horizon, length, epochs, random_seed, recursive=True)

    rows = []
    for origin in origins:
        train_close = close.iloc[: origin + 1]
        actual = float(close.iloc[origin + horizon])
        rows.append(
            {
                "origin": close.index[origin],
                "target": close.index[origin + horizon],
                "actual": actual,
                BASELINE: float(train_close.iloc[-1]),
                DIRECT: direct(origin, train_close),
                RECURSIVE: recursive(origin, train_close),
            }
        )
    return pd.DataFrame(rows)
