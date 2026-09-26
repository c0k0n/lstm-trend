"""Walk-forward backtesting: score a model across many windows, not one.

A single train/test split produces one number, and one number cannot support a
verdict — you cannot tell a model that learned something from a model that got
lucky. This rolls the origin forward through history and scores every window,
so the result is a distribution you can actually reason about.

The rule that makes it honest: at every origin, the model may only see rows
whose outcome was already known by that date. For a `horizon`-day-ahead target
that means training rows must sit at least `horizon` days before the origin.
Get this wrong by one row and every model looks brilliant.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .features import build_features, forward_log_return
from .skill import BASELINE

logger = logging.getLogger(__name__)

MOVING_AVERAGE = "Moving average"
LIGHTGBM = "LightGBM"
LSTM_UPGRADED = "LSTM + features"
PANEL = "LightGBM (panel)"

_WARMUP = 180
"""Days before the earliest usable origin.

Three costs stack. The slowest feature (60-day volatility) does not exist until
day 60. A 60-day input window then needs 60 more days behind the origin, so no
window is complete before day 119. And the first origin still needs enough
complete windows behind it to train on. Sixty was enough while the LSTM read
only raw prices; once it reads features, windows reach back into rows that are
still NaN and the whole run silently returns NaN.
"""


@dataclass
class BacktestConfig:
    horizon: int = 5
    n_origins: int = 12
    include_upgraded_lstm: bool = False
    include_panel: bool = False
    use_ohlcv: bool = False
    """Feed the candle-structure columns to the point-forecast models.

    **Measured, and off by default because it made things worse.** 200 windows,
    same network, paired per ticker: close-only −11.5% vs +OHLCV −18.9%, a
    mean delta of −7.4 points with the candle helping on only 6 of 20 tickers
    (t p = 0.135, Wilcoxon p = 0.083). More information about the past is not
    more information about the future.

    It stays available because the candle earns its place somewhere else —
    Parkinson and Garman-Klass range estimators are far better measures of
    *volatility* than close-to-close, and the band is where that belongs.
    """
    sequence_length: int = 60
    lstm_epochs: int = 20
    random_seed: int = 42


def _origins(n_rows: int, config: BacktestConfig) -> list[int]:
    """Evenly spaced origin positions, always leaving a full horizon ahead.

    The earliest usable origin needs `_WARMUP` days of features behind it and
    `horizon` days of prices ahead of it to settle against.
    """
    first = _WARMUP + config.horizon
    last = n_rows - 1 - config.horizon
    if last <= first:
        return []
    count = min(config.n_origins, last - first + 1)
    if count <= 1:
        return [last]
    step = (last - first) / (count - 1)
    return sorted({first + round(i * step) for i in range(count)})


def _naive(train_close: pd.Series) -> float:
    return float(train_close.iloc[-1])


def _moving_average(train_close: pd.Series, window: int = 20) -> float:
    return float(train_close.iloc[-window:].mean())


# --- LightGBM configuration ------------------------------------------------
# Two settings, not one, and the difference is deliberate. The per-ticker
# model sees a few thousand rows for a single name, so it is capped harder
# (200 trees, min 20 rows per leaf) to stop it memorising. The panel model
# pools every ticker and can afford more capacity (300 trees, min 40). They
# used to be copy-pasted inline with the difference unexplained, which is how
# the two silently converged once.


def _lightgbm(*, panel: bool) -> Any:
    """The LightGBM regressor, per-ticker or panel."""
    import lightgbm as lgb

    if panel:
        return lgb.LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=15,
            min_child_samples=40,
            subsample=0.9,
            subsample_freq=1,
            colsample_bytree=0.9,
            random_state=42,
            verbose=-1,
        )
    return lgb.LGBMRegressor(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=15,
        min_child_samples=20,
        subsample=0.9,
        subsample_freq=1,
        colsample_bytree=0.9,
        random_state=42,
        verbose=-1,
    )


def _lightgbm_factory(
    data: pd.DataFrame,
    horizon: int,
    context: pd.DataFrame | None = None,
    ohlcv: bool = True,
) -> Callable[[int, pd.Series], float]:
    """A forecaster that refits LightGBM at each origin on features only.

    Features and targets are built once over the whole history, then masked per
    origin so the fit never sees a row whose outcome post-dates the origin.
    """
    close = data["Close"].astype(float)
    features = build_features(data, context, ohlcv=ohlcv)
    target = forward_log_return(close, horizon)
    usable = features.notna().all(axis=1) & target.notna()

    def predict(origin: int, train_close: pd.Series) -> float:
        # Only rows whose outcome was known `horizon` days before the origin.
        trainable = usable.copy()
        trainable.iloc[origin - horizon + 1 :] = False
        rows = np.flatnonzero(trainable.to_numpy())
        if not rows.size:
            # A feature column that is entirely NaN empties every row, and
            # falling back to naive then looks exactly like a model that chose
            # to repeat yesterday. Say so instead.
            logger.warning(
                "No complete feature rows at origin %d; LightGBM cannot train "
                "and will fall back to naive.",
                origin,
            )
        if len(rows) < 40:
            return _naive(train_close)

        model = _lightgbm(panel=False)
        model.fit(
            features.iloc[rows].to_numpy(dtype=float),
            target.iloc[rows].to_numpy(dtype=float),
        )
        predicted_return = float(
            model.predict(features.iloc[[origin]].to_numpy(dtype=float))[0]
        )
        return float(train_close.iloc[-1] * np.exp(predicted_return))

    return predict


def _lstm_upgraded_factory(
    data: pd.DataFrame, config: BacktestConfig, context: pd.DataFrame | None = None
) -> Callable[[int, pd.Series], float]:
    """The modernised LSTM: covariates in, log-returns out, horizon direct.

    Four changes over the original raw-price network, each aimed at a known
    weakness:

    - **Log-return target.** Price levels carry a unit root (this project's own
      Dickey-Fuller test says so); log-returns are roughly stationary, which is
      what a regression actually wants.
    - **Covariates.** Volume, volatility, trend distance, RSI and calendar, so
      the network sees the state of the market and not just one price line.
    - **Direct horizon.** One prediction for day H, no chaining of guesses.
    - **Layer normalisation** after each recurrent layer, which stabilises
      training on sequences far better than dropout alone.
    """
    import os

    os.environ.setdefault("KERAS_BACKEND", "torch")

    from sklearn.preprocessing import MinMaxScaler

    from .lstm_model import _keras, create_lstm_model, train_model

    _keras()[0].utils.set_random_seed(config.random_seed)

    length = config.sequence_length
    horizon = config.horizon
    features = build_features(data, context, ohlcv=config.use_ohlcv).to_numpy(
        dtype=float
    )
    target = forward_log_return(data["Close"].astype(float), horizon).to_numpy(
        dtype=float
    )
    complete = ~np.isnan(features).any(axis=1)
    first_complete = int(np.argmax(complete))
    min_position = first_complete + length - 1
    """Earliest row whose entire input window is free of NaN.

    Checking the row alone is not enough: a window ending at row 60 starts at
    row 1, which has no 60-day volatility yet.
    """
    usable = complete & ~np.isnan(target)
    n_features = features.shape[1]

    def predict(origin: int, train_close: pd.Series) -> float:
        if origin < min_position or origin - length + 1 < 0:
            return _naive(train_close)

        # Rows whose outcome was known `horizon` days before the origin.
        allowed = np.zeros(len(features), dtype=bool)
        allowed[: origin - horizon + 1] = True
        rows = np.flatnonzero(usable & allowed)
        rows = rows[rows >= min_position]
        if not rows.size:
            logger.warning(
                "No complete feature window at origin %d; the upgraded LSTM "
                "cannot train and will fall back to naive.",
                origin,
            )
        if len(rows) < 40:
            return _naive(train_close)

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(features[rows])
        scaled = scaler.transform(features)

        x = np.stack([scaled[p - length + 1 : p + 1] for p in rows])
        y = target[rows].reshape(-1, 1)

        # The same trunk and the same early-stopping configuration the
        # dashboard and the quantile model use. This arm was a third
        # hand-written copy of the same network; three copies is how they
        # drift.
        net = create_lstm_model(
            input_shape=(length, n_features), units=64, dropout_rate=0.2, dense_units=25
        )
        train_model(
            net,
            x,
            y,
            epochs=config.lstm_epochs,
            batch_size=32,
            validation_split=0.1,
            patience=5,
        )
        window = scaled[origin - length + 1 : origin + 1].reshape(1, length, n_features)
        if np.isnan(window).any():
            # A NaN would silently poison the whole run rather than fail loudly,
            # so it is named in the log rather than quietly becoming a naive
            # forecast that looks exactly like a model decision.
            logger.warning(
                "Upgraded LSTM: prediction window at origin %d contains NaN; "
                "falling back to naive.",
                origin,
            )
            return _naive(train_close)

        predicted_return = float(net.predict(window, verbose="0")[0, 0])
        return float(train_close.iloc[-1] * np.exp(predicted_return))

    return predict


def _panel_factory(
    all_data: dict[str, pd.DataFrame],
    horizon: int,
    contexts: dict[str, pd.DataFrame] | None = None,
    ohlcv: bool = True,
) -> Callable[[int, pd.Series, str], float]:
    """One model trained on every ticker at once, rather than one each.

    A single stock gives a couple of thousand rows, which is thin for a model
    with any capacity. Ten stocks give twenty thousand. This is where most of
    the modern gains in forecasting come from — not from a cleverer
    architecture but from seeing many more series.

    Alignment is by date, not position: the cutoff is the origin's date, and
    every ticker contributes only rows whose outcome was known by then.
    """
    prepared = {
        symbol: (
            build_features(frame, (contexts or {}).get(symbol), ohlcv=ohlcv),
            forward_log_return(frame["Close"].astype(float), horizon),
        )
        for symbol, frame in all_data.items()
    }

    widths = {features.shape[1] for features, _ in prepared.values()}
    if len(widths) > 1:
        # Stacking frames of different widths either raises or, worse,
        # silently misaligns columns across tickers. Context is all or nothing.
        raise ValueError("Panel features differ in width; give every ticker context.")

    def predict(origin: int, train_close: pd.Series, symbol: str) -> float:
        cutoff = train_close.index[origin]

        blocks_x = []
        blocks_y = []
        for name, (features, target) in prepared.items():
            usable = (features.notna().all(axis=1) & target.notna()).to_numpy()
            position = int(features.index.searchsorted(cutoff, side="right")) - 1
            if position < horizon:
                continue
            keep = usable[: position - horizon + 1]
            rows = np.flatnonzero(keep)
            if len(rows):
                blocks_x.append(features.iloc[rows].to_numpy(dtype=float))
                blocks_y.append(target.iloc[rows].to_numpy(dtype=float))

        if not blocks_x:
            logger.warning(
                "No complete feature rows for any ticker at %s; the panel "
                "model cannot train and will fall back to naive.",
                cutoff,
            )
        if not blocks_x or sum(len(b) for b in blocks_x) < 200:
            total = sum(len(b) for b in blocks_x) if blocks_x else 0
            logger.warning(
                "Panel model: only %d pooled training rows at %s, need 200; "
                "falling back to naive.",
                total,
                cutoff.date(),
            )
            return _naive(train_close)

        model = _lightgbm(panel=True)
        model.fit(np.vstack(blocks_x), np.concatenate(blocks_y))

        own_features = prepared[symbol][0]
        row = own_features.iloc[[origin]].to_numpy(dtype=float)
        return float(train_close.iloc[-1] * np.exp(float(model.predict(row)[0])))

    return predict


def walk_forward(
    data: pd.DataFrame,
    config: BacktestConfig | None = None,
    on_origin: Callable[[int, int], None] | None = None,
    panel_data: dict[str, pd.DataFrame] | None = None,
    symbol: str = "",
    contexts: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Score every model at every origin and return one row per window.

    `contexts` maps ticker to the market context frame from
    `market_context.py`. Pass it and every model reads those columns; omit it
    and the run reproduces the no-context baseline exactly.

    Raises ValueError when the history is too short for even one window.
    """
    config = config or BacktestConfig()
    close = data["Close"].astype(float)
    origins = _origins(len(close), config)
    if not origins:
        raise ValueError(
            f"Not enough history for a {config.horizon}-day walk-forward test "
            f"({len(close)} rows). Extend the date range."
        )

    own_context: pd.DataFrame | None = None
    if contexts:
        if symbol not in contexts:
            # Silently falling back to no context would produce a run that
            # looks like the covariate arm and is actually the control.
            raise ValueError(
                f"No market context supplied for {symbol!r}; pass one for "
                "every ticker or none at all."
            )
        own_context = contexts[symbol]

    lightgbm_predict = _lightgbm_factory(
        data, config.horizon, own_context, ohlcv=config.use_ohlcv
    )
    upgraded_predict = (
        _lstm_upgraded_factory(data, config, own_context)
        if config.include_upgraded_lstm
        else None
    )
    panel_predict = (
        _panel_factory(panel_data, config.horizon, contexts, ohlcv=config.use_ohlcv)
        if config.include_panel and panel_data
        else None
    )

    rows = []
    for step, origin in enumerate(origins):
        if on_origin is not None:
            on_origin(step + 1, len(origins))

        train_close = close.iloc[: origin + 1]
        row = {
            "origin": close.index[origin],
            "target": close.index[origin + config.horizon],
            "actual": float(close.iloc[origin + config.horizon]),
            BASELINE: _naive(train_close),
            MOVING_AVERAGE: _moving_average(train_close),
            LIGHTGBM: lightgbm_predict(origin, train_close),
        }
        if upgraded_predict is not None:
            row[LSTM_UPGRADED] = upgraded_predict(origin, train_close)
        if panel_predict is not None:
            row[PANEL] = panel_predict(origin, train_close, symbol)
        rows.append(row)

    return pd.DataFrame(rows)
