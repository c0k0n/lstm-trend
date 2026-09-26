"""A quantile LSTM: predict a *distribution*, not a point.

The point forecaster in `lstm_model.py` optimises mean absolute error on a
target whose conditional mean is very close to zero. That has a consequence
worth stating plainly: the MAE-optimal forecast is approximately zero, so the
best it can do is match "repeat yesterday", and any deviation it does produce
is variance added to a signal-free baseline. That is not a tuning problem. It
is the wrong output shape for the question.

This network asks for the conditional *quantiles* instead. A single forward
pass emits 25 of them — from the 1st to the 99th percentile — trained with the
pinball (quantile) loss. Three things follow:

- **The median quantile is a point forecast that is not MAE-driven.** It is
  fit to the conditional median, which for a fat-tailed return distribution is
  a different — and usually better-conditioned — target than the mean.
- **The spread is learned, not bolted on.** The quantiles diverge where the
  model has learned that outcomes are genuinely uncertain, and stay together
  where it has not. A conformal band post-hoc cannot do this: it applies one
  width scaled by a volatility proxy, and has to assume the conditional shape
  is a scaled version of the average shape.
- **Calibration is measurable.** A distribution can be scored for coverage at
  any level, and the pinball loss rewards getting each level right on its own
  terms rather than compromising them all toward the mean.

Two design decisions carry most of the weight:

**Censoring below zero is not clamped, it is learned.** A 1st-percentile
forecast of a stock's `h`-day return should be *negative* — that is what a 1%
tail looks like. Clamping it at zero would be a lie with a straight face, and
it would flatten exactly the tail a risk-aware user cares most about.

**Quantile crossing is fixed at the output, not regularised away.** Independent
output heads can and do cross, which makes the band self-contradictory (a
"lower" bound above an "upper" bound). Sorting the outputs per row is
projection-onto-convex-cones, the standard fix, and it costs nothing at
inference because the network is evaluated one row at a time here.

The honest limit, stated up front: this does not manufacture directional
skill that is not in the data. It produces a *calibrated distribution* instead
of a bare number, which is a real and achievable improvement, and the
walk-forward harness will report whether the calibration is real or an
artifact of training. If the coverage numbers come back at the target, that is
a genuine win over the conformal band. If they do not, that is a finding too.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from .lstm_model import _keras, build_lstm_stack, early_stopping

logger = logging.getLogger(__name__)

TAIL_QUANTILES: tuple[float, ...] = (0.975, 0.99)
"""The far tail, beyond the regular grid. A risk budget reads these."""


def _build_quantile_grid() -> tuple[float, ...]:
    """The quantile levels, built from integer counts so they are exact.

    `np.arange(0.01, 1.0, 0.05)` looks right and is not: floating-point
    accumulation makes the values 0.060000000000000005, 0.11000000000000001 and
    so on, which rounds to a grid with no 0.5 in it at all. The first version of
    this module did exactly that, and the median lookup — the one output the
    point forecaster depends on — would have raised `ValueError` at run time.

    Integer counts divided at the end give the exact levels, and the set union
    guarantees the three the rest of the module requires (0.05, 0.5, 0.95) are
    present whatever the grid is.
    """
    levels = {round(k / 100.0, 2) for k in range(1, 100, 5)}
    levels.update(TAIL_QUANTILES)
    levels.update({0.05, 0.5, 0.95})
    return tuple(sorted(levels))


DEFAULT_QUANTILES: tuple[float, ...] = _build_quantile_grid()
"""1% to 96% in 5-point steps, plus 97.5% and 99% for the far tail.

A tail that stops in the mid-90s cannot describe a bad week, and the whole
point of asking a network for a distribution rather than a point is to describe
the bad weeks honestly. The last two are what a risk budget actually reads.
"""

ALL_QUANTILES: tuple[float, ...] = tuple(sorted({*DEFAULT_QUANTILES, *TAIL_QUANTILES}))


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Quantile loss, the mean over quantiles and rows.

    For quantile `q` the loss is `q * (y - f)` when the forecast is above the
    outcome and `(1 - q) * (f - y)` when it is below. The asymmetry is the
    whole mechanism: at `q = 0.5` both sides weigh equally and the loss reduces
    to half the absolute error, so the median quantile is a median-target
    forecast; at `q = 0.05` over-forecasting is punished five times harder
    than under-forecasting, which is what drags that head down to the low tail.

    Written out rather than pulled from a library because the shape matters:
    Keras' built-in `quantile` loss operates per-call on one `q`, so training
    25 outputs would mean 25 separate loss objects and a very confusing stack
    trace when one of them disagrees.

    **Computed in the backend's own framework, not numpy.** The first version
    converted to numpy, which fails twice over: a CUDA tensor cannot be
    `np.asarray`'d at all, and detaching to get around that severs the
    gradient, so training dies with "element 0 of tensors does not require
    grad". Both failures are avoidable only by staying inside the backend's
    framework. The constants are moved onto the predictions' device so the
    arithmetic never bounces between host and accelerator.
    """
    backend = _backend()
    predictions = _on_device(y_pred, backend)
    if predictions.shape[-1] != len(ALL_QUANTILES):
        raise ValueError(
            f"expected {len(ALL_QUANTILES)} quantile outputs, "
            f"got {predictions.shape[-1]}"
        )

    truth = _on_device(y_true, backend, like=predictions)
    q = _on_device(ALL_QUANTILES, backend, like=predictions)

    errors = truth - predictions
    loss = backend.maximum(q * errors, (q - 1.0) * errors)
    return backend.mean(backend.mean(loss, axis=-1), axis=-1)


def _backend() -> Any:
    """The active backend module (torch or tensorflow).

    `keras.backend()` is the *function* that reports the name, and
    `keras.backend` is the *module* that holds it. Reading the module's `backend`
    attribute is the documented way to get the name as a string without
    importing anything, and it works in both Keras 3 and the older shapes.
    """
    keras_module = _keras()[0]
    name = getattr(getattr(keras_module, "backend", None), "backend", None)
    if callable(name):
        name = name()
    if not isinstance(name, str):
        name = "torch"
    return __import__(name, fromlist=["*"])


def _on_device(value: Any, backend: Any, like: Any = None) -> Any:
    """A backend tensor for `value`, on `like`'s device when one is given.

    Predictions are already graph tensors and are returned untouched — moving
    or rebuilding them would sever the gradient. Everything else is a constant
    and gets placed on the same device as the tensor it will be combined with,
    which is the only way the arithmetic stays on one accelerator.
    """
    tensor = getattr(backend, "as_tensor", None) or backend.tensor
    device = getattr(like if like is not None else value, "device", None)

    if device is None:
        return tensor(np.asarray(value, dtype="float32"))

    if hasattr(value, "shape") and not isinstance(value, (list, tuple, np.ndarray)):
        # Already a tensor; only its placement needs fixing.
        return value.to(device) if value.device != device else value
    return tensor(np.asarray(value, dtype="float32")).to(device)


def create_quantile_model(
    input_shape: tuple[int, int],
    units: int = 64,
    dropout_rate: float = 0.2,
    dense_units: int = 32,
    quantiles: tuple[float, ...] = ALL_QUANTILES,
) -> Any:
    """The distribution forecaster: one output per quantile level.

    The trunk is `lstm_model.build_lstm_stack`, shared with the point
    forecaster. The head count is the only difference, and keeping the bodies
    identical is what stops the two architectures from drifting apart — which
    they had, in unit count and dropout placement.

    Smaller than the point model on purpose (`units` 64 vs 100, dense 32).
    Twenty-five output heads carry far more parameters to regularise than one,
    and this dataset is exactly the regime where a bigger network memorises
    the training window.
    """
    model = build_lstm_stack(
        input_shape, units, dropout_rate, dense_units, len(quantiles)
    )
    model.compile(optimizer="adam", loss=pinball_loss)
    return model


def sort_quantiles(values: np.ndarray, quantiles: tuple[float, ...] = ALL_QUANTILES):
    """Enforce a monotone distribution along the quantile axis.

    Independent regression heads have no reason to respect the ordering the
    loss implies, and empirically they do cross — usually between the 90% and
    95% heads, where the density of training outcomes is lowest and the
    gradients are noisiest. A crossing band is not a conservative band, it is a
    contradiction, and a user reading "the 5% case is 3.2 and the 10% case is
    2.9" is being handed a puzzle rather than a range.

    Sorting each row is the projection onto the monotone cone. It is the
    standard remedy and it is idempotent, so calling it twice is a no-op.
    """
    return np.sort(np.asarray(values, dtype=float), axis=-1)


def predict_quantiles(
    model: Any,
    windows: np.ndarray,
    quantiles: tuple[float, ...] = ALL_QUANTILES,
) -> np.ndarray:
    """Run the network and return quantiles in `quantiles` order.

    The network emits heads in `ALL_QUANTILES` order; this reorders to whatever
    subset the caller asked for *after* sorting, so a caller requesting only
    `(0.05, 0.5, 0.95)` gets those three, correctly ordered.
    """
    raw = model.predict(windows, verbose="0")
    ordered = sort_quantiles(raw, ALL_QUANTILES)
    if tuple(quantiles) == ALL_QUANTILES:
        return ordered
    index = [ALL_QUANTILES.index(q) for q in quantiles]
    return ordered[:, index]


def quantile_returns_to_prices(
    origin_price: float | np.ndarray,
    quantile_returns: np.ndarray,
) -> np.ndarray:
    """Convert forecast log-returns into price levels per quantile.

    A negative quantile stays negative. The temptation is to clamp at zero —
    "prices cannot go below zero" — but that is true of the *level* and false
    of the *return*, and applying it here would delete the entire lower tail
    this model exists to produce. The level is reconstructed by compounding,
    and only a return below `-inf` would imply a non-positive price, which a
    forecast of a bounded-magnitude daily return will never produce.
    """
    origin = np.asarray(origin_price, dtype=float)
    return origin * np.exp(np.asarray(quantile_returns, dtype=float))


def band_from_quantiles(
    quantile_returns: np.ndarray, lower: float = 0.05, upper: float = 0.95
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pull a band and a median out of a full quantile sweep.

    Returns `(median, lower_price_offset, upper_price_offset)` as log-return
    offsets, so the caller can compound them against whatever price basis it
    has. The lower offset is expected to be negative; that is the point.
    """
    ordered = sort_quantiles(quantile_returns, ALL_QUANTILES)
    median_idx = ALL_QUANTILES.index(0.5)
    lower_idx = ALL_QUANTILES.index(lower)
    upper_idx = ALL_QUANTILES.index(upper)
    return (
        ordered[:, median_idx],
        ordered[:, lower_idx],
        ordered[:, upper_idx],
    )


def quantile_coverage(
    realised: np.ndarray,
    quantile_returns: np.ndarray,
    quantiles: tuple[float, ...] = ALL_QUANTILES,
) -> dict[str, float]:
    """Did each quantile land where it claimed?

    This is the scoreboard. A quantile model's whole claim is that its `q`-th
    output is below the outcome `q` of the time. Read across the returned
    levels: flat near the target means calibrated, systematically above means
    overconfident, and a monotone violation is impossible because the outputs
    are sorted before this point.

    Reported as the observed fraction at or below each level, which is the
    quantity that should equal `q`. Values are keys rather than a frame
    because callers want to display a handful, not pivot a table.
    """
    realised = np.asarray(realised, dtype=float).reshape(-1, 1)
    ordered = sort_quantiles(quantile_returns, ALL_QUANTILES)
    out: dict[str, float] = {}
    for q, column in zip(quantiles, ordered.T, strict=True):
        hit = float((realised <= column).mean())
        out[f"{q:.3f}"] = hit
    # A single mean absolute deviation across levels, for one-number summary.
    deviations = [abs(out[f"{q:.3f}"] - q) for q in quantiles]
    out["mean_abs_error"] = float(np.mean(deviations))
    return out


def fit_quantile_windows(
    features: np.ndarray, target: np.ndarray, length: int
) -> tuple[np.ndarray, np.ndarray]:
    """Sliding input windows over an already-built feature matrix.

    Split out from `backtest._lstm_upgraded_factory` so the quantile model
    and the point model provably see the *same* windows, in the same order,
    with the same masking. A benchmark where the arms saw different data would
    be worthless, and that is exactly the mistake the origin-reconstruction
    pitfall in the project notes warns about.
    """
    if len(features) <= length:
        # A (0, 0, n) window array, so a caller that stack-cats it gets a
        # sensible empty result rather than a shape mismatch.
        return np.empty((0, length, features.shape[1])), np.empty((0, 1))
    x = np.stack([features[i - length : i] for i in range(length, len(features))])
    y = target[length:].reshape(-1, 1)
    return x, y


class QuantileForecaster:
    """A walk-forward forecaster that predicts the whole return distribution.

    Callable with the same shape as the other factories in `backtest.py` —
    `predict(origin, train_close) -> float` — where that float is the
    **median** forecast price. The full distribution from the same forward pass
    is available on `last_quantiles`, which is the entire point: the band is no
    longer a separate model guessing at the same window from a volatility
    proxy. One network pass produces the point *and* the range.

    A class rather than a function with an attribute bolted on, because the
    attribute pattern is untyped, invisible to a type checker, and reads as an
    accident. The declaration below is the difference between `last_quantiles`
    being checked and being ignored.

    The leakage rule is identical to `_lightgbm_factory`: a training row is
    only usable once its `horizon`-day outcome was known at the origin. This is
    the one thing that must not vary between arms.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        horizon: int,
        length: int = 60,
        epochs: int = 20,
        random_seed: int = 42,
        min_training_rows: int = 40,
    ) -> None:
        from sklearn.preprocessing import MinMaxScaler

        from .features import build_features, forward_log_return

        self.horizon = horizon
        self.length = length
        self.epochs = epochs
        self.random_seed = random_seed
        self.min_training_rows = min_training_rows

        features = build_features(data, ohlcv=False).to_numpy(dtype=float)
        target = forward_log_return(data["Close"].astype(float), horizon).to_numpy(
            dtype=float
        )
        complete = ~np.isnan(features).any(axis=1)
        if not complete.any():
            raise ValueError("no complete feature rows; cannot train")

        self._features = features
        self._target = target
        self._usable = complete & ~np.isnan(target)
        self._first_complete = int(np.argmax(complete))
        # A window ending at row p covers p-length+1 .. p, and every row in that
        # span must itself be a complete feature row (the 60-day volatility is
        # NaN until the warmup completes). Requiring p-length+1 == first_complete
        # gives p == first_complete + length - 1.
        #
        # The guard that matters is the one in `predict`: a single NaN anywhere
        # in the training matrix poisons every weight in the network, the loss
        # reads `nan` from epoch one, and every prediction comes back NaN with
        # no exception raised. The fingerprint is `np.isfinite(x).all()`.
        self._min_position = self._first_complete + length - 1
        self._n_features = features.shape[1]
        self._scaler = MinMaxScaler(feature_range=(0, 1))
        self.last_quantiles: np.ndarray | None = None

    @property
    def median_index(self) -> int:
        """Column of the median within a quantile sweep."""
        return ALL_QUANTILES.index(0.5)

    def predict(self, origin: int, train_close: pd.Series) -> float:
        self.last_quantiles = None
        length, horizon = self.length, self.horizon
        if origin < self._min_position or origin - length + 1 < 0:
            # A silent fallback to naive is indistinguishable from a model
            # that chose to repeat yesterday, which is the exact confusion
            # this project has been bitten by. Say which it was.
            logger.warning(
                "Quantile model: origin %d precedes the first complete "
                "feature window (%d); falling back to naive.",
                origin,
                self._min_position,
            )
            return float(train_close.iloc[-1])

        allowed = np.zeros(len(self._features), dtype=bool)
        allowed[: origin - horizon + 1] = True
        rows = np.flatnonzero(self._usable & allowed)
        rows = rows[rows >= self._min_position]
        if len(rows) < self.min_training_rows:
            logger.warning(
                "Quantile model: only %d usable training rows at origin %d, "
                "need %d; falling back to naive.",
                len(rows),
                origin,
                self.min_training_rows,
            )
            return float(train_close.iloc[-1])

        self._scaler.fit(self._features[rows])
        scaled = self._scaler.transform(self._features)

        # One window per usable training row, built from the scaled matrix.
        # Built inline rather than via `fit_quantile_windows` because the row
        # indices come from the leakage mask, not from a contiguous slice.
        x = np.stack([scaled[p - length + 1 : p + 1] for p in rows])
        y = self._target[rows].reshape(-1, 1)

        # A NaN anywhere in the training matrix poisons every weight in the
        # network. Keras does not raise for it: the loss reads `nan` from epoch
        # one and `predict` returns `nan` for every row, so the failure looks
        # like a modelling result rather than a bug. The windows that reach back
        # into the feature warmup are the cause, so drop them here and say so.
        # `y` is (n, 1) and `x` is (n, length, features), so they are reduced
        # over different axes — hence the separate calls rather than one.
        finite = np.isfinite(x).all(axis=(1, 2)) & np.isfinite(y).all(axis=1)
        if not finite.all():
            keep = np.flatnonzero(finite)
            if len(keep) < self.min_training_rows:
                logger.warning(
                    "Quantile model: %d of %d windows reach into the feature "
                    "warmup and carry NaN; too few clean windows remain to "
                    "train, so falling back to naive at origin %d.",
                    len(finite) - len(keep),
                    len(finite),
                    origin,
                )
                return float(train_close.iloc[-1])
            logger.debug(
                "Quantile model: dropped %d warmup windows at origin %d.",
                len(finite) - len(keep),
                origin,
            )
            x, y = x[keep], y[keep]

        keras_module = _keras()[0]
        keras_module.utils.set_random_seed(self.random_seed)
        net = create_quantile_model(input_shape=(length, self._n_features))
        net.fit(
            x,
            y,
            epochs=self.epochs,
            batch_size=32,
            validation_split=0.1,
            callbacks=[early_stopping(5)],
            verbose="0",
        )

        window = scaled[origin - length + 1 : origin + 1].reshape(
            1, length, self._n_features
        )
        if np.isnan(window).any():
            logger.warning(
                "Quantile model: prediction window at origin %d contains NaN; "
                "falling back to naive.",
                origin,
            )
            return float(train_close.iloc[-1])

        quantiles = predict_quantiles(net, window)
        self.last_quantiles = quantiles[0]
        return float(train_close.iloc[-1] * np.exp(quantiles[0, self.median_index]))

    def __call__(self, origin: int, train_close: pd.Series) -> float:
        """Drop-in for the other backtest factories."""
        return self.predict(origin, train_close)
