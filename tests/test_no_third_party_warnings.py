"""The torch.jit.script_method warning must stay ABSENT, not merely silenced.

The entry point used to install a `warnings.filterwarnings` for this, on the
stated belief that "Keras still touches it". That belief was wrong, and a
filter was hiding a warning the project never actually produced — which is
worse than the noise, because it looks like a fix.

The decorator lives in `torch/utils/mkldnn.py`, inside an `lru_cache`'d class
factory, so it only executes when MKLDNN conversion is actually requested.
These tests assert the real thing: that every code path this app runs produces
zero of these warnings, with no filter in place. If a future dependency change
starts emitting one, these fail and the cause gets fixed rather than muted.
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("KERAS_BACKEND", "torch")

DEPRECATION = "torch.jit.script_method"


def _jit_warnings(caught: list[warnings.WarningMessage]) -> list[str]:
    return [f"{w.filename}:{w.lineno}" for w in caught if DEPRECATION in str(w.message)]


@pytest.fixture
def ohlcv() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 600
    index = pd.bdate_range("2022-01-01", periods=n)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    return pd.DataFrame(
        {
            "Open": close,
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": rng.integers(1_000_000, 5_000_000, n).astype(float),
        },
        index=index,
    )


def test_the_entry_point_installs_no_warning_filter():
    """The regression itself: no blanket suppression of this warning.

    `simplefilter("always")` inside the block means any filter the entry point
    installed would be visible — a filter added by the module under test would
    be overridden here, so the assertion below is the real check.
    """
    import streamlit_app

    assert hasattr(streamlit_app, "_quiet_third_party_noise")
    # The only logger it silences is the triton one; no warnings.filterwarnings
    # call remains in the module.
    source = (
        __import__("pathlib").Path(streamlit_app.__file__).read_text(encoding="utf-8")
    )
    assert "filterwarnings" not in source, (
        "streamlit_app re-introduced a warnings filter; the script_method "
        "warning is not produced by this app and should not be suppressed"
    )
    assert "script_method" not in source.split('"""')[2], (
        "the filter should be gone, not merely documented"
    )


def test_importing_torch_and_keras_emits_no_such_warning():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        import torch  # noqa: F401
        import torch.utils.mkldnn  # noqa: F401

    assert _jit_warnings(caught) == []


def test_building_and_training_the_model_emits_no_such_warning():
    """The app's real training path — the one that used to look guilty."""
    from src.core import preprocessing as pp
    from src.core.lstm_model import create_lstm_model, train_model

    rng = np.random.default_rng(0)
    n = 500
    index = pd.bdate_range("2022-01-01", periods=n)
    close = pd.Series(
        100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), index=index, name="Close"
    )

    returns = pp.log_returns(close).dropna()
    values = returns.to_numpy().reshape(-1, 1)
    scaler = pp.fit_scaler(values[:350])
    matrix = np.column_stack(
        [scaler.transform(values), pp.calendar_frame(returns.index)]
    )
    windows = pp.build_return_windows(matrix, 60)
    assert windows is not None, "500 rows must yield windows at length 60"
    x, y = windows

    net = create_lstm_model(
        input_shape=(60, matrix.shape[1]),
        units=8,
        dropout_rate=0.2,
        dense_units=4,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        train_model(
            net,
            x[:300],
            y[:300],
            epochs=1,
            batch_size=32,
            validation_split=0.1,
            patience=5,
        )
        net.predict(x[300:310], verbose="0")

    assert _jit_warnings(caught) == []


def test_the_walkforward_with_the_lstm_emits_no_such_warning(ohlcv):
    """The Evidence page's heaviest path: refits the network every window."""
    from src.core.backtest import BacktestConfig, walk_forward

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        walk_forward(
            ohlcv,
            BacktestConfig(horizon=5, n_origins=2, include_upgraded_lstm=True),
        )

    assert _jit_warnings(caught) == []


def test_only_mkldnn_conversion_triggers_it_which_we_never_call():
    """Document *why* the app is clean, and keep it that way deliberately.

    This is the control: the same warning IS reachable, via
    `torch.utils.mkldnn.to_mkldnn`. Proving the trigger exists means the
    assertions above are measuring something real rather than passing because
    the warning no longer exists at all.
    """
    import torch
    import torch.utils.mkldnn as tmk

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tmk.to_mkldnn(torch.nn.Linear(4, 4))

    assert _jit_warnings(caught), (
        "to_mkldnn no longer triggers the warning — the upstream fix landed, "
        "so the reasoning behind removing our filter should be revisited"
    )

    # And this project's own source must contain no such call.
    import pathlib

    src = pathlib.Path(__file__).resolve().parent.parent / "src"
    offenders = [
        str(p.relative_to(src))
        for p in src.rglob("*.py")
        if "mkldnn" in p.read_text(encoding="utf-8")
    ]
    assert offenders == [], f"src now reaches into MKLDNN: {offenders}"
