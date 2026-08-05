"""End-to-end test of the analysis pipeline with synthetic data."""

import numpy as np
import pandas as pd
import pytest

import src.core.pipeline as pipeline

from src.core.baselines import MOVING_AVERAGE, NAIVE


def make_synthetic_close(n=400, seed=42):
    rng = np.random.default_rng(seed)
    drift = np.linspace(100.0, 160.0, n)
    noise = rng.normal(0, 1.5, n)
    index = pd.bdate_range("2024-01-01", periods=n)
    return pd.DataFrame(
        {
            "Open": drift + noise,
            "High": drift + noise + 2.0,
            "Low": drift + noise - 2.0,
            "Close": drift + noise,
            "Volume": rng.integers(1_000_000, 5_000_000, n).astype("int64"),
        },
        index=index,
    )


@pytest.fixture()
def fake_loader(monkeypatch):
    monkeypatch.setattr(
        pipeline, "download_stock_data", lambda *a, **k: make_synthetic_close()
    )


def test_run_analysis_structure(fake_loader):
    result = pipeline.run_analysis(
        symbol="TEST",
        start_date="2024-01-01",
        end_date="2025-02-01",
        sequence_length=10,
        future_steps=5,
        epochs=2,
        batch_size=16,
        progress_callback=None,
    )

    assert result.symbol == "TEST"
    assert len(result.data) > 100
    assert len(result.test_predictions) == len(result.close) - result.test_start_index
    assert len(result.future) == 5

    for name in ("mse", "rmse", "mae", "mape", "r2"):
        assert name in result.lstm_metrics
    assert result.lstm_metrics["mse"] >= 0.0
    assert result.lstm_metrics["mape"] >= 0.0

    assert set(result.baselines.keys()) == {NAIVE, MOVING_AVERAGE}


def test_pipeline_error_on_empty_data(monkeypatch):
    monkeypatch.setattr(pipeline, "download_stock_data", lambda *a, **k: pd.DataFrame())

    with pytest.raises(pipeline.PipelineError):
        pipeline.run_analysis(
            "EMPTY",
            "2024-01-01",
            "2025-01-01",
            sequence_length=10,
            future_steps=5,
            epochs=2,
            batch_size=16,
        )
