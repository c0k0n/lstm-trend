"""Unit tests for the regression metrics."""

import numpy as np
import pytest

from src.core.metrics import METRIC_NAMES, regression_metrics


def test_perfect_prediction():
    y_true = np.array([100.0, 101.0, 102.0])
    m = regression_metrics(y_true, y_true)

    assert m["mse"] == 0.0
    assert m["rmse"] == 0.0
    assert m["mae"] == 0.0
    assert m["mape"] == 0.0
    assert m["r2"] == 1.0


def test_known_errors():
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([12.0, 20.0, 27.0])

    m = regression_metrics(y_true, y_pred)

    # errors: +2, 0, -3 -> SSE 13, MAE 5/3, MAPE (0.2 + 0 + 0.1) / 3
    assert m["mse"] == pytest.approx(13.0 / 3)
    assert m["rmse"] == pytest.approx((13.0 / 3) ** 0.5)
    assert m["mae"] == pytest.approx(5.0 / 3)
    assert m["mape"] == pytest.approx(0.1)
    # SS_total = 100 + 0 + 100 = 200
    assert m["r2"] == pytest.approx(1 - (13.0 / 200.0))


def test_all_metrics_present():
    m = regression_metrics(np.array([1.0, 2.0]), np.array([1.5, 2.5]))
    assert set(m.keys()) == set(METRIC_NAMES)
