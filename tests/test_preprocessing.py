"""Unit tests for scaling and sequence creation."""

import numpy as np

from src.core.preprocessing import create_sequences, inverse_scale_data, scale_data


def test_scale_roundtrip():
    prices = np.array([[100.0], [150.0], [200.0], [130.0], [170.0]])
    scaled, scaler = scale_data(prices)

    assert scaled.min() == 0.0
    assert scaled.max() == 1.0

    restored = inverse_scale_data(scaled, scaler)
    np.testing.assert_allclose(restored, prices, atol=1e-6)


def test_create_sequences_shapes():
    data = np.arange(20.0).reshape(-1, 1)
    sequences = create_sequences(data, sequence_length=5)
    assert sequences is not None
    x, y = sequences

    assert x.shape == (15, 5, 1)
    assert y.shape == (15, 1)
    # First window holds indices 0..4, target is index 5
    np.testing.assert_array_equal(x[0, :, 0], np.arange(5))
    assert y[0, 0] == 5.0


def test_create_sequences_not_enough_data():
    data = np.arange(5.0).reshape(-1, 1)
    assert create_sequences(data, sequence_length=5) is None
