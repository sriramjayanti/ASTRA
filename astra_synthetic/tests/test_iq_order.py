"""
Unit tests for IQ vs QI interleaving layout formatting (Engine 7).
"""

import numpy as np
import pytest
from astra_synthetic.capture.layouts import format_iq_scalar_stream, deformat_iq_scalar_stream


def test_iq_ordering_known_values():
    # Input: (1 + 10j), (2 + 20j)
    x = np.array([1.0 + 10.0j, 2.0 + 20.0j], dtype=np.complex64)

    # In IQ order: [1.0, 10.0, 2.0, 20.0]
    scalars_iq = format_iq_scalar_stream(x, iq_order="IQ")
    expected_iq = np.array([1.0, 10.0, 2.0, 20.0], dtype=np.float32)
    assert np.allclose(scalars_iq, expected_iq)

    # In QI order: [10.0, 1.0, 20.0, 2.0]
    scalars_qi = format_iq_scalar_stream(x, iq_order="QI")
    expected_qi = np.array([10.0, 1.0, 20.0, 2.0], dtype=np.float32)
    assert np.allclose(scalars_qi, expected_qi)


def test_iq_order_round_trip():
    rng = np.random.default_rng(42)
    x = (rng.standard_normal(256) + 1j * rng.standard_normal(256)).astype(np.complex64)

    # IQ Round trip
    s_iq = format_iq_scalar_stream(x, iq_order="IQ")
    rec_iq = deformat_iq_scalar_stream(s_iq, iq_order="IQ")
    assert np.allclose(x, rec_iq)

    # QI Round trip
    s_qi = format_iq_scalar_stream(x, iq_order="QI")
    rec_qi = deformat_iq_scalar_stream(s_qi, iq_order="QI")
    assert np.allclose(x, rec_qi)


def test_invalid_iq_order_raises():
    x = np.array([1.0 + 1.0j], dtype=np.complex64)
    with pytest.raises(ValueError):
        format_iq_scalar_stream(x, iq_order="XYZ")

    with pytest.raises(ValueError):
        deformat_iq_scalar_stream(np.array([1.0, 1.0]), iq_order="XYZ")
