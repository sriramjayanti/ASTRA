"""
Unit tests for Carrier Phase Offset generator (Engine 6).
"""

import numpy as np
import pytest
from astra_synthetic.channel.phase import apply_phase_offset


def test_phase_offset_identity():
    x = np.array([1 + 0j, 0 + 1j, -1 + 0j, 0 - 1j], dtype=np.complex64)
    y = apply_phase_offset(x, phase_offset_rad=0.0)
    assert np.allclose(x, y)
    assert y.dtype == np.complex64


def test_phase_offset_pi_over_2():
    x = np.array([1 + 0j, 0 + 1j, -1 + 0j, 0 - 1j], dtype=np.complex64)
    phi = np.pi / 2.0
    y = apply_phase_offset(x, phase_offset_rad=phi)

    expected = x * np.exp(1j * phi)
    assert np.allclose(y, expected, atol=1e-6)


def test_phase_offset_negative_angle():
    x = np.array([0.5 - 0.5j, -0.7 + 0.3j], dtype=np.complex64)
    phi = -np.pi / 3.0
    y = apply_phase_offset(x, phase_offset_rad=phi)

    expected = x * np.exp(1j * phi)
    assert np.allclose(y, expected, atol=1e-6)
