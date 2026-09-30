"""
Unit tests for Timing Offset generator (Engine 6).
"""

import numpy as np
import pytest
from astra_synthetic.channel.timing import apply_timing_offset


def test_timing_zero_offset():
    x = np.array([1 + 1j, -2 + 0.5j, 3 - 2j, 0.4 + 1.2j], dtype=np.complex64)
    y = apply_timing_offset(x, timing_offset_samples=0.0)
    assert np.allclose(x, y)
    assert y.dtype == np.complex64


def test_timing_integer_shift():
    x = np.zeros(20, dtype=np.complex64)
    x[5] = 1.0 + 0j  # delta at index 5

    # Delay by +2 samples
    y = apply_timing_offset(x, timing_offset_samples=2.0)
    assert np.isclose(y[7], 1.0 + 0j, atol=1e-3)

    # Advance by -2 samples
    y_adv = apply_timing_offset(x, timing_offset_samples=-2.0)
    assert np.isclose(y_adv[3], 1.0 + 0j, atol=1e-3)


def test_timing_fractional_shift():
    # Sinusoid shifted by 0.5 samples
    Fs = 1000.0
    t = np.arange(200) / Fs
    f0 = 25.0
    x = np.sin(2 * np.pi * f0 * t).astype(np.complex64)

    shift = 0.5
    y = apply_timing_offset(x, timing_offset_samples=shift)

    # In continuous domain: sin(2*pi*f0*(t - shift/Fs))
    t_shifted = (np.arange(200) - shift) / Fs
    expected = np.sin(2 * np.pi * f0 * t_shifted).astype(np.complex64)

    # Compare middle region away from filter transients (filter length ~ 21 taps)
    assert np.allclose(y[30:170], expected[30:170], atol=1e-2)
