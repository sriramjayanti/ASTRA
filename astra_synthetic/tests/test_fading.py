"""
Unit tests for Rayleigh and Rician Fading generators (Engine 6).
"""

import numpy as np
import pytest
from astra_synthetic.channel.fading import apply_rayleigh_fading, apply_rician_fading


def test_rayleigh_fading_energy():
    # Over many trials, E[|h|^2] = 1.0
    num_trials = 5000
    h_powers = []
    dummy_x = np.ones(10, dtype=np.complex64)

    for i in range(num_trials):
        _, h = apply_rayleigh_fading(dummy_x, seed=i)
        h_powers.append(np.abs(h) ** 2)

    mean_power = np.mean(h_powers)
    # Expected mean power is 1.0; check within statistical tolerance
    assert np.isclose(mean_power, 1.0, atol=0.08)


def test_rician_fading_k_dominance():
    # As K -> infinity, h -> 1.0 (pure LOS)
    dummy_x = np.array([1.0 + 0j], dtype=np.complex64)

    # Low K (e.g. K = 0 dB) -> strong scattering variance
    _, h_low, _ = apply_rician_fading(dummy_x, k_factor_db=0.0, seed=42)

    # High K (e.g. K = 40 dB) -> dominated by LOS component (1.0)
    _, h_high, _ = apply_rician_fading(dummy_x, k_factor_db=40.0, seed=42)

    assert np.isclose(np.abs(h_high), 1.0, atol=0.02)
    assert np.isclose(h_high.real, 1.0, atol=0.02)


def test_fading_reproducibility():
    x = np.array([1.0 + 1j, -1.0 + 0.5j], dtype=np.complex64)

    y1, h1 = apply_rayleigh_fading(x, seed=777)
    y2, h2 = apply_rayleigh_fading(x, seed=777)
    y3, h3 = apply_rayleigh_fading(x, seed=778)

    assert np.allclose(y1, y2)
    assert h1 == h2
    assert not np.allclose(y1, y3)
