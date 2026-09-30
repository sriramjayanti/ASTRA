"""
Unit tests for Multipath tapped-delay line channel generator (Engine 6).
"""

import numpy as np
import pytest
from astra_synthetic.channel.multipath import apply_multipath, MULTIPATH_PROFILES


def test_multipath_impulse_response():
    # An impulse input delta[n] through a tapped delay line should directly reveal the FIR taps
    impulse = np.zeros(20, dtype=np.complex64)
    impulse[0] = 1.0 + 0j

    custom_taps = np.array([1.0 + 0j, 0.5 * np.exp(1j * 0.8), 0.2 * np.exp(-1j * 1.2)], dtype=np.complex64)
    custom_delays = np.array([0, 3, 7], dtype=np.int64)

    y, taps_used, delays_used = apply_multipath(
        impulse,
        taps=custom_taps,
        delays_samples=custom_delays,
        normalize=False,
    )

    # Check tap values at delay indices
    assert np.isclose(y[0], custom_taps[0], atol=1e-5)
    assert np.isclose(y[3], custom_taps[1], atol=1e-5)
    assert np.isclose(y[7], custom_taps[2], atol=1e-5)
    # Intermediate non-tap samples should be zero
    assert np.isclose(y[1], 0.0, atol=1e-5)
    assert np.isclose(y[2], 0.0, atol=1e-5)


def test_multipath_energy_normalization():
    impulse = np.zeros(10, dtype=np.complex64)
    impulse[0] = 1.0 + 0j

    # With normalize=True, total tap energy sum(|h_k|^2) must equal 1.0
    for prof_name in ["mild", "medium", "severe"]:
        _, taps_used, _ = apply_multipath(impulse, profile=prof_name, normalize=True)
        total_energy = np.sum(np.abs(taps_used) ** 2)
        assert np.isclose(total_energy, 1.0, atol=1e-6)


def test_multipath_profiles_exist():
    assert "mild" in MULTIPATH_PROFILES
    assert "medium" in MULTIPATH_PROFILES
    assert "severe" in MULTIPATH_PROFILES
