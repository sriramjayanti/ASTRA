"""
Unit tests for AWGN impairment generator (Engine 6).
"""

import numpy as np
import pytest
from astra_synthetic.channel.awgn import apply_awgn


def test_awgn_zero_noise_large_snr():
    t = np.linspace(0, 1, 1000, endpoint=False)
    x = np.exp(1j * 2 * np.pi * 10 * t).astype(np.complex64)
    # 100 dB SNR should have negligible noise
    noisy, p_sig, p_noise, snr_meas = apply_awgn(x, snr_db=100.0, seed=42)
    assert np.isclose(snr_meas, 100.0, atol=1.0)
    assert np.allclose(noisy, x, atol=1e-4)


@pytest.mark.parametrize("target_snr", [-5.0, 0.0, 5.0, 10.0, 20.0])
def test_awgn_measured_snr_accuracy(target_snr):
    rng = np.random.default_rng(123)
    # Long signal to reduce empirical variance of noise power estimation
    x = (rng.standard_normal(50000) + 1j * rng.standard_normal(50000)).astype(np.complex64)
    noisy, p_sig, p_noise, snr_meas = apply_awgn(x, snr_db=target_snr, seed=42)

    # Check that realized SNR matches target within 0.25 dB for large sample size
    assert np.isclose(snr_meas, target_snr, atol=0.25)
    assert noisy.dtype == np.complex64


def test_awgn_reproducibility():
    x = np.ones(500, dtype=np.complex64)
    noisy1, _, _, _ = apply_awgn(x, snr_db=10.0, seed=999)
    noisy2, _, _, _ = apply_awgn(x, snr_db=10.0, seed=999)
    noisy3, _, _, _ = apply_awgn(x, snr_db=10.0, seed=1000)

    assert np.array_equal(noisy1, noisy2)
    assert not np.array_equal(noisy1, noisy3)


def test_awgn_empty_input():
    x = np.empty(0, dtype=np.complex64)
    noisy, p_sig, p_noise, snr_meas = apply_awgn(x, snr_db=10.0, seed=42)
    assert len(noisy) == 0
    assert p_sig == 0.0
