"""
test_evm.py
Unit tests for EVM calculations and SNR conversions.
"""

import numpy as np
from astra_demodulation.src.evm import compute_evm
from astra_demodulation.src.noise import noise_var_to_snr_db, snr_db_to_noise_var


def test_evm_and_snr_helpers():
    ref = np.array([1+1j, 1-1j, -1+1j, -1-1j] * 50, dtype=np.complex64)
    noisy = ref + 0.1 * (np.random.randn(len(ref)) + 1j * np.random.randn(len(ref)))
    evm_rms, evm_pct, evm_db = compute_evm(noisy, ref)
    assert evm_rms > 0.0
    assert evm_pct > 0.0

    snr = 20.0
    n_var = snr_db_to_noise_var(snr)
    snr_est = noise_var_to_snr_db(n_var)
    assert abs(snr - snr_est) < 1e-4
