"""
Additive White Gaussian Noise (AWGN) Impairment Module for ASTRA Engine 6.
Calculates exact complex Gaussian noise scaled to target SNR in dB and measures realized SNR.
"""

from __future__ import annotations

import numpy as np


def apply_awgn(
    iq: np.ndarray,
    snr_db: float,
    seed: int | None = None,
) -> tuple[np.ndarray, float, float, float]:
    """Add complex zero-mean Additive White Gaussian Noise to baseband IQ.

    Args:
        iq: 1D complex NumPy array of baseband IQ samples.
        snr_db: Target Signal-to-Noise Ratio in dB.
        seed: Random seed for deterministic reproducibility.

    Returns:
        tuple (noisy_iq, signal_power, noise_power, measured_snr_db)
    """
    if len(iq) == 0:
        return np.empty(0, dtype=np.complex64), 0.0, 0.0, float(snr_db)

    signal_power = float(np.mean(np.abs(iq) ** 2))
    if signal_power <= 1e-15:
        # Near-zero signal power
        return iq.astype(np.complex64), signal_power, 0.0, 0.0

    # Target noise power: P_noise = P_signal / (10^(SNR_dB / 10))
    snr_linear = 10.0 ** (snr_db / 10.0)
    target_noise_power = signal_power / snr_linear

    # Generate standard normal complex Gaussian noise with variance target_noise_power
    # Real and Imag components each have variance target_noise_power / 2
    rng = np.random.default_rng(seed)
    noise_std = np.sqrt(target_noise_power / 2.0)
    noise_real = rng.normal(loc=0.0, scale=noise_std, size=len(iq))
    noise_imag = rng.normal(loc=0.0, scale=noise_std, size=len(iq))
    noise = (noise_real + 1j * noise_imag).astype(np.complex64)

    measured_noise_power = float(np.mean(np.abs(noise) ** 2))
    if measured_noise_power > 0:
        measured_snr_db = float(10.0 * np.log10(signal_power / measured_noise_power))
    else:
        measured_snr_db = float(snr_db)

    noisy_iq = (iq + noise).astype(np.complex64)
    return noisy_iq, signal_power, measured_noise_power, measured_snr_db
