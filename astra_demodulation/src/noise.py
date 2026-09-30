"""
noise.py
Noise variance estimation from decision-directed residuals and SNR telemetry.
"""

from typing import Tuple, Optional, Any
import numpy as np


def estimate_noise_variance(
    symbols: np.ndarray,
    nearest_points: np.ndarray,
    sync_noise_var: Optional[float] = None
) -> Tuple[float, str]:
    """
    Estimate noise variance sigma^2 per complex sample.
    Uses sync telemetry if available, otherwise calculates decision-directed error variance.
    """
    if sync_noise_var is not None and sync_noise_var > 1e-8:
        return float(sync_noise_var), "sync_telemetry"

    if symbols is None or len(symbols) < 4:
        return 0.05, "default_fallback"

    errors = symbols - nearest_points
    var = float(np.mean(np.real(errors * np.conj(errors))))
    # Clamp to reasonable bounds [1e-6, 5.0]
    clamped_var = max(1e-6, min(5.0, var))
    return clamped_var, "decision_directed"


def snr_db_to_noise_var(snr_db: float, signal_power: float = 1.0) -> float:
    """Convert SNR in dB to noise variance."""
    snr_lin = 10.0 ** (snr_db / 10.0)
    return float(signal_power / snr_lin)


def noise_var_to_snr_db(noise_var: float, signal_power: float = 1.0) -> float:
    """Convert noise variance to estimated SNR in dB."""
    nv = max(1e-12, float(noise_var))
    snr_lin = signal_power / nv
    return float(10.0 * np.log10(snr_lin))
