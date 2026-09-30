"""
ASTRA Signal Preconditioning and Quality Metric Estimation.
"""

from __future__ import annotations

import math
from typing import Tuple, Union
import numpy as np
from scipy import signal

from .models import InvalidSignalError


def preprocess_signal(
    iq: Union[np.ndarray, Sequence[complex]],
    sample_rate_hz: float,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Standard signal preconditioning:
    1. Validates input and sanitizes NaNs/Infs
    2. Converts to 1D complex64
    3. Removes DC bias (zero-mean)
    4. Normalizes to unit RMS amplitude
    5. Computes SNR and quality metrics
    
    Returns:
        Tuple[clean_complex_iq, quality_metrics_dict]
    """
    if iq is None or len(iq) == 0:
        raise InvalidSignalError("Input signal is empty.")

    x = np.asarray(iq)
    if not np.iscomplexobj(x):
        if x.ndim == 2 and x.shape[0] == 2:
            x = x[0] + 1j * x[1]
        elif x.ndim == 2 and x.shape[1] == 2:
            x = x[:, 0] + 1j * x[:, 1]
        else:
            x = x.astype(np.complex64)

    # Sanitize NaNs and Infs
    x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
    if len(x) < 32:
        raise InvalidSignalError(f"Signal too short: length={len(x)} (minimum 32 required).")

    # Zero-mean DC removal
    dc_mean = np.mean(x)
    x_zero_mean = x - dc_mean

    # Unit RMS normalization
    pwr = np.mean(np.abs(x_zero_mean) ** 2)
    rms = np.sqrt(pwr) + eps
    x_norm = (x_zero_mean / rms).astype(np.complex64)

    # Signal Quality / SNR estimation (M2M4 fourth-moment method)
    r2 = np.mean(np.abs(x_norm) ** 2)
    r4 = np.mean(np.abs(x_norm) ** 4)
    # Kurtosis / Signal-to-noise ratio estimation
    if r4 - 2 * (r2 ** 2) > 0:
        snr_est_lin = np.sqrt(r4 - 2 * (r2 ** 2)) / (r2 - np.sqrt(r4 - 2 * (r2 ** 2)) + eps)
        snr_est_db = float(np.clip(10.0 * np.log10(max(1e-3, snr_est_lin)), -20.0, 50.0))
    else:
        # Fallback spectral noise-floor estimate
        fft_mag = np.abs(np.fft.fft(x_norm[:min(len(x_norm), 4096)]))
        p_sig = np.percentile(fft_mag, 95)
        p_noise = np.percentile(fft_mag, 15) + eps
        snr_est_db = float(np.clip(20.0 * np.log10(p_sig / p_noise), -20.0, 50.0))

    # Clipping detection
    max_amp = np.max(np.abs(x))
    clipping_ratio = float(np.mean(np.abs(x) >= 0.99 * max_amp))

    # Rough CFO estimation (spectrum center of gravity or FFT peak)
    fft_spec = np.abs(np.fft.fftshift(np.fft.fft(x_norm[:min(len(x_norm), 4096)])))
    freqs = np.fft.fftshift(np.fft.fftfreq(len(fft_spec), d=1.0 / sample_rate_hz))
    cfo_hz = float(freqs[np.argmax(fft_spec)])

    metrics = {
        "estimated_snr_db": snr_est_db,
        "clipping_ratio": clipping_ratio,
        "cfo_estimate_hz": cfo_hz,
        "rms_power": float(pwr),
        "dc_offset_mag": float(np.abs(dc_mean)),
        "sample_count": len(x_norm),
    }

    return x_norm, metrics


# Helper wrappers for module consistency
def preprocess_iq(
    iq: Union[np.ndarray, Sequence[complex]],
    sample_rate_hz: float,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, Dict[str, float]]:
    return preprocess_signal(iq, sample_rate_hz, eps)


def compute_snr_estimate(iq: np.ndarray) -> float:
    x = np.asarray(iq)
    r2 = np.mean(np.abs(x) ** 2)
    r4 = np.mean(np.abs(x) ** 4)
    if r4 - 2 * (r2 ** 2) > 0:
        snr_lin = np.sqrt(r4 - 2 * (r2 ** 2)) / (r2 - np.sqrt(r4 - 2 * (r2 ** 2)) + 1e-12)
        return float(np.clip(10.0 * np.log10(max(1e-3, snr_lin)), -20.0, 50.0))
    fft_mag = np.abs(np.fft.fft(x[:min(len(x), 4096)]))
    p_sig = np.percentile(fft_mag, 95)
    p_noise = np.percentile(fft_mag, 15) + 1e-12
    return float(np.clip(20.0 * np.log10(p_sig / p_noise), -20.0, 50.0))


def compute_cfo_estimate(iq: np.ndarray, sample_rate_hz: float) -> float:
    x = np.asarray(iq)
    n = min(len(x), 4096)
    fft_spec = np.abs(np.fft.fftshift(np.fft.fft(x[:n])))
    freqs = np.fft.fftshift(np.fft.fftfreq(len(fft_spec), d=1.0 / sample_rate_hz))
    return float(freqs[np.argmax(fft_spec)])

