"""
autocorrelation.py
Direct and FFT-based normalized bipolar bitstream autocorrelation and peak detection.
"""

from typing import List, Tuple
import numpy as np
from scipy.signal import find_peaks

from .models import AutocorrPeak


def bits_to_bipolar(bits: np.ndarray) -> np.ndarray:
    """Convert binary bits {0, 1} to bipolar floats {-1.0, +1.0}."""
    return 2.0 * bits.astype(np.float32) - 1.0


def direct_autocorrelation(bipolar: np.ndarray, max_lag: int) -> np.ndarray:
    """Compute normalized sample autocorrelation directly for small N."""
    n = len(bipolar)
    max_lag = min(max_lag, n - 1)
    if max_lag <= 0:
        return np.array([1.0], dtype=np.float32)

    r = np.zeros(max_lag + 1, dtype=np.float32)
    # Mean zero centered or raw bipolar product
    denom = np.dot(bipolar, bipolar)
    if denom <= 1e-9:
        denom = float(n)

    for k in range(max_lag + 1):
        r[k] = np.dot(bipolar[:n - k], bipolar[k:]) / denom

    return r


def fft_autocorrelation(bipolar: np.ndarray, max_lag: int) -> np.ndarray:
    """Compute normalized sample autocorrelation via FFT (O(N log N))."""
    n = len(bipolar)
    max_lag = min(max_lag, n - 1)
    if max_lag <= 0:
        return np.array([1.0], dtype=np.float32)

    # Next power of 2 for circular-to-linear convolution padding
    n_fft = 2 ** int(np.ceil(np.log2(2 * n - 1)))
    f = np.fft.rfft(bipolar, n=n_fft)
    psd = f * np.conj(f)
    corr = np.fft.irfft(psd, n=n_fft)[:max_lag + 1]

    denom = corr[0]
    if denom <= 1e-9:
        denom = float(n)

    normalized = corr / denom
    return normalized.astype(np.float32)


def bit_autocorrelation(
    bits: np.ndarray,
    max_lag_fraction: float = 0.5,
    max_lag_limit: int = 4096,
    fft_threshold: int = 2048
) -> np.ndarray:
    """
    Compute normalized autocorrelation of binary bitstream after bipolar conversion.

    Args:
        bits: 1D uint8 binary array.
        max_lag_fraction: Fraction of bitstream length to compute lags for.
        max_lag_limit: Absolute cap on maximum lag.
        fft_threshold: Bitstream size above which FFT method is used.

    Returns:
        1D array of normalized autocorrelation values for lag 0..max_lag.
    """
    n = len(bits)
    if n < 2:
        return np.array([1.0], dtype=np.float32)

    max_lag = min(int(n * max_lag_fraction), max_lag_limit, n - 1)
    if max_lag < 1:
        max_lag = 1

    bipolar = bits_to_bipolar(bits)

    if n >= fft_threshold:
        return fft_autocorrelation(bipolar, max_lag)
    else:
        return direct_autocorrelation(bipolar, max_lag)


def find_autocorrelation_peaks(
    corr: np.ndarray,
    min_lag: int = 16,
    peak_threshold: float = 0.12,
    top_k: int = 20
) -> List[AutocorrPeak]:
    """
    Find top-K peaks in autocorrelation curve, strictly excluding lag 0.

    Args:
        corr: 1D array of normalized autocorrelation values.
        min_lag: Minimum lag to consider (avoids trivial mainlobe spread near 0).
        peak_threshold: Minimum correlation value to qualify as a peak.
        top_k: Maximum number of peaks to return.

    Returns:
        List of AutocorrPeak sorted by correlation descending.
    """
    if len(corr) <= min_lag:
        return []

    # Search peaks from min_lag onward
    search_corr = corr[min_lag:]
    peak_indices, properties = find_peaks(
        search_corr,
        height=peak_threshold,
        prominence=0.03
    )

    if len(peak_indices) == 0:
        # Fallback: if no distinct sharp peak found above threshold, check max peak if positive
        if np.max(search_corr) >= peak_threshold:
            max_idx = int(np.argmax(search_corr))
            peak_indices = np.array([max_idx])
            properties = {"prominences": np.array([search_corr[max_idx]])}
        else:
            return []

    peaks: List[AutocorrPeak] = []
    prominences = properties.get("prominences", np.zeros_like(peak_indices, dtype=np.float32))

    for idx, prom in zip(peak_indices, prominences):
        actual_lag = int(idx + min_lag)
        val = float(corr[actual_lag])
        peaks.append(AutocorrPeak(
            lag=actual_lag,
            correlation=val,
            prominence=float(prom)
        ))

    # Sort descending by correlation strength
    peaks.sort(key=lambda p: p.correlation, reverse=True)
    return peaks[:top_k]
