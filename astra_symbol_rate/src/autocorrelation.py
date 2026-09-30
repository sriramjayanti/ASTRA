"""
ASTRA Autocorrelation-Based Symbol-Rate Estimators.
Envelope and Power Autocorrelation peak analysis.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple
import numpy as np
from scipy.signal import find_peaks


def compute_autocorrelation(x: np.ndarray, max_lag: int) -> np.ndarray:
    """
    Computes unbiased normalized autocorrelation up to max_lag using FFT.
    """
    n = len(x)
    x_centered = x - np.mean(x)
    var = np.var(x)
    if var < 1e-12:
        return np.zeros(max_lag + 1, dtype=np.float32)

    # FFT-based correlation with biased normalization (tapers variance at large lags)
    n_fft = 1 << (2 * n - 1).bit_length()
    fx = np.fft.fft(x_centered, n=n_fft)
    r = np.fft.ifft(fx * np.conj(fx)).real[:max_lag + 1]
    r = (r / n) / var
    return r.astype(np.float32)



def extract_autocorrelation_candidates(
    iq: np.ndarray,
    sample_rate_hz: float,
    min_sps: float = 2.0,
    max_sps: float = 256.0,
    prominence_threshold: float = 0.08,
) -> Tuple[List[Dict[str, float]], List[Dict[str, float]]]:
    """
    Extracts candidate symbol rates from Envelope and Power Autocorrelations.
    
    Returns:
        Tuple[envelope_candidates, power_candidates]
    """
    n = len(iq)
    max_lag = min(int(max_sps * 2), n // 2)
    min_lag = max(2, int(min_sps))

    if max_lag <= min_lag:
        return [], []

    # 1. Envelope Autocorrelation: a[n] = |x[n]|
    env = np.abs(iq)
    r_env = compute_autocorrelation(env, max_lag)

    # 2. Magnitude-Squared Autocorrelation: p[n] = |x[n]|^2
    pwr = env ** 2
    r_pwr = compute_autocorrelation(pwr, max_lag)

    # Extract peaks from Envelope Autocorrelation (both positive peaks and transition dips)
    env_candidates: List[Dict[str, float]] = []
    if len(r_env) > min_lag:
        dyn_range_env = max(1e-6, np.max(r_env[min_lag:]) - np.min(r_env[min_lag:]))
        prom_thresh_env = max(0.005, min(prominence_threshold, dyn_range_env * 0.10))
        peaks_pos, props_pos = find_peaks(r_env[min_lag:], prominence=prom_thresh_env, distance=2)
        peaks_neg, props_neg = find_peaks(-r_env[min_lag:], prominence=prom_thresh_env, distance=2)

        for p_idx, lag in enumerate(peaks_pos + min_lag):
            prom = float(props_pos["prominences"][p_idx])
            sps = float(lag)
            rate = float(sample_rate_hz / sps)
            score = float(np.clip(prom * 2.5, 0.0, 1.0))
            env_candidates.append({
                "rate_hz": rate,
                "sps": sps,
                "lag": float(lag),
                "prominence": prom,
                "score": score,
                "source": "envelope_autocorr",
            })

        for p_idx, lag in enumerate(peaks_neg + min_lag):
            prom = float(props_neg["prominences"][p_idx])
            sps = float(lag)
            rate = float(sample_rate_hz / sps)
            score = float(np.clip(prom * 2.0, 0.0, 1.0))
            env_candidates.append({
                "rate_hz": rate,
                "sps": sps,
                "lag": float(lag),
                "prominence": prom,
                "score": score,
                "source": "envelope_autocorr",
            })

    # Extract peaks from Power Autocorrelation (both positive peaks and transition dips)
    pwr_candidates: List[Dict[str, float]] = []
    if len(r_pwr) > min_lag:
        dyn_range_pwr = max(1e-6, np.max(r_pwr[min_lag:]) - np.min(r_pwr[min_lag:]))
        prom_thresh_pwr = max(0.005, min(prominence_threshold, dyn_range_pwr * 0.10))
        peaks_pos, props_pos = find_peaks(r_pwr[min_lag:], prominence=prom_thresh_pwr, distance=2)
        peaks_neg, props_neg = find_peaks(-r_pwr[min_lag:], prominence=prom_thresh_pwr, distance=2)

        for p_idx, lag in enumerate(peaks_pos + min_lag):
            prom = float(props_pos["prominences"][p_idx])
            sps = float(lag)
            rate = float(sample_rate_hz / sps)
            score = float(np.clip(prom * 2.5, 0.0, 1.0))
            pwr_candidates.append({
                "rate_hz": rate,
                "sps": sps,
                "lag": float(lag),
                "prominence": prom,
                "score": score,
                "source": "power_autocorr",
            })

        for p_idx, lag in enumerate(peaks_neg + min_lag):
            prom = float(props_neg["prominences"][p_idx])
            sps = float(lag)
            rate = float(sample_rate_hz / sps)
            score = float(np.clip(prom * 2.0, 0.0, 1.0))
            pwr_candidates.append({
                "rate_hz": rate,
                "sps": sps,
                "lag": float(lag),
                "prominence": prom,
                "score": score,
                "source": "power_autocorr",
            })



    # Sort descending by score
    env_candidates.sort(key=lambda c: c["score"], reverse=True)
    pwr_candidates.sort(key=lambda c: c["score"], reverse=True)

    return env_candidates[:12], pwr_candidates[:12]



# Alias for backward/interface compatibility
extract_autocorr_evidence = extract_autocorrelation_candidates

