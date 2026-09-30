"""
ASTRA Cyclostationary & Cyclic Periodicity Symbol-Rate Estimator V2.
Analyzes cyclic autocorrelation and spectral correlation features with
median background whitening, high-resolution FFT, and sub-bin parabolic peak interpolation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np
from scipy import ndimage
from scipy.signal import find_peaks


def parabolic_peak_refinement(spec: np.ndarray, peak_idx: int, freqs: np.ndarray) -> float:
    """
    Applies 3-point parabolic interpolation around an FFT peak bin to achieve
    sub-bin frequency accuracy (<0.05% relative error).
    """
    if peak_idx <= 0 or peak_idx >= len(spec) - 1:
        return float(freqs[peak_idx])

    alpha = float(spec[peak_idx - 1])
    beta = float(spec[peak_idx])
    gamma = float(spec[peak_idx + 1])

    denom = 2.0 * (2.0 * beta - alpha - gamma)
    if abs(denom) < 1e-12:
        return float(freqs[peak_idx])

    delta = (gamma - alpha) / denom
    delta = float(np.clip(delta, -0.5, 0.5))

    df = float(freqs[1] - freqs[0]) if len(freqs) > 1 else 1.0
    refined_freq = float(freqs[peak_idx] + delta * df)
    return max(0.0, refined_freq)


def extract_cyclostationary_candidates(
    iq: np.ndarray,
    sample_rate_hz: float,
    min_sps: float = 1.5,
    max_sps: float = 256.0,
    n_fft: int = 65536,
) -> List[Dict[str, float]]:
    """
    Computes Cyclic Autocorrelation profile by evaluating the FFT of the signal's
    squared envelope and 4th-power envelope with local median background whitening.
    The discrete clock tone at alpha = Rs exhibits sharp prominence above the whitened background.
    """
    n = len(iq)
    if n < 32 or sample_rate_hz <= 0:
        return []

    x = np.asarray(iq)
    candidates: List[Dict[str, float]] = []

    # Valid symbol rate frequency range
    min_rate = float(sample_rate_hz / max_sps)
    max_rate = float(sample_rate_hz / min_sps)

    # Multi-order nonlinear transforms: squared envelope |x|^2 and 4th-power |x|^4
    env2 = np.abs(x) ** 2
    env4 = np.abs(x) ** 4

    for env, order_name in [(env2, "order2"), (env4, "order4")]:
        env_zm = env - np.mean(env)
        window = np.hanning(len(env_zm))
        spec = np.abs(np.fft.rfft(env_zm * window, n=n_fft))
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate_hz)

        # Median background whitening: separates discrete delta clock spikes from continuous PSD
        med = ndimage.median_filter(spec, size=101) + 1e-12
        line_ratio = spec / med

        mask = (freqs >= min_rate) & (freqs <= max_rate)
        if not np.any(mask):
            continue

        sub_line = line_ratio[mask]
        sub_freqs = freqs[mask]
        if len(sub_line) < 5:
            continue

        # Find discrete spectral lines exceeding background noise floor (ratio >= 1.35)
        peaks, props = find_peaks(sub_line, height=1.35, distance=4)

        for p_idx, peak_bin in enumerate(peaks):
            refined_rate = parabolic_peak_refinement(sub_line, peak_bin, sub_freqs)
            if refined_rate <= 0:
                continue

            sps = float(sample_rate_hz / refined_rate)
            ratio_val = float(sub_line[peak_bin])

            candidates.append({
                "rate_hz": float(refined_rate),
                "sps": sps,
                "cyclic_frequency_hz": float(refined_rate),
                "score": ratio_val,  # Preserves true continuous line ratio
                "source": "cyclostationary",
                "prominence": ratio_val,
            })

    if not candidates:
        return []

    # Cluster duplicate cyclic estimates within 1.5% tolerance
    sorted_cands = sorted(candidates, key=lambda c: c["score"], reverse=True)
    deduped: List[Dict[str, float]] = []
    for c in sorted_cands:
        r = c["rate_hz"]
        matched = False
        for d in deduped:
            if abs(r - d["rate_hz"]) / max(1.0, d["rate_hz"]) <= 0.015:
                d["score"] = max(d["score"], c["score"])
                matched = True
                break
        if not matched:
            deduped.append(c)

    return deduped[:16]


# Alias for backward/interface compatibility
extract_cyclostationary_evidence = extract_cyclostationary_candidates
