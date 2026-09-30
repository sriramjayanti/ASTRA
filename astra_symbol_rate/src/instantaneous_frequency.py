"""
ASTRA Instantaneous Frequency and Phase Transition Symbol-Rate Estimator V2.
Particularly effective for FSK, MSK, and continuous-phase modulations.
Analyzes tone-switch periodicity, frequency transition spectra, and discriminator autocorrelation.
"""

from __future__ import annotations

from typing import Any, Dict, List
import numpy as np
from scipy import signal
from scipy.signal import find_peaks

from .autocorrelation import compute_autocorrelation
from .cyclostationary import parabolic_peak_refinement


def extract_instantaneous_frequency_candidates(
    iq: np.ndarray,
    sample_rate_hz: float,
    min_sps: float = 1.5,
    max_sps: float = 256.0,
    prominence_threshold: float = 0.05,
    n_fft: int = 65536,
) -> List[Dict[str, float]]:
    """
    Computes the instantaneous frequency track and analyzes its periodicity to estimate
    FSK/MSK baud rates using both cyclic transition spectra and discriminator autocorrelation.
    """
    if len(iq) < 32 or sample_rate_hz <= 0:
        return []

    x = np.asarray(iq)
    candidates: List[Dict[str, float]] = []

    # 1. Phase differences dphi[n] = angle(x[n] * conj(x[n-1]))
    prod = x[1:] * np.conj(x[:-1])
    dphi = np.angle(prod)  # [-pi, +pi]

    # Instantaneous frequency in Hz
    f_inst = dphi * (sample_rate_hz / (2.0 * np.pi))

    # Noise suppression on discriminator: 3-tap median filter
    f_smooth = signal.medfilt(f_inst, 3)

    # 2. Transition Detector: Energy of frequency transitions |f[n] - f[n-1]|^2
    diff_pwr = np.abs(np.diff(f_smooth)) ** 2

    min_rate = float(sample_rate_hz / max_sps)
    max_rate = float(sample_rate_hz / min_sps)

    # Multi-scale detrending to detect transitions across small and large SPS
    for win_len in [15, 63, 255]:
        if len(diff_pwr) < win_len:
            continue

        pedestal = np.convolve(diff_pwr, np.ones(win_len) / win_len, mode="same")
        diff_detrend = diff_pwr - pedestal

        window = np.hanning(len(diff_detrend))
        spec = np.abs(np.fft.rfft(diff_detrend * window, n=n_fft))
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate_hz)

        mask = (freqs >= min_rate) & (freqs <= max_rate)
        if not np.any(mask):
            continue

        sub_spec = spec[mask]
        sub_freqs = freqs[mask]
        if len(sub_spec) < 5:
            continue

        max_val = float(np.max(sub_spec))
        if max_val < 1e-9:
            continue

        peaks, props = find_peaks(sub_spec, prominence=max_val * 0.015, distance=5)
        for p_idx, peak_bin in enumerate(peaks):
            refined_rate = parabolic_peak_refinement(sub_spec, peak_bin, sub_freqs)
            if refined_rate <= 0:
                continue

            sps = float(sample_rate_hz / refined_rate)
            prom = float(props["prominences"][p_idx])
            score = float(np.clip(prom / max_val * 1.5, 0.05, 1.0))

            candidates.append({
                "rate_hz": float(refined_rate),
                "sps": sps,
                "score": score,
                "source": "instantaneous_frequency",
                "prominence": prom,
            })

    # 3. Fallback: Autocorrelation of smoothed frequency transitions
    f_diff = np.abs(np.diff(f_smooth))
    max_lag = min(int(max_sps * 2), len(f_diff) // 2)
    min_lag = max(2, int(min_sps))

    if max_lag > min_lag:
        r_if = compute_autocorrelation(f_diff, max_lag)
        if len(r_if) > min_lag:
            dyn_range = max(1e-6, np.max(r_if[min_lag:]) - np.min(r_if[min_lag:]))
            prom_thresh = max(0.005, min(prominence_threshold, dyn_range * 0.10))
            peaks_pos, props_pos = find_peaks(r_if[min_lag:], prominence=prom_thresh, distance=2)

            for p_idx, lag in enumerate(peaks_pos + min_lag):
                prom = float(props_pos["prominences"][p_idx])
                sps = float(lag)
                rate = float(sample_rate_hz / sps)
                if min_rate <= rate <= max_rate:
                    score = float(np.clip(prom * 3.0, 0.05, 0.8))
                    candidates.append({
                        "rate_hz": rate,
                        "sps": sps,
                        "lag": float(lag),
                        "prominence": prom,
                        "score": score,
                        "source": "instantaneous_frequency",
                    })

    if not candidates:
        return []

    # Cluster duplicate estimates within 1.0% tolerance
    sorted_cands = sorted(candidates, key=lambda c: c["score"], reverse=True)
    deduped: List[Dict[str, float]] = []
    for c in sorted_cands:
        r = c["rate_hz"]
        matched = False
        for d in deduped:
            if abs(r - d["rate_hz"]) / max(1.0, d["rate_hz"]) <= 0.01:
                d["score"] = max(d["score"], c["score"])
                matched = True
                break
        if not matched:
            deduped.append(c)

    return deduped[:16]


# Alias for backward/interface compatibility
extract_instantaneous_frequency_evidence = extract_instantaneous_frequency_candidates
