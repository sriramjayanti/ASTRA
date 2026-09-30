"""
ASTRA Spectral Peak Spacing and Squaring Transformation Symbol-Rate Estimator V2.
Extracts clock lines, MSK carrier line spacing, and harmonic spectral features.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple
import numpy as np
from scipy import signal
from scipy.signal import find_peaks

from .cyclostationary import parabolic_peak_refinement


def compute_spectral_features(
    iq: np.ndarray,
    sample_rate_hz: float,
    nperseg: int = 1024,
    n_fft: int = 65536,
) -> Tuple[float, float, List[Dict[str, float]]]:
    """
    Computes Welch PSD, Spectral Flatness, Centroid, and High-Resolution Squaring Peak Spacing.
    For MSK, squaring x[n]^2 generates discrete tones at 2*fc +- Rs/2 with separation Delta f = Rs.
    """
    n = len(iq)
    if n < 32 or sample_rate_hz <= 0:
        return 0.0, 0.0, []

    x = np.asarray(iq)
    seg_len = min(n, nperseg)
    freqs_w, psd_w = signal.welch(x, fs=sample_rate_hz, nperseg=seg_len, return_onesided=False)
    freqs_w = np.fft.fftshift(freqs_w)
    psd_w = np.fft.fftshift(psd_w)

    # 1. Spectral Flatness (Wiener entropy)
    psd_pos = np.clip(psd_w, 1e-15, None)
    geom_mean = np.exp(np.mean(np.log(psd_pos)))
    arith_mean = np.mean(psd_pos)
    flatness = float(geom_mean / (arith_mean + 1e-15))

    # 2. Spectral Centroid
    centroid = float(np.sum(freqs_w * psd_w) / (np.sum(psd_w) + 1e-15))

    # 3. High-Resolution Squaring Spectrum Analysis: x[n]^2
    x_sq = x ** 2
    window = np.hanning(len(x_sq))
    spec_sq = np.abs(np.fft.fftshift(np.fft.fft(x_sq * window, n=n_fft)))
    freqs_sq = np.fft.fftshift(np.fft.fftfreq(n_fft, d=1.0 / sample_rate_hz))

    candidates: List[Dict[str, float]] = []
    max_sq = float(np.max(spec_sq))
    if max_sq > 1e-9:
        p_peaks, props = find_peaks(spec_sq, prominence=max_sq * 0.08, distance=10)
        if len(p_peaks) >= 2:
            # Sort detected peaks descending by prominence
            sorted_p = sorted(p_peaks, key=lambda p: spec_sq[p], reverse=True)[:6]
            refined_peaks = []
            for p in sorted_p:
                r_f = parabolic_peak_refinement(spec_sq, p, freqs_sq)
                refined_peaks.append(r_f)

            # Check primary pair of highest peaks in squared spectrum for MSK
            sp = abs(refined_peaks[0] - refined_peaks[1])
            if sp > 100.0:
                sps = float(sample_rate_hz / sp)
                if 1.5 <= sps <= 256.0:
                    candidates.append({
                        "rate_hz": float(sp),
                        "sps": sps,
                        "score": 18.0,
                        "source": "msk_squaring",
                    })

    # 4. Direct Spectrum Peak Spacing for FSK Multi-tone Analysis
    n_raw = min(len(x), 16384)
    win_raw = np.hanning(n_raw)
    spec_raw = np.abs(np.fft.fftshift(np.fft.fft(x[:n_raw] * win_raw)))
    freqs_raw = np.fft.fftshift(np.fft.fftfreq(n_raw, d=1.0 / sample_rate_hz))
    med_raw = float(np.median(spec_raw))
    max_raw = float(np.max(spec_raw))
    if max_raw > med_raw * 2.5:
        min_dist = max(3, int(n_raw * 0.0008))
        raw_peaks, _ = find_peaks(spec_raw, height=max(med_raw * 2.2, max_raw * 0.15), distance=min_dist)
        if 2 <= len(raw_peaks) <= 12:
            sorted_rp = sorted(raw_peaks, key=lambda p: spec_raw[p], reverse=True)[:8]
            ref_rp = sorted([parabolic_peak_refinement(spec_raw, p, freqs_raw) for p in sorted_rp])
            # Tone spacings between adjacent tone peaks
            for i in range(len(ref_rp) - 1):
                d = abs(ref_rp[i+1] - ref_rp[i])
                if d > 20.0:
                    for cand_rate in [d, d / 2.0, d * 2.0]:
                        sps = float(sample_rate_hz / cand_rate) if cand_rate > 0 else 0
                        if 1.4 <= sps <= 500.0:
                            candidates.append({
                                "rate_hz": float(cand_rate),
                                "sps": sps,
                                "score": 25.0,
                                "source": "fsk_tone_spacing",
                            })

    candidates.sort(key=lambda c: c["score"], reverse=True)
    return flatness, centroid, candidates[:12]


def extract_spectral_evidence(
    iq: np.ndarray,
    sample_rate_hz: float,
    nperseg: int = 1024,
) -> Dict[str, Any]:
    """
    Unified dictionary interface for spectral evidence.
    """
    flatness, centroid, candidates = compute_spectral_features(iq, sample_rate_hz, nperseg)
    return {
        "spectral_flatness": flatness,
        "spectral_centroid_hz": centroid,
        "spectral_peak_spacings": candidates,
    }
