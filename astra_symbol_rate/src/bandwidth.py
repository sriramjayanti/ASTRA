"""
ASTRA Bandwidth-Derived Symbol-Rate Estimator V2.
RRC Rolloff candidate generation and spectral power integration with noise-subtracted OBW.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple
import numpy as np


def measure_occupied_bandwidth(
    iq: np.ndarray,
    sample_rate_hz: float,
    power_fraction: float = 0.95,
) -> float:
    """
    Computes the occupied bandwidth (OBW) containing the active signal passband.
    Uses robust peak-cluster extent analysis against median background noise floor,
    preventing wideband AWGN from artificially expanding measured bandwidth to Fs.
    """
    n = min(len(iq), 8192)
    if n < 32 or sample_rate_hz <= 0:
        return 0.0

    win = np.hanning(n)
    spec = np.abs(np.fft.fftshift(np.fft.fft(iq[:n] * win))) ** 2

    # Robust noise floor estimation: median of periodogram
    med_noise = float(np.median(spec))
    p_max = float(np.max(spec))

    # If peak is not at least 3 dB above median noise, signal is buried in noise
    if p_max < med_noise * 2.0:
        return 0.0

    # Signal extent threshold: 5% of peak power or 2.5x median noise floor
    thresh = max(med_noise * 2.5, p_max * 0.05)
    above = np.where(spec >= thresh)[0]
    if len(above) == 0:
        return 0.0

    freqs = np.fft.fftshift(np.fft.fftfreq(n, d=1.0 / sample_rate_hz))
    obw = float(abs(freqs[above[-1]] - freqs[above[0]]))
    return max(50.0, obw)


def extract_bandwidth_candidates(
    iq: np.ndarray,
    sample_rate_hz: float,
    rolloff_alphas: Sequence[float] = (0.20, 0.25, 0.35, 0.50),
) -> Tuple[float, List[Dict[str, float]]]:
    """
    Computes Occupied Bandwidth and generates RRC candidates:
        Rs = BW / (1 + alpha)
    """
    obw_hz = measure_occupied_bandwidth(iq, sample_rate_hz, power_fraction=0.95)
    candidates: List[Dict[str, float]] = []

    for alpha in rolloff_alphas:
        rate = float(obw_hz / (1.0 + alpha))
        sps = float(sample_rate_hz / rate) if rate > 0 else 0.0
        if 1.5 <= sps <= 256.0:
            candidates.append({
                "rate_hz": rate,
                "sps": sps,
                "alpha": float(alpha),
                "obw_hz": obw_hz,
                "score": 0.40,  # Baseline score for RRC bandwidth hypothesis
                "source": "bandwidth_rrc",
            })

    return obw_hz, candidates


def extract_bandwidth_evidence(
    iq: np.ndarray,
    sample_rate_hz: float,
    rolloff_alphas: Sequence[float] = (0.20, 0.25, 0.35, 0.50),
) -> Dict[str, Any]:
    """
    Unified dictionary interface for bandwidth evidence.
    """
    obw_hz, cands = extract_bandwidth_candidates(iq, sample_rate_hz, rolloff_alphas)
    return {
        "occupied_bandwidth_hz": obw_hz,
        "bandwidth_candidates": cands,
    }
