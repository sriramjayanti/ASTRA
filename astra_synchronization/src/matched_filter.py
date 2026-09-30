"""
matched_filter.py
Root-Raised Cosine (RRC) and pulse matched filtering with group delay compensation.
"""

from typing import Tuple, Optional
import numpy as np
from scipy import signal


def design_rrc_filter(
    sps: float,
    rolloff: float = 0.35,
    span_symbols: int = 8
) -> np.ndarray:
    """
    Design unit-energy Root-Raised Cosine (RRC) impulse response.
    Supports non-integer SPS.
    """
    rolloff = max(0.01, min(0.99, float(rolloff)))
    sps = max(1.0, float(sps))
    n_taps = int(round(span_symbols * sps))
    if n_taps % 2 == 0:
        n_taps += 1

    t = np.arange(-(n_taps // 2), n_taps // 2 + 1, dtype=np.float64) / sps
    h = np.zeros(len(t), dtype=np.float64)

    for i, ti in enumerate(t):
        if abs(ti) < 1e-8:
            h[i] = 1.0 - rolloff + 4.0 * rolloff / np.pi
        elif abs(abs(4.0 * rolloff * ti) - 1.0) < 1e-6:
            h[i] = (rolloff / np.sqrt(2.0)) * (
                (1.0 + 2.0 / np.pi) * np.sin(np.pi / (4.0 * rolloff)) +
                (1.0 - 2.0 / np.pi) * np.cos(np.pi / (4.0 * rolloff))
            )
        else:
            num = np.sin(np.pi * ti * (1.0 - rolloff)) + 4.0 * rolloff * ti * np.cos(np.pi * ti * (1.0 + rolloff))
            denom = np.pi * ti * (1.0 - (4.0 * rolloff * ti) ** 2)
            h[i] = num / denom

    # Normalize to unit energy
    energy = np.sqrt(np.sum(h ** 2))
    if energy > 1e-12:
        h = h / energy
    return h.astype(np.float32)


def apply_matched_filter(
    iq: np.ndarray,
    sps: float,
    filter_type: str = "rrc",
    rolloff: float = 0.35,
    span_symbols: int = 8
) -> Tuple[np.ndarray, int]:
    """
    Apply matched filter to IQ waveform and return filtered signal with group delay compensation.
    
    Returns:
        (filtered_iq, group_delay_samples)
    """
    if filter_type == "rrc":
        h = design_rrc_filter(sps=sps, rolloff=rolloff, span_symbols=span_symbols)
    elif filter_type == "gaussian":
        n_taps = int(round(span_symbols * sps)) | 1
        t = np.linspace(-span_symbols / 2, span_symbols / 2, n_taps)
        bt = max(0.1, min(1.0, float(rolloff)))
        sigma = np.sqrt(np.log(2.0)) / (2.0 * np.pi * bt)
        h = np.exp(-t**2 / (2.0 * sigma**2))
        h = h / np.sum(h)
    else:
        return iq.copy(), 0

    group_delay = (len(h) - 1) // 2
    # Apply symmetric FIR convolution with 'same' mode to keep length identical and centered
    filtered = signal.convolve(iq, h, mode="same").astype(np.complex64)
    return filtered, group_delay
