"""
resampling.py
Rational polyphase resampling to convert arbitrary non-integer SPS into standard working SPS (e.g., 2.0 SPS).
"""

from typing import Tuple, Dict, Any
import numpy as np
from scipy import signal
from fractions import Fraction


def resample_to_working_sps(
    iq: np.ndarray,
    input_sample_rate_hz: float,
    symbol_rate_hz: float,
    target_sps: float = 2.0,
    max_resample_ratio: float = 100.0
) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    """
    Resample input signal to an exact working SPS (e.g., 2.0 SPS for Gardner TED).
    Handles arbitrary non-integer input SPS without quantization error.

    Returns:
        (resampled_iq, working_sample_rate_hz, metadata)
    """
    input_sps = float(input_sample_rate_hz) / float(symbol_rate_hz)
    
    # If already within 0.1% of target SPS, return directly
    if abs(input_sps - target_sps) < 1e-3:
        meta = {
            "resampled": False,
            "original_fs": float(input_sample_rate_hz),
            "working_fs": float(input_sample_rate_hz),
            "up": 1,
            "down": 1,
            "ratio": 1.0,
            "working_sps": input_sps
        }
        return iq.copy(), float(input_sample_rate_hz), meta

    target_fs = float(symbol_rate_hz) * float(target_sps)
    ratio = target_fs / float(input_sample_rate_hz)

    # Compute closest rational fraction up / down
    frac = Fraction(ratio).limit_denominator(256)
    up, down = frac.numerator, frac.denominator

    # Polyphase rational resample
    resampled = signal.resample_poly(iq, up, down).astype(np.complex64)
    actual_working_fs = float(input_sample_rate_hz) * (up / down)
    actual_working_sps = actual_working_fs / float(symbol_rate_hz)

    meta = {
        "resampled": True,
        "original_fs": float(input_sample_rate_hz),
        "working_fs": float(actual_working_fs),
        "up": int(up),
        "down": int(down),
        "ratio": float(up / down),
        "working_sps": float(actual_working_sps)
    }
    return resampled, actual_working_fs, meta
