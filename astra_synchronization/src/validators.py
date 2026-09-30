"""
validators.py
Input verification and physical bounds checks for synchronization inputs.
"""

from typing import Tuple, Optional
import numpy as np


def validate_iq_input(
    iq: np.ndarray,
    sample_rate_hz: float,
    symbol_rate_hz: float,
    min_samples: int = 64
) -> Tuple[bool, Optional[str]]:
    """
    Validate input IQ array and rate parameters.
    """
    if iq is None or not isinstance(iq, np.ndarray):
        return False, "Input IQ signal must be a valid NumPy array"

    if iq.ndim > 1 and iq.shape[0] == 2:
        # Convert [2, N] real/imag channels to 1D complex
        iq = iq[0] + 1j * iq[1]
    elif iq.ndim != 1:
        return False, f"Expected 1D IQ array or 2-channel [2, N], got shape {iq.shape}"

    if len(iq) < min_samples:
        return False, f"IQ buffer length ({len(iq)}) is below minimum required ({min_samples})"

    if not np.all(np.isfinite(iq)):
        return False, "Input IQ signal contains non-finite (NaN or Inf) values"

    if sample_rate_hz <= 0 or not np.isfinite(sample_rate_hz):
        return False, f"Invalid sampling rate: {sample_rate_hz}"

    if symbol_rate_hz <= 0 or not np.isfinite(symbol_rate_hz):
        return False, f"Invalid symbol rate: {symbol_rate_hz}"

    sps = float(sample_rate_hz) / float(symbol_rate_hz)
    if sps < 1.0:
        return False, f"Nyquist violation: SPS ({sps:.3f}) is less than 1.0"

    return True, None
