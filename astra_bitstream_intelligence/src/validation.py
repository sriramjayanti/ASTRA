"""
validation.py
Input bitstream validation and sanitization for ASTRA Stage 12.
"""

from typing import Tuple, Optional
import numpy as np


def validate_bitstream(
    bits: np.ndarray,
    soft_info: Optional[np.ndarray] = None,
    min_bits: int = 8
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Validate and sanitize input bitstream and optional soft information.

    Args:
        bits: 1D array of hard bits (0 or 1).
        soft_info: Optional 1D array of soft reliability/LLR values.
        min_bits: Minimum allowed bits.

    Returns:
        Tuple[np.ndarray, Optional[np.ndarray]]: Sanitized uint8 bits and float32 soft_info.

    Raises:
        ValueError: If bits are invalid, non-binary, or too short.
        TypeError: If input is not a numpy array or convertible.
    """
    if bits is None:
        raise ValueError("Input bits cannot be None.")

    if not isinstance(bits, np.ndarray):
        bits = np.asarray(bits)

    if bits.ndim == 0 or bits.size == 0:
        raise ValueError("Input bits array is empty.")

    if bits.ndim > 1:
        bits = bits.ravel()

    # Check for NaN or Inf
    if not np.all(np.isfinite(bits)):
        raise ValueError("Input bits contain NaN or infinite values.")

    # Check that values are strictly in {0, 1}
    unique_vals = np.unique(bits)
    if not np.all(np.isin(unique_vals, [0, 1])):
        raise ValueError(f"Input bits must contain only binary values {{0, 1}}, found: {unique_vals}")

    if bits.size < min_bits:
        raise ValueError(f"Input bitstream has length {bits.size}, minimum required is {min_bits}.")

    sanitized_bits = bits.astype(np.uint8)

    sanitized_soft = None
    if soft_info is not None:
        if not isinstance(soft_info, np.ndarray):
            soft_info = np.asarray(soft_info)
        if soft_info.ndim > 1:
            soft_info = soft_info.ravel()
        if soft_info.size != sanitized_bits.size:
            raise ValueError(
                f"Soft info length ({soft_info.size}) does not match bits length ({sanitized_bits.size})."
            )
        if not np.all(np.isfinite(soft_info)):
            soft_info = np.nan_to_num(soft_info, nan=0.0, posinf=10.0, neginf=-10.0)
        sanitized_soft = soft_info.astype(np.float32)

    return sanitized_bits, sanitized_soft
