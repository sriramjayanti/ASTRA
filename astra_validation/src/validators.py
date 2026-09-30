"""
validators.py
Input validation and assertion utilities for ASTRA Stage 10.
"""

from typing import Any, Dict, List, Optional
import numpy as np
from .models import CRCProfile


def validate_bitstream(bits: Any) -> np.ndarray:
    """Validate and convert input into a 1D uint8 bit array containing only 0 and 1."""
    if bits is None:
        return np.array([], dtype=np.uint8)
    arr = np.asarray(bits, dtype=np.uint8).ravel()
    if len(arr) > 0 and not np.all((arr == 0) | (arr == 1)):
        # Normalize non-binary
        arr = (arr > 0).astype(np.uint8)
    return arr


def validate_crc_profile(profile: CRCProfile) -> bool:
    """Check that CRC profile parameters are consistent."""
    if profile.width not in (8, 16, 24, 32, 64):
        return False
    if profile.poly < 0:
        return False
    return True
