"""
validators.py
Input verification and physical bounds checks for demodulation inputs.
"""

from typing import Tuple, Optional
import numpy as np


def validate_demod_input(
    symbols: np.ndarray,
    modulation: str,
    modulation_family: str,
    min_symbols: int = 4
) -> Tuple[bool, Optional[str]]:
    """
    Validate input symbol stream and modulation parameters.
    """
    if symbols is None or not isinstance(symbols, np.ndarray):
        return False, "Symbols must be a valid NumPy array"

    if symbols.ndim != 1:
        return False, f"Expected 1D symbol array, got shape {symbols.shape}"

    if len(symbols) < min_symbols:
        return False, f"Insufficient symbols ({len(symbols)} < {min_symbols})"

    if not np.all(np.isfinite(symbols)):
        return False, "Input symbols contain non-finite (NaN or Inf) values"

    mod_clean = str(modulation).upper().strip()
    if mod_clean in ["UNKNOWN", "NOISE", "CW", "CHIRP"]:
        return False, f"Cannot demodulate non-target / unclassified modulation: '{modulation}'"

    fam = modulation_family.upper()
    if fam not in ["PSK", "QAM", "FSK"]:
        return False, f"Unsupported modulation family for digital demodulation: '{modulation_family}'"

    return True, None
