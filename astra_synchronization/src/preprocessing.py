"""
preprocessing.py
Signal conditioning and canonical formatting for the Synchronization Engine.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np


def condition_iq_signal(
    iq: np.ndarray,
    remove_dc: bool = True,
    normalize_power: bool = True,
    target_power: float = 1.0
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Format IQ buffer to canonical complex64, remove DC component, and normalize power.
    Does NOT distort phase or frequency information.
    """
    # Canonical complex64 formatting
    if iq.ndim > 1 and iq.shape[0] == 2:
        conditioned = (iq[0] + 1j * iq[1]).astype(np.complex64)
    else:
        conditioned = iq.astype(np.complex64)

    dc_offset = 0.0 + 0.0j
    if remove_dc:
        dc_offset = np.mean(conditioned)
        conditioned = conditioned - dc_offset

    initial_power = float(np.mean(np.abs(conditioned) ** 2))
    scale_factor = 1.0
    if normalize_power and initial_power > 1e-12:
        scale_factor = np.sqrt(target_power / initial_power)
        conditioned = (conditioned * scale_factor).astype(np.complex64)

    metadata = {
        "dc_offset_real": float(np.real(dc_offset)),
        "dc_offset_imag": float(np.imag(dc_offset)),
        "initial_power": float(initial_power),
        "scale_factor": float(scale_factor),
        "length_samples": len(conditioned)
    }

    return conditioned, metadata
