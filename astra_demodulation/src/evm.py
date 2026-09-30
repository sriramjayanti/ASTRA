"""
evm.py
Error Vector Magnitude (EVM) calculation across complex constellations.
"""

from typing import Tuple
import numpy as np


def compute_evm(
    received_symbols: np.ndarray,
    reference_symbols: np.ndarray
) -> Tuple[float, float, float]:
    """
    Calculate Error Vector Magnitude (EVM).
    
    Formula:
      EVM_RMS = sqrt( sum |y[n] - s_hat[n]|^2 / sum |s_hat[n]|^2 )
      EVM_%   = 100 * EVM_RMS
      EVM_dB  = 20 * log10(EVM_RMS + 1e-12)
      
    Returns:
        (evm_rms, evm_percent, evm_db)
    """
    if received_symbols is None or len(received_symbols) == 0:
        return 1.0, 100.0, 0.0

    errors = received_symbols - reference_symbols
    error_power = np.sum(np.real(errors * np.conj(errors)))
    ref_power = np.sum(np.real(reference_symbols * np.conj(reference_symbols)))

    if ref_power < 1e-12:
        ref_power = float(len(received_symbols))

    evm_rms = float(np.sqrt(error_power / ref_power))
    evm_percent = float(100.0 * evm_rms)
    evm_db = float(20.0 * np.log10(max(1e-6, evm_rms)))

    return evm_rms, evm_percent, evm_db
