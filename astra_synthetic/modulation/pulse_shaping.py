"""
High-level pulse shaping manager, filter transient handlers, and EVM calculation for Engine 5.
"""

from __future__ import annotations

from typing import Any
import numpy as np

from .filters import apply_rrc_pulse_shaping, root_raised_cosine_filter


def compute_evm_rms(
    estimated_symbols: np.ndarray,
    reference_symbols: np.ndarray,
) -> float:
    """Compute Root Mean Square (RMS) Error Vector Magnitude (EVM) in percentage.

    Args:
        estimated_symbols: Extracted / estimated complex symbols.
        reference_symbols: Ideal ground-truth complex constellation points.

    Returns:
        EVM RMS percentage (0.0% for perfect noiseless match).
    """
    if len(estimated_symbols) == 0 or len(reference_symbols) == 0:
        return 0.0
    min_len = min(len(estimated_symbols), len(reference_symbols))
    est = estimated_symbols[:min_len]
    ref = reference_symbols[:min_len]

    error_vector = est - ref
    error_power = np.mean(np.abs(error_vector) ** 2)
    ref_power = np.mean(np.abs(ref) ** 2)

    if ref_power <= 1e-12:
        return 0.0
    evm_fraction = np.sqrt(error_power / ref_power)
    return float(evm_fraction * 100.0)


def extract_matched_symbols(
    clean_iq: np.ndarray,
    sps: int,
    group_delay: int,
    symbol_count: int,
    filter_taps: np.ndarray | None = None,
) -> np.ndarray:
    """Extract symbol decisions from pulse-shaped clean IQ waveform.

    For RRC pulse-shaped signals, applying a matched RRC filter at the receiver
    yields full Nyquist pulse shaping (RC) with zero inter-symbol interference (ISI)
    at the optimum sampling points.

    Args:
        clean_iq: Received/generated clean IQ samples.
        sps: Samples per symbol.
        group_delay: Transmit filter group delay.
        symbol_count: Expected number of symbols.
        filter_taps: Optional matched filter taps.

    Returns:
        1D complex array of sliced symbol samples.
    """
    if len(clean_iq) == 0 or symbol_count == 0:
        return np.empty(0, dtype=np.complex64)

    if filter_taps is not None:
        # Matched filtering: convolve clean_iq with filter_taps
        rx_filtered = np.convolve(clean_iq, filter_taps, mode="full")
        # Total group delay = TX delay + RX delay
        rx_group_delay = (len(filter_taps) - 1) // 2
        total_delay = group_delay + rx_group_delay
        sample_indices = total_delay + np.arange(symbol_count) * sps
        valid_indices = sample_indices[sample_indices < len(rx_filtered)]
        return rx_filtered[valid_indices].astype(np.complex64)
    else:
        # Direct downsampling at TX peak
        sample_indices = group_delay + np.arange(symbol_count) * sps
        valid_indices = sample_indices[sample_indices < len(clean_iq)]
        return clean_iq[valid_indices].astype(np.complex64)
