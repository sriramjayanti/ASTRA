"""
Digital filter design and pulse shaping implementations for ASTRA Engine 5.
Implements exact analytical Root-Raised Cosine (RRC) filters and vectorized FIR upsampling.
"""

from __future__ import annotations

import numpy as np

try:
    from scipy.signal import upfirdn
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def root_raised_cosine_filter(
    beta: float,
    span_symbols: int,
    sps: int,
) -> np.ndarray:
    """Generate exact Root Raised Cosine (RRC) FIR filter taps with unit energy normalization.

    Args:
        beta: Roll-off excess bandwidth factor (0.0 <= beta <= 1.0).
        span_symbols: Filter duration in symbol periods (must be >= 1, usually even, e.g. 8).
        sps: Samples per symbol (integer >= 1).

    Returns:
        1D NumPy float64 array of unit-energy normalized RRC filter coefficients.
    """
    if not (0.0 <= beta <= 1.0):
        raise ValueError(f"RRC roll-off beta must be in [0.0, 1.0], got {beta}")
    if span_symbols < 1:
        raise ValueError(f"span_symbols must be >= 1, got {span_symbols}")
    if sps < 1:
        raise ValueError(f"sps must be >= 1, got {sps}")

    num_taps = span_symbols * sps + 1
    t = np.linspace(-span_symbols / 2, span_symbols / 2, num_taps, dtype=np.float64)
    h = np.zeros(num_taps, dtype=np.float64)

    # Threshold for singularity detection
    eps = 1e-8

    if abs(beta) < eps:
        # Ideal Sinc filter when beta == 0
        h = np.sinc(t)
    else:
        for i, val in enumerate(t):
            # Singularity at t = 0
            if abs(val) < eps:
                h[i] = 1.0 - beta + (4.0 * beta / np.pi)
            # Singularities at t = +- 1 / (4 * beta)
            elif abs(abs(val) - (1.0 / (4.0 * beta))) < eps:
                term1 = (1.0 + 2.0 / np.pi) * np.sin(np.pi / (4.0 * beta))
                term2 = (1.0 - 2.0 / np.pi) * np.cos(np.pi / (4.0 * beta))
                h[i] = (beta / np.sqrt(2.0)) * (term1 + term2)
            else:
                num = np.sin(np.pi * val * (1.0 - beta)) + (
                    4.0 * beta * val * np.cos(np.pi * val * (1.0 + beta))
                )
                den = np.pi * val * (1.0 - (4.0 * beta * val) ** 2)
                h[i] = num / den

    # Normalize filter energy to 1.0
    energy = np.sum(h ** 2)
    if energy > 0:
        h = h / np.sqrt(energy)

    return h


def apply_rrc_pulse_shaping(
    symbols: np.ndarray,
    sps: int,
    beta: float = 0.35,
    span_symbols: int = 8,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Upsample and apply Root Raised Cosine pulse shaping to complex symbol stream.

    Args:
        symbols: 1D complex array of mapped baseband symbols.
        sps: Integer samples per symbol.
        beta: RRC roll-off factor.
        span_symbols: Filter span in symbols.

    Returns:
        tuple (clean_iq_samples, filter_taps, group_delay_samples)
    """
    if len(symbols) == 0:
        taps = root_raised_cosine_filter(beta=beta, span_symbols=span_symbols, sps=sps)
        return np.empty(0, dtype=np.complex64), taps, (span_symbols * sps) // 2

    taps = root_raised_cosine_filter(beta=beta, span_symbols=span_symbols, sps=sps)
    group_delay = (span_symbols * sps) // 2

    if HAS_SCIPY:
        # upfirdn inserts (sps-1) zeros between symbols and convolves with taps
        # Note: upfirdn supports complex inputs
        filtered = upfirdn(h=taps, x=symbols, up=sps, down=1)
    else:
        # Pure NumPy fallback
        upsampled = np.zeros(len(symbols) * sps, dtype=np.complex128)
        upsampled[::sps] = symbols
        filtered = np.convolve(upsampled, taps, mode="full")

    return filtered.astype(np.complex64), taps, group_delay
