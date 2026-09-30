"""
Timing Offset and Fractional Delay Impairment Module for ASTRA Engine 6.
Implements windowed-sinc fractional delay interpolation and sample-level time shifts.
"""

from __future__ import annotations

import numpy as np


def design_fractional_delay_filter(
    fractional_delay: float,
    filter_half_length: int = 16,
) -> tuple[np.ndarray, int]:
    """Design a Blackman-windowed sinc fractional delay FIR filter.

    Args:
        fractional_delay: Desired fractional sample delay tau in (-1.0, 1.0).
        filter_half_length: One-sided tap count M (total taps N = 2M + 1).

    Returns:
        tuple (filter_taps, center_delay)
    """
    m = filter_half_length
    n = np.arange(-m, m + 1, dtype=np.float64)
    # Target response: h[n] = sinc(n - tau)
    t = n - fractional_delay
    sinc_vals = np.sinc(t)  # np.sinc(x) = sin(pi*x)/(pi*x)
    # Blackman window
    window = np.blackman(len(n))
    h = sinc_vals * window
    # Normalize DC gain to 1.0
    sum_h = np.sum(h)
    if abs(sum_h) > 1e-12:
        h = h / sum_h
    return h, m


def apply_timing_offset(
    iq: np.ndarray,
    timing_offset_samples: float,
    filter_half_length: int = 16,
) -> np.ndarray:
    """Apply integer and fractional sample timing delay to baseband IQ.

    A positive timing_offset_samples delays the signal (shifts right in time).
    A negative timing_offset_samples advances the signal (shifts left in time).

    Args:
        iq: 1D complex NumPy array of IQ samples.
        timing_offset_samples: Total sample offset (can be float, e.g. 0.35, -2.4).
        filter_half_length: Filter half-length for fractional delay filter.

    Returns:
        1D complex64 array with timing offset applied.
    """
    if len(iq) == 0:
        return np.empty(0, dtype=np.complex64)

    if abs(timing_offset_samples) < 1e-6:
        return iq.astype(np.complex64)

    int_delay = int(np.floor(timing_offset_samples))
    frac_delay = float(timing_offset_samples - int_delay)

    # Design fractional delay filter
    if abs(frac_delay) < 1e-6:
        # Pure integer delay
        filtered = iq
        group_delay = 0
    else:
        h, group_delay = design_fractional_delay_filter(
            fractional_delay=frac_delay, filter_half_length=filter_half_length
        )
        filtered = np.convolve(iq, h, mode="full")

    # Shift by integer delay and compensate filter center delay
    # Output length kept equal to input length
    start_idx = group_delay - int_delay
    n_samples = len(iq)

    if start_idx >= 0:
        if start_idx < len(filtered):
            out_slice = filtered[start_idx : start_idx + n_samples]
            if len(out_slice) < n_samples:
                out = np.pad(out_slice, (0, n_samples - len(out_slice)), mode="constant")
            else:
                out = out_slice
        else:
            out = np.zeros(n_samples, dtype=np.complex64)
    else:
        # Advance (negative start_idx) -> pad leading zeros
        pad_front = -start_idx
        avail = filtered[: n_samples - pad_front]
        out = np.pad(avail, (pad_front, max(0, n_samples - (pad_front + len(avail)))), mode="constant")

    return out.astype(np.complex64)
