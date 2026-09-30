"""
mueller_muller.py
Decision-Directed Mueller & Müller (M&M) Symbol Timing Recovery.
"""

from typing import Tuple, Optional
import numpy as np
from .models import TimingState


def slice_symbol(sample: complex, modulation: str = "QPSK") -> complex:
    """Decision slicer for standard constellations."""
    if "BPSK" in modulation.upper():
        return 1.0 + 0j if np.real(sample) >= 0 else -1.0 + 0j
    elif "QPSK" in modulation.upper() or "4PSK" in modulation.upper():
        re = 1.0 / np.sqrt(2.0) if np.real(sample) >= 0 else -1.0 / np.sqrt(2.0)
        im = 1.0 / np.sqrt(2.0) if np.imag(sample) >= 0 else -1.0 / np.sqrt(2.0)
        return re + 1j * im
    else:
        re = 1.0 if np.real(sample) >= 0 else -1.0
        im = 1.0 if np.imag(sample) >= 0 else -1.0
        return re + 1j * im


def mueller_muller_recover(
    symbols_in: np.ndarray,
    modulation: str = "QPSK",
    loop_gain_kp: float = 0.01,
    loop_gain_ki: float = 0.0002,
    initial_state: Optional[TimingState] = None
) -> Tuple[np.ndarray, float, float, TimingState]:
    """
    Mueller & Müller decision-directed timing recovery.
    """
    if len(symbols_in) < 4:
        return symbols_in.copy(), 0.0, 0.0, TimingState()

    state = initial_state or TimingState()
    timing_errors = []
    recovered_symbols = []

    prev_y = symbols_in[0]
    prev_a = slice_symbol(prev_y, modulation)
    integrator = 0.0

    for k in range(1, len(symbols_in)):
        curr_y = symbols_in[k]
        curr_a = slice_symbol(curr_y, modulation)

        err = float(np.real(prev_a * curr_y - curr_a * prev_y))
        err = np.clip(err, -2.0, 2.0)
        timing_errors.append(err)

        integrator += loop_gain_ki * err
        correction = loop_gain_kp * err + integrator

        adj_symbol = curr_y * np.exp(-1j * correction)
        recovered_symbols.append(adj_symbol)

        prev_y = adj_symbol
        prev_a = slice_symbol(adj_symbol, modulation)

    symbols_out = np.array(recovered_symbols, dtype=np.complex64)
    state.timing_error_history = timing_errors

    tail_errors = timing_errors[len(timing_errors)//2:] if len(timing_errors) > 10 else timing_errors
    error_var = float(np.var(tail_errors)) if tail_errors else 0.0
    lock_score = float(max(0.0, min(1.0, 1.0 / (1.0 + 3.0 * error_var))))

    return symbols_out, float(np.mean(timing_errors) if timing_errors else 0.0), lock_score, state
