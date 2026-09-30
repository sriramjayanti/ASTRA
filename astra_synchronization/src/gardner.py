"""
gardner.py
Non-data-aided Gardner Timing Error Detector (TED) and interpolating timing recovery loop.
Features two-state operation: ACQUISITION (wide bandwidth) and TRACKING (narrow bandwidth).
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
from .models import TimingState


def cubic_interpolate(samples: np.ndarray, index_float: float) -> complex:
    """Hermite / cubic interpolation for continuous fractional sample delay."""
    idx = int(np.floor(index_float))
    mu = float(index_float - idx)

    if idx < 1 or idx >= len(samples) - 2:
        return complex(samples[max(0, min(len(samples) - 1, idx))])

    y0 = samples[idx - 1]
    y1 = samples[idx]
    y2 = samples[idx + 1]
    y3 = samples[idx + 2]

    c0 = y1
    c1 = 0.5 * (y2 - y0)
    c2 = y0 - 2.5 * y1 + 2.0 * y2 - 0.5 * y3
    c3 = 0.5 * (y3 - y0) + 1.5 * (y1 - y2)

    return complex(c0 + mu * (c1 + mu * (c2 + mu * c3)))


def gardner_recover(
    iq: np.ndarray,
    sps: float = 2.0,
    loop_gain_kp: float = 0.015,
    loop_gain_ki: float = 0.0005,
    initial_state: Optional[TimingState] = None,
    **kwargs
) -> Tuple[np.ndarray, float, float, TimingState]:
    """
    Two-state Gardner timing recovery operating on 2 samples/symbol.
    
    Error Formula:
      e[k] = Re{ (y[k] - y[k-1]) * conj(y[k - 1/2]) }
      
    State Transition:
      - First 25% symbols: ACQUISITION (wide bandwidth, Kp = 2.5 * loop_gain_kp)
      - Next 75% symbols:  TRACKING    (narrow bandwidth, Kp = 0.8 * loop_gain_kp)
      
    Sign Convention:
      When sampling is late (e[k] > 0), next index must advance earlier (subtract correction).
    """
    if len(iq) < 8:
        return iq.copy(), 0.0, 0.0, TimingState()

    state = initial_state or TimingState()
    timing_errors = []
    symbol_samples = []

    current_index = 2.0
    nominal_step = float(sps)
    half_step = nominal_step / 2.0
    integrator = 0.0
    prev_symbol = cubic_interpolate(iq, 0.0)

    total_est_symbols = int(len(iq) / nominal_step)
    acq_limit = max(16, int(0.25 * total_est_symbols))

    # Loop gains based on configured parameters
    kp_acq, ki_acq = loop_gain_kp * 2.5, loop_gain_ki * 3.0
    kp_trk, ki_trk = loop_gain_kp * 0.7, loop_gain_ki * 0.5

    while current_index < len(iq) - 2:
        k = len(symbol_samples)
        # Adapt loop gains based on acquisition vs tracking state
        if k < acq_limit:
            kp, ki = kp_acq, ki_acq
        else:
            kp, ki = kp_trk, ki_trk

        curr_symbol = cubic_interpolate(iq, current_index)
        mid_sample = cubic_interpolate(iq, current_index - half_step)

        # Gardner TED: Real part of (curr - prev) * conj(mid)
        diff = curr_symbol - prev_symbol
        error = float(np.real(diff) * np.real(mid_sample) + np.imag(diff) * np.imag(mid_sample))
        norm_power = float(np.abs(curr_symbol)**2 + np.abs(prev_symbol)**2 + 1e-6)
        error = np.clip(error / norm_power, -1.5, 1.5)

        timing_errors.append(error)
        symbol_samples.append(curr_symbol)

        integrator += ki * error
        # Clamp integrator to prevent runaway
        integrator = np.clip(integrator, -0.2 * nominal_step, 0.2 * nominal_step)
        timing_correction = kp * error + integrator

        # Crucial fix: Late strobe (error > 0) requires advancing strobe (subtracting correction)
        current_index += nominal_step - timing_correction
        prev_symbol = curr_symbol

    symbols_out = np.array(symbol_samples, dtype=np.complex64)
    state.timing_error_history = timing_errors

    # Calculate timing lock score on tracking segment
    if len(timing_errors) > 20:
        tail_errors = timing_errors[acq_limit:]
        error_var = float(np.var(tail_errors)) if len(tail_errors) > 0 else 1.0
        lock_score = float(max(0.0, min(1.0, 1.0 / (1.0 + 2.5 * error_var))))
    else:
        lock_score = 0.65

    mean_offset = float(np.mean(timing_errors[-acq_limit:])) if len(timing_errors) >= acq_limit else 0.0
    return symbols_out, mean_offset, lock_score, state
