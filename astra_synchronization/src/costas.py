"""
costas.py
Two-stage Costas Loop Phase and Carrier Recovery for BPSK, QPSK, 8PSK, and DQPSK.
Implements ACQUISITION (wide bandwidth) and TRACKING (narrow bandwidth) states.
"""

from typing import Tuple, List, Optional
import numpy as np
from .models import CarrierState


def get_phase_ambiguity_states(modulation: str) -> List[float]:
    """Return fundamental rotational phase ambiguities for modulation."""
    mod = modulation.upper()
    if "BPSK" in mod:
        return [0.0, float(np.pi)]
    elif "QPSK" in mod or "4PSK" in mod or "OQPSK" in mod or "DQPSK" in mod:
        return [0.0, float(np.pi / 2.0), float(np.pi), float(3.0 * np.pi / 2.0)]
    elif "8PSK" in mod:
        return [float(k * np.pi / 4.0) for k in range(8)]
    elif "QAM" in mod:
        return [0.0, float(np.pi / 2.0), float(np.pi), float(3.0 * np.pi / 2.0)]
    return [0.0]


def costas_recover(
    symbols_in: np.ndarray,
    modulation: str = "QPSK",
    loop_bw: float = 0.02,
    damping: float = 0.707,
    initial_state: Optional[CarrierState] = None
) -> Tuple[np.ndarray, float, float, CarrierState, List[float]]:
    """
    Two-Stage Costas Loop Carrier & Phase Tracking.
    
    States:
      - ACQUISITION (first 25% symbols): wide loop bandwidth (2.5x loop_bw) for rapid pull-in
      - TRACKING    (remaining 75% symbols): nominal narrow loop bandwidth for low jitter
    """
    if len(symbols_in) < 4:
        return symbols_in.copy(), 0.0, 0.0, CarrierState(), get_phase_ambiguity_states(modulation)

    state = initial_state or CarrierState()
    mod = modulation.upper()
    ambiguities = get_phase_ambiguity_states(modulation)

    n_symbols = len(symbols_in)
    acq_limit = max(16, int(0.25 * n_symbols))

    # Pre-estimate and wipe any residual symbol-rate frequency offset using M-th power on symbols
    is_bpsk = "BPSK" in mod
    is_8psk = "8PSK" in mod
    m_pow = 2 if is_bpsk else (8 if is_8psk else 4)
    if n_symbols >= 48:
        s_norm = symbols_in / (np.sqrt(np.mean(np.abs(symbols_in)**2)) + 1e-8)
        rm = s_norm ** m_pow
        rm = rm - np.mean(rm)
        fft_len = min(2048, len(rm))
        spec = np.fft.fftshift(np.fft.fft(rm[:fft_len] * np.blackman(fft_len), n=4096))
        freqs = np.fft.fftshift(np.fft.fftfreq(4096, d=1.0))  # cycles/symbol
        center = 2048
        spec[center - 4: center + 5] = 0.0  # notch DC
        peak_idx = np.argmax(np.abs(spec))
        res_freq_rot = float(freqs[peak_idx]) / float(m_pow)
        if abs(res_freq_rot) < 0.20 and np.max(np.abs(spec)) > 2.0 * np.mean(np.abs(spec)):
            rot = np.exp(-1j * 2.0 * np.pi * res_freq_rot * np.arange(n_symbols))
            symbols_in = symbols_in * rot

    # Pre-calculate loop gains for both states with critical damping
    bw_acq = min(0.02, loop_bw * 1.2)
    kp_acq = 2.0 * damping * bw_acq
    ki_acq = (bw_acq ** 2) / 2.0

    # Tracking gains
    bw_trk = max(0.004, loop_bw * 0.5)
    kp_trk = 2.0 * damping * bw_trk
    ki_trk = (bw_trk ** 2) / 2.0

    phase = state.phase_rad
    integrator = state.integrator
    freq = state.freq_rad_per_sample

    corrected_symbols = []
    phase_errors = []

    is_bpsk = "BPSK" in mod
    is_8psk = "8PSK" in mod

    for k in range(n_symbols):
        # Select gains
        if k < acq_limit:
            kp, ki = kp_acq, ki_acq
        else:
            kp, ki = kp_trk, ki_trk

        rotator = np.exp(-1j * phase)
        y = symbols_in[k] * rotator

        re = float(np.real(y))
        im = float(np.imag(y))

        # Phase Error Detectors
        if is_bpsk:
            # BPSK Costas: e = sign(re) * im
            sign_re = 1.0 if re >= 0.0 else -1.0
            error = sign_re * im
        elif is_8psk:
            # 8PSK: Decision-directed phase error to nearest 8-PSK star point
            ang = np.angle(y)
            nearest_sector = np.round(ang / (np.pi / 4.0)) * (np.pi / 4.0)
            error = float(ang - nearest_sector)
        else:
            # QPSK / DQPSK: 4th-order Costas (sign(re)*im - sign(im)*re)
            sign_re = 1.0 if re >= 0.0 else -1.0
            sign_im = 1.0 if im >= 0.0 else -1.0
            error = sign_re * im - sign_im * re

        norm_power = float(re**2 + im**2 + 1e-6)
        error = np.clip(error / np.sqrt(norm_power), -1.5, 1.5)

        phase_errors.append(error)
        corrected_symbols.append(y)

        integrator += ki * error
        # Clamp frequency integrator to reasonable pull-in range (+/- 0.2 rad/sample)
        integrator = np.clip(integrator, -0.2, 0.2)
        freq = integrator
        phase += kp * error + freq
        # Wrap phase to [-pi, +pi]
        phase = float((phase + np.pi) % (2.0 * np.pi) - np.pi)

    out_symbols = np.array(corrected_symbols, dtype=np.complex64)
    state.phase_rad = phase
    state.freq_rad_per_sample = freq
    state.integrator = integrator
    state.phase_error_history = phase_errors

    # Lock score computed exclusively on tracking symbols
    tail_errors = phase_errors[acq_limit:] if len(phase_errors) > acq_limit else phase_errors
    var_err = float(np.var(tail_errors)) if tail_errors else 1.0
    carrier_lock_score = float(max(0.0, min(1.0, 1.0 / (1.0 + 3.5 * var_err))))

    return out_symbols, phase, carrier_lock_score, state, ambiguities
