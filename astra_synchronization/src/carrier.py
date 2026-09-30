"""
carrier.py
Carrier recovery coordinator supporting Costas Loop (PSK) and Decision-Directed PLL (QAM).
Supports 16-QAM, 64-QAM, and 256-QAM rectangular constellations.
"""

from typing import Tuple, Dict, Any, Optional, List
import numpy as np
from .costas import costas_recover, get_phase_ambiguity_states
from .models import CarrierState


def qam_slicer_decision(y: complex, modulation: str = "16QAM") -> complex:
    """Nearest constellation point slicer for rectangular QAM (16QAM, 64QAM, 256QAM)."""
    re = float(np.real(y))
    im = float(np.imag(y))
    mod = modulation.upper().replace("-", "")

    if "256" in mod:
        # 256-QAM: 16 levels [-15, -13, ..., +15] / sqrt(170)
        levels = np.arange(-15, 17, 2, dtype=np.float64) / np.sqrt(170.0)
    elif "64" in mod:
        # 64-QAM: 8 levels [-7, -5, -3, -1, 1, 3, 5, 7] / sqrt(42)
        levels = np.array([-7, -5, -3, -1, 1, 3, 5, 7], dtype=np.float64) / np.sqrt(42.0)
    else:
        # 16-QAM: 4 levels [-3, -1, 1, 3] / sqrt(10)
        levels = np.array([-3, -1, 1, 3], dtype=np.float64) / np.sqrt(10.0)

    dec_re = levels[np.argmin(np.abs(re - levels))]
    dec_im = levels[np.argmin(np.abs(im - levels))]
    return complex(dec_re + 1j * dec_im)


def qam_carrier_recover(
    symbols_in: np.ndarray,
    modulation: str = "16QAM",
    loop_bw: float = 0.015,
    damping: float = 0.707,
    initial_state: Optional[CarrierState] = None
) -> Tuple[np.ndarray, float, float, CarrierState, List[float]]:
    """
    Two-Stage Decision-Directed PLL for Rectangular QAM carrier recovery.
    Includes residual frequency pre-estimation via 4th-power on symbols.
    """
    if len(symbols_in) < 4:
        return symbols_in.copy(), 0.0, 0.0, CarrierState(), get_phase_ambiguity_states(modulation)

    state = initial_state or CarrierState()
    ambiguities = get_phase_ambiguity_states(modulation)
    n_symbols = len(symbols_in)

    # 1. Pre-estimate any residual symbol-rate frequency rotation using 4th-power
    if n_symbols >= 64:
        s_norm = symbols_in / (np.sqrt(np.mean(np.abs(symbols_in)**2)) + 1e-8)
        r4 = s_norm ** 4
        r4 = r4 - np.mean(r4)
        fft_len = min(2048, len(r4))
        spec = np.fft.fftshift(np.fft.fft(r4[:fft_len] * np.blackman(fft_len), n=4096))
        freqs = np.fft.fftshift(np.fft.fftfreq(4096, d=1.0))  # cycles/symbol
        center = 2048
        spec[center - 4: center + 5] = 0.0  # notch DC
        peak_idx = np.argmax(np.abs(spec))
        res_freq_rot = float(freqs[peak_idx]) / 4.0  # cycles/symbol
        if abs(res_freq_rot) < 0.15 and np.max(np.abs(spec)) > 2.0 * np.mean(np.abs(spec)):
            rot = np.exp(-1j * 2.0 * np.pi * res_freq_rot * np.arange(n_symbols))
            symbols_in = symbols_in * rot

    # 2. Decision-Directed Tracking Loop
    acq_limit = max(16, int(0.25 * n_symbols))
    bw_acq = min(0.04, loop_bw * 2.0)
    kp_acq = 2.0 * 1.0 * bw_acq
    ki_acq = bw_acq ** 2

    bw_trk = max(0.003, loop_bw * 0.7)
    kp_trk = 2.0 * damping * bw_trk
    ki_trk = bw_trk ** 2

    phase = state.phase_rad
    integrator = state.integrator

    corrected_symbols = []
    phase_errors = []

    for k in range(n_symbols):
        if k < acq_limit:
            kp, ki = kp_acq, ki_acq
        else:
            kp, ki = kp_trk, ki_trk

        rotator = np.exp(-1j * phase)
        y = symbols_in[k] * rotator

        dec = qam_slicer_decision(y, modulation)
        # Decision-directed phase error: angle between y and decision point
        err = float(np.imag(y * np.conj(dec)))
        norm_factor = float(np.abs(dec)**2 + 1e-6)
        err = np.clip(err / norm_factor, -1.0, 1.0)

        phase_errors.append(err)
        corrected_symbols.append(y)

        integrator += ki * err
        integrator = np.clip(integrator, -0.15, 0.15)
        phase += kp * err + integrator
        phase = float((phase + np.pi) % (2.0 * np.pi) - np.pi)

    out_symbols = np.array(corrected_symbols, dtype=np.complex64)
    state.phase_rad = phase
    state.integrator = integrator
    state.phase_error_history = phase_errors

    tail_errors = phase_errors[acq_limit:] if len(phase_errors) > acq_limit else phase_errors
    var_err = float(np.var(tail_errors)) if tail_errors else 1.0
    carrier_lock_score = float(max(0.0, min(1.0, 1.0 / (1.0 + 3.0 * var_err))))

    return out_symbols, phase, carrier_lock_score, state, ambiguities


def recover_carrier(
    symbols_in: np.ndarray,
    modulation: str = "QPSK",
    modulation_family: str = "PSK",
    config: Optional[Dict[str, Any]] = None
) -> Tuple[np.ndarray, float, float, List[float], Dict[str, Any]]:
    """
    Route fine carrier and phase recovery based on modulation family.
    """
    cfg = config or {}
    fam = modulation_family.upper()
    mod = modulation.upper().replace("-", "")

    if "QAM" in mod or fam == "QAM":
        loop_bw = float(cfg.get("qam_loop_bw", 0.015))
        symbols, phase, lock_score, state, amb = qam_carrier_recover(
            symbols_in=symbols_in,
            modulation=modulation,
            loop_bw=loop_bw
        )
        method = "qam_decision_directed_pll"
    else:
        # PSK, DQPSK, and general linear modulations
        loop_bw = float(cfg.get("costas_loop_bw", 0.02))
        symbols, phase, lock_score, state, amb = costas_recover(
            symbols_in=symbols_in,
            modulation=modulation,
            loop_bw=loop_bw
        )
        method = "costas_loop"

    meta = {
        "carrier_method": method,
        "estimated_phase_rad": float(phase),
        "carrier_lock_score": float(lock_score),
        "phase_ambiguity_count": len(amb)
    }
    return symbols, phase, lock_score, amb, meta
