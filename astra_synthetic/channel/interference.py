"""
Interference Generator Module for ASTRA Engine 6.
Synthesizes single-tone and multi-tone co-channel/narrowband RF interferers.
"""

from __future__ import annotations

from typing import Any
import numpy as np


def apply_sinusoidal_interference(
    iq: np.ndarray,
    sample_rate: float,
    frequency_offset_hz: float = 5000.0,
    power_relative_db: float = -10.0,
    initial_phase_rad: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Add a single-tone complex sinusoidal interferer to baseband IQ.

    Args:
        iq: 1D complex NumPy array of IQ samples.
        sample_rate: Sampling frequency F_s in Hz.
        frequency_offset_hz: Tone frequency offset relative to center (Hz).
        power_relative_db: Interference power relative to signal power (dB, e.g. -10 dB, +3 dB).
        initial_phase_rad: Initial interferer phase angle.

    Returns:
        tuple (interfered_iq, interferer_signal, parameters_dict)
    """
    if len(iq) == 0:
        return np.empty(0, dtype=np.complex64), np.empty(0, dtype=np.complex64), {}

    signal_power = float(np.mean(np.abs(iq) ** 2))
    # Target interference power P_int = P_sig * 10^(power_relative_db / 10)
    int_power = signal_power * (10.0 ** (power_relative_db / 10.0))
    int_amplitude = np.sqrt(int_power)

    n = np.arange(len(iq), dtype=np.float64)
    phase = 2.0 * np.pi * (frequency_offset_hz / sample_rate) * n + initial_phase_rad
    interferer = (int_amplitude * np.exp(1j * phase)).astype(np.complex64)

    interfered_iq = (iq + interferer).astype(np.complex64)

    params = {
        "interference_type": "single_tone",
        "frequency_offset_hz": float(frequency_offset_hz),
        "power_relative_db": float(power_relative_db),
        "initial_phase_rad": float(initial_phase_rad),
        "interference_power_measured": float(np.mean(np.abs(interferer) ** 2)),
    }

    return interfered_iq, interferer, params
