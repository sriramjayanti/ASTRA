"""
ASTRA Symbol-Rate Estimation Engine Utilities and Synthetic Signal Synthesizer.
Used for unit testing, candidate recall benchmarks, and generating training sets.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from scipy import signal


def generate_synthetic_signal(
    mod_type: str = "QPSK",
    symbol_rate_hz: float = 9600.0,
    sample_rate_hz: float = 192000.0,
    num_symbols: int = 1024,
    snr_db: float = 20.0,
    cfo_hz: float = 0.0,
    rolloff: float = 0.25,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Synthesizes a realistic communications IQ burst for testing DSP and ranker performance.
    """
    if seed is not None:
        np.random.seed(seed)

    sps = sample_rate_hz / symbol_rate_hz
    sps_int = max(1, int(round(sps)))

    mod = mod_type.upper().replace("-", "")

    if mod in ["BPSK"]:
        bits = np.random.randint(0, 2, num_symbols)
        symbols = 2 * bits - 1.0 + 0j
    elif mod in ["QPSK"]:
        bits_i = 2 * np.random.randint(0, 2, num_symbols) - 1.0
        bits_q = 2 * np.random.randint(0, 2, num_symbols) - 1.0
        symbols = (bits_i + 1j * bits_q) / np.sqrt(2.0)
    elif mod in ["8PSK"]:
        phases = np.random.randint(0, 8, num_symbols) * (2 * np.pi / 8.0)
        symbols = np.cos(phases) + 1j * np.sin(phases)
    elif mod in ["16QAM"]:
        grid = np.array([-3, -1, 1, 3])
        i_syms = np.random.choice(grid, num_symbols)
        q_syms = np.random.choice(grid, num_symbols)
        symbols = (i_syms + 1j * q_syms) / np.sqrt(10.0)
    elif mod in ["64QAM"]:
        grid = np.array([-7, -5, -3, -1, 1, 3, 5, 7])
        i_syms = np.random.choice(grid, num_symbols)
        q_syms = np.random.choice(grid, num_symbols)
        symbols = (i_syms + 1j * q_syms) / np.sqrt(42.0)
    elif "FSK" in mod:
        # FSK waveform generation
        m_ary = 4 if "4" in mod else 2
        fsk_symbols = np.random.randint(0, m_ary, num_symbols)
        freq_dev = symbol_rate_hz / 2.0  # standard deviation
        
        # Tone offsets
        offsets = (np.arange(m_ary) - (m_ary - 1) / 2.0) * freq_dev
        
        # Continuous phase FSK
        phase = 0.0
        iq_samples = []
        samples_per_sym = int(round(sample_rate_hz / symbol_rate_hz))
        dt = 1.0 / sample_rate_hz
        
        for s in fsk_symbols:
            freq = offsets[s]
            t = np.arange(samples_per_sym) * dt
            dphi = 2 * np.pi * freq * dt
            sym_phase = phase + 2 * np.pi * freq * t
            iq_samples.append(np.exp(1j * sym_phase))
            phase = (phase + 2 * np.pi * freq * samples_per_sym * dt) % (2 * np.pi)
            
        iq_tx = np.concatenate(iq_samples)
    else:
        # Default QPSK fallback
        bits_i = 2 * np.random.randint(0, 2, num_symbols) - 1.0
        bits_q = 2 * np.random.randint(0, 2, num_symbols) - 1.0
        symbols = (bits_i + 1j * bits_q) / np.sqrt(2.0)

    if "FSK" not in mod:
        # Upsample with pulses and apply RRC filter
        up = np.zeros(num_symbols * sps_int, dtype=np.complex64)
        up[::sps_int] = symbols

        # Raised cosine pulse shape approximation
        filter_span = 8
        num_taps = filter_span * sps_int + 1
        t = np.arange(-filter_span * sps_int / 2, filter_span * sps_int / 2 + 1) / sps_int
        h = np.sinc(t) * np.cos(np.pi * rolloff * t) / (1 - (2 * rolloff * t) ** 2 + 1e-12)
        h /= np.sum(h)

        iq_tx = signal.convolve(up, h, mode="same")

    # Apply Carrier Frequency Offset (CFO)
    n = np.arange(len(iq_tx))
    iq_cfo = iq_tx * np.exp(1j * 2 * np.pi * (cfo_hz / sample_rate_hz) * n)

    # Apply AWGN noise
    sig_power = np.mean(np.abs(iq_cfo) ** 2)
    if sig_power > 0:
        noise_power = sig_power / (10.0 ** (snr_db / 10.0))
        noise = (np.random.normal(0, np.sqrt(noise_power / 2.0), len(iq_cfo)) +
                 1j * np.random.normal(0, np.sqrt(noise_power / 2.0), len(iq_cfo)))
        iq_rx = iq_cfo + noise
    else:
        iq_rx = iq_cfo

    metadata = {
        "mod_type": mod_type,
        "true_symbol_rate_hz": float(symbol_rate_hz),
        "sample_rate_hz": float(sample_rate_hz),
        "true_sps": float(sample_rate_hz / symbol_rate_hz),
        "snr_db": float(snr_db),
        "cfo_hz": float(cfo_hz),
        "rolloff": float(rolloff),
    }

    return iq_rx.astype(np.complex64), metadata
