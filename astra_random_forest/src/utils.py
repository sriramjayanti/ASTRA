"""
ASTRA Random Forest Support Utilities and Synthetic Impairment Synthesizer.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import numpy as np
from scipy import signal


def generate_impaired_signal(
    mod_type: str = "QPSK",
    sample_rate_hz: float = 192000.0,
    symbol_rate_hz: float = 9600.0,
    num_symbols: int = 1024,
    snr_db: float = 20.0,
    cfo_hz: float = 0.0,
    clipping_ratio: float = 0.0,
    multipath: bool = False,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Generates realistic communication signals with controlled channel impairments.
    """
    if seed is not None:
        np.random.seed(seed)

    mod = mod_type.upper().replace("-", "")
    sps = int(round(sample_rate_hz / max(1.0, symbol_rate_hz)))

    # 1. Base constellation / modulation synthesis
    if mod == "NOISE":
        iq_tx = (np.random.normal(0, 1, num_symbols * sps) + 1j * np.random.normal(0, 1, num_symbols * sps)).astype(np.complex64)
    elif "FSK" in mod:
        m_ary = 4 if "4" in mod else 2
        fsk_symbols = np.random.randint(0, m_ary, num_symbols)
        freq_dev = symbol_rate_hz / 2.0
        offsets = (np.arange(m_ary) - (m_ary - 1) / 2.0) * freq_dev
        phase = 0.0
        dt = 1.0 / sample_rate_hz
        iq_list = []
        for s in fsk_symbols:
            freq = offsets[s]
            t = np.arange(sps) * dt
            sym_phase = phase + 2 * np.pi * freq * t
            iq_list.append(np.exp(1j * sym_phase))
            phase = (phase + 2 * np.pi * freq * sps * dt) % (2 * np.pi)
        iq_tx = np.concatenate(iq_list)
    elif mod == "BPSK":
        bits = 2 * np.random.randint(0, 2, num_symbols) - 1.0
        up = np.zeros(num_symbols * sps, dtype=np.complex64)
        up[::sps] = bits
        h = np.ones(sps) / np.sqrt(sps)
        iq_tx = signal.convolve(up, h, mode="same")
    elif mod in ["QPSK", "4QAM"]:
        bits_i = 2 * np.random.randint(0, 2, num_symbols) - 1.0
        bits_q = 2 * np.random.randint(0, 2, num_symbols) - 1.0
        syms = (bits_i + 1j * bits_q) / np.sqrt(2.0)
        up = np.zeros(num_symbols * sps, dtype=np.complex64)
        up[::sps] = syms
        h = np.ones(sps) / np.sqrt(sps)
        iq_tx = signal.convolve(up, h, mode="same")
    elif mod == "8PSK":
        phases = np.random.randint(0, 8, num_symbols) * (2 * np.pi / 8.0)
        syms = np.cos(phases) + 1j * np.sin(phases)
        up = np.zeros(num_symbols * sps, dtype=np.complex64)
        up[::sps] = syms
        h = np.ones(sps) / np.sqrt(sps)
        iq_tx = signal.convolve(up, h, mode="same")
    elif mod == "16QAM":
        grid = np.array([-3, -1, 1, 3])
        syms = (np.random.choice(grid, num_symbols) + 1j * np.random.choice(grid, num_symbols)) / np.sqrt(10.0)
        up = np.zeros(num_symbols * sps, dtype=np.complex64)
        up[::sps] = syms
        h = np.ones(sps) / np.sqrt(sps)
        iq_tx = signal.convolve(up, h, mode="same")
    elif mod == "64QAM":
        grid = np.array([-7, -5, -3, -1, 1, 3, 5, 7])
        syms = (np.random.choice(grid, num_symbols) + 1j * np.random.choice(grid, num_symbols)) / np.sqrt(42.0)
        up = np.zeros(num_symbols * sps, dtype=np.complex64)
        up[::sps] = syms
        h = np.ones(sps) / np.sqrt(sps)
        iq_tx = signal.convolve(up, h, mode="same")
    else:
        # Default fallback
        syms = (2 * np.random.randint(0, 2, num_symbols) - 1.0 + 1j * (2 * np.random.randint(0, 2, num_symbols) - 1.0)) / np.sqrt(2.0)
        up = np.zeros(num_symbols * sps, dtype=np.complex64)
        up[::sps] = syms
        h = np.ones(sps) / np.sqrt(sps)
        iq_tx = signal.convolve(up, h, mode="same")

    # 2. Multipath channel filter
    if multipath and mod != "NOISE":
        channel_taps = np.array([1.0, 0.4 * np.exp(1j * np.pi / 4), 0.2 * np.exp(-1j * np.pi / 3)], dtype=np.complex64)
        iq_tx = signal.convolve(iq_tx, channel_taps, mode="same")

    # 3. Carrier Frequency Offset (CFO)
    n = np.arange(len(iq_tx))
    iq_cfo = iq_tx * np.exp(1j * 2 * np.pi * (cfo_hz / sample_rate_hz) * n)

    # 4. Clipping / ADC saturation
    if clipping_ratio > 0.0:
        max_level = np.percentile(np.abs(iq_cfo), 100.0 * (1.0 - clipping_ratio))
        iq_cfo = np.clip(np.real(iq_cfo), -max_level, max_level) + 1j * np.clip(np.imag(iq_cfo), -max_level, max_level)

    # 5. AWGN Noise
    sig_pwr = np.mean(np.abs(iq_cfo) ** 2)
    if sig_pwr > 1e-12:
        noise_pwr = sig_pwr / (10.0 ** (snr_db / 10.0))
        noise = (np.random.normal(0, np.sqrt(noise_pwr / 2.0), len(iq_cfo)) +
                 1j * np.random.normal(0, np.sqrt(noise_pwr / 2.0), len(iq_cfo)))
        iq_rx = iq_cfo + noise
    else:
        iq_rx = iq_cfo

    meta = {
        "mod_type": mod_type,
        "sample_rate_hz": sample_rate_hz,
        "symbol_rate_hz": symbol_rate_hz,
        "snr_db": float(snr_db),
        "cfo_hz": float(cfo_hz),
        "clipping_ratio": float(clipping_ratio),
        "multipath": bool(multipath),
    }
    return iq_rx.astype(np.complex64), meta
