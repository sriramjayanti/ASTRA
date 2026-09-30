"""
utils.py
Synthetic signal generators with ground-truth parameters (CFO, phase, timing offset, SNR) for testing.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
from scipy import signal
from .matched_filter import design_rrc_filter


def generate_synthetic_symbols(modulation: str = "QPSK", num_symbols: int = 500) -> np.ndarray:
    """Generate random unit-power complex constellation symbols."""
    mod = modulation.upper()
    if "BPSK" in mod:
        bits = np.random.randint(0, 2, num_symbols)
        return (2 * bits - 1).astype(np.complex64)
    elif "QPSK" in mod or "4PSK" in mod:
        bits_i = 2 * np.random.randint(0, 2, num_symbols) - 1
        bits_q = 2 * np.random.randint(0, 2, num_symbols) - 1
        return ((bits_i + 1j * bits_q) / np.sqrt(2.0)).astype(np.complex64)
    elif "8PSK" in mod:
        phases = np.random.randint(0, 8, num_symbols) * (2.0 * np.pi / 8.0)
        return np.exp(1j * phases).astype(np.complex64)
    elif "16-QAM" in mod or "16QAM" in mod:
        levels = np.array([-3, -1, 1, 3])
        i_sym = np.random.choice(levels, num_symbols)
        q_sym = np.random.choice(levels, num_symbols)
        return ((i_sym + 1j * q_sym) / np.sqrt(10.0)).astype(np.complex64)
    else:
        # Default QPSK
        bits_i = 2 * np.random.randint(0, 2, num_symbols) - 1
        bits_q = 2 * np.random.randint(0, 2, num_symbols) - 1
        return ((bits_i + 1j * bits_q) / np.sqrt(2.0)).astype(np.complex64)


def generate_synthetic_test_signal(
    modulation: str = "QPSK",
    num_symbols: int = 600,
    symbol_rate_hz: float = 9600.0,
    sample_rate_hz: float = 192000.0,
    cfo_hz: float = 1200.0,
    phase_offset_rad: float = 0.5,
    timing_offset_samples: float = 0.25,
    snr_db: float = 25.0,
    rolloff: float = 0.35
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Generate synthetic IQ signal with known impairments for ground-truth synchronization validation.
    """
    sps = float(sample_rate_hz) / float(symbol_rate_hz)
    
    if "FSK" in modulation.upper():
        # Generate FSK waveform directly via frequency modulation
        m_tones = 4 if "4" in modulation else 2
        syms = np.random.randint(0, m_tones, num_symbols)
        freq_dev = symbol_rate_hz / 2.0  # standard h=1.0 or h=0.5
        tone_freqs = (syms - (m_tones - 1) / 2.0) * freq_dev
        
        # Upsample frequency trajectory to sample rate
        inst_f_up = np.repeat(tone_freqs, int(round(sps)))
        phase_accum = 2.0 * np.pi * np.cumsum(inst_f_up) / sample_rate_hz
        tx_signal = np.exp(1j * (phase_accum + phase_offset_rad)).astype(np.complex64)
    else:
        # Linear digital modulation (PSK / QAM)
        symbols = generate_synthetic_symbols(modulation, num_symbols)
        
        # Pulse shaping with RRC
        rrc_h = design_rrc_filter(sps=sps, rolloff=rolloff, span_symbols=8)
        
        # Upsample symbols with zeros
        upsampled = np.zeros(int(round(num_symbols * sps)), dtype=np.complex64)
        indices = (np.arange(num_symbols) * sps).astype(int)
        valid_idx = indices[indices < len(upsampled)]
        upsampled[valid_idx] = symbols[:len(valid_idx)]
        
        # Convolve with RRC transmit filter
        tx_signal = signal.convolve(upsampled, rrc_h, mode="same").astype(np.complex64)

    # 1. Apply fractional timing offset via fractional delay FIR
    if abs(timing_offset_samples) > 1e-4:
        # Fractional delay filter
        n_taps = 21
        t = np.arange(-(n_taps // 2), n_taps // 2 + 1) - timing_offset_samples
        sinc_filter = np.sinc(t) * np.hamming(n_taps)
        sinc_filter = sinc_filter / np.sum(sinc_filter)
        tx_signal = signal.convolve(tx_signal, sinc_filter, mode="same").astype(np.complex64)

    # 2. Apply Carrier Frequency Offset (CFO) and initial phase
    n = np.arange(len(tx_signal), dtype=np.float64)
    rotator = np.exp(1j * (2.0 * np.pi * cfo_hz * n / sample_rate_hz + phase_offset_rad)).astype(np.complex64)
    cfo_signal = (tx_signal * rotator).astype(np.complex64)

    # 3. Add AWGN noise according to target SNR
    sig_power = float(np.mean(np.abs(cfo_signal) ** 2))
    noise_power = sig_power / (10.0 ** (snr_db / 10.0))
    noise_std = np.sqrt(noise_power / 2.0)
    noise = (np.random.randn(len(cfo_signal)) + 1j * np.random.randn(len(cfo_signal))) * noise_std
    rx_signal = (cfo_signal + noise).astype(np.complex64)

    ground_truth = {
        "modulation": modulation,
        "symbol_rate_hz": float(symbol_rate_hz),
        "sample_rate_hz": float(sample_rate_hz),
        "sps": float(sps),
        "cfo_hz": float(cfo_hz),
        "phase_offset_rad": float(phase_offset_rad),
        "timing_offset_samples": float(timing_offset_samples),
        "snr_db": float(snr_db),
        "rolloff": float(rolloff)
    }

    return rx_signal, ground_truth
