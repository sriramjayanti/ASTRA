"""
fsk_sync.py
Frequency-centric synchronization pipeline for 2-FSK and 4-FSK waveforms.
Avoids Costas loops and performs tone centering, discriminator filtering, and strobe alignment.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np


def extract_instantaneous_frequency(
    iq: np.ndarray,
    sample_rate_hz: float
) -> np.ndarray:
    """
    Extract instantaneous frequency trajectory via phase difference discriminator:
    f_inst[n] = (Fs / 2pi) * angle( x[n] * conj(x[n-1]) )
    """
    if len(iq) < 2:
        return np.zeros(len(iq), dtype=np.float32)
    
    diff = iq[1:] * np.conj(iq[:-1])
    inst_freq = np.angle(diff) * (sample_rate_hz / (2.0 * np.pi))
    return np.concatenate([[inst_freq[0]], inst_freq]).astype(np.float32)


def fsk_tone_center_correction(
    iq: np.ndarray,
    sample_rate_hz: float,
    symbol_rate_hz: float
) -> Tuple[np.ndarray, float, float]:
    """
    Estimate and correct carrier frequency offset across FSK tones using power spectrum symmetry.
    """
    fft_size = min(8192, len(iq))
    centered_iq = iq[:fft_size] - np.mean(iq[:fft_size])
    spec = np.fft.fftshift(np.fft.fft(centered_iq * np.blackman(fft_size), n=fft_size))
    freqs = np.fft.fftshift(np.fft.fftfreq(fft_size, d=1.0 / sample_rate_hz))
    mag_sq = np.abs(spec) ** 2

    # Smooth spectrum to avoid noise spikes
    kernel = np.ones(7) / 7.0
    smooth_mag = np.convolve(mag_sq, kernel, mode="same")

    max_search = min(0.40 * sample_rate_hz, 4.0 * symbol_rate_hz)
    mask = (freqs >= -max_search) & (freqs <= max_search)
    valid_mag = smooth_mag[mask]
    valid_freqs = freqs[mask]

    peak1_idx = np.argmax(valid_mag)
    f1 = valid_freqs[peak1_idx]

    # Find second tone peak
    min_sep = 0.25 * symbol_rate_hz
    mask2 = np.abs(valid_freqs - f1) > min_sep
    if np.any(mask2):
        peak2_idx = np.argmax(valid_mag[mask2])
        f2 = valid_freqs[mask2][peak2_idx]
        offset_hz = float(0.5 * (f1 + f2))
    else:
        offset_hz = float(f1)

    # Correct offset
    n = np.arange(len(iq), dtype=np.float64)
    rotator = np.exp(-1j * 2.0 * np.pi * offset_hz * n / sample_rate_hz).astype(np.complex64)
    corrected_iq = (iq * rotator).astype(np.complex64)

    balance = float(min(1.0, 1.0 / (1.0 + abs(offset_hz) / max(symbol_rate_hz, 1.0))))
    return corrected_iq, offset_hz, balance


def fsk_timing_and_symbol_extraction(
    iq: np.ndarray,
    sample_rate_hz: float,
    symbol_rate_hz: float,
    sps: float
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Extract aligned symbol-center samples and instantaneous frequency deviations for FSK.
    """
    inst_f = extract_instantaneous_frequency(iq, sample_rate_hz)
    
    # Smooth instantaneous frequency slightly (over ~1/4 symbol) to reject high frequency noise
    smooth_len = max(3, int(round(sps / 4.0))) | 1
    kernel = np.ones(smooth_len) / float(smooth_len)
    inst_f_smooth = np.convolve(inst_f, kernel, mode="same")

    # Find transitions: |d(inst_f)/dt|
    transitions = np.abs(np.diff(inst_f_smooth))
    transitions = np.concatenate([[transitions[0]], transitions])

    num_symbols = int(np.floor(len(iq) / sps))
    if num_symbols < 2:
        return iq.copy(), inst_f_smooth[:len(iq)], 0.0

    # Grid search for the optimal symbol strobe phase
    # Optimal phase is where transitions are MINIMAL (symbol centers)
    # Search fractional phase grid over [0, sps)
    phase_steps = max(16, min(128, int(round(sps * 4))))
    phase_grid = np.linspace(0.0, sps, phase_steps, endpoint=False)

    best_phase = 0.0
    min_trans_energy = float("inf")
    sample_indices = np.arange(len(iq), dtype=np.float64)

    for phase_offset in phase_grid:
        sub_pos = phase_offset + np.arange(num_symbols, dtype=np.float64) * sps
        valid_mask = (sub_pos >= 0) & (sub_pos < len(iq))
        if np.count_nonzero(valid_mask) > 8:
            trans_vals = np.interp(sub_pos[valid_mask], sample_indices, transitions)
            trans_score = float(np.mean(trans_vals))
            if trans_score < min_trans_energy:
                min_trans_energy = trans_score
                best_phase = float(phase_offset)

    opt_positions = best_phase + np.arange(num_symbols, dtype=np.float64) * sps
    valid_mask = (opt_positions >= 0) & (opt_positions < len(iq))
    opt_positions = opt_positions[valid_mask]

    # Fractional interpolation for complex IQ samples and instantaneous frequency
    iq_real_interp = np.interp(opt_positions, sample_indices, np.real(iq))
    iq_imag_interp = np.interp(opt_positions, sample_indices, np.imag(iq))
    symbol_samples = (iq_real_interp + 1j * iq_imag_interp).astype(np.complex64)
    symbol_freqs = np.interp(opt_positions, sample_indices, inst_f_smooth)

    # Timing lock score based on peak-to-trough transition ratio
    mean_trans = float(np.mean(transitions)) + 1e-6
    timing_lock = float(max(0.0, min(1.0, 1.0 - (min_trans_energy / mean_trans))))

    return symbol_samples, symbol_freqs, timing_lock


def synchronize_fsk_pipeline(
    iq: np.ndarray,
    sample_rate_hz: float,
    symbol_rate_hz: float,
    modulation: str = "2-FSK",
    config: Optional[Dict[str, Any]] = None
) -> Tuple[np.ndarray, np.ndarray, float, float, float, Dict[str, Any]]:
    """
    Dedicated FSK synchronization route.
    """
    sps = float(sample_rate_hz) / float(symbol_rate_hz)
    
    # 1. Tone Centering
    centered_iq, cfo_est, balance = fsk_tone_center_correction(iq, sample_rate_hz, symbol_rate_hz)

    # 2. Timing Recovery & Symbol Alignment
    symbol_samples, symbol_freqs, timing_lock = fsk_timing_and_symbol_extraction(
        iq=centered_iq,
        sample_rate_hz=sample_rate_hz,
        symbol_rate_hz=symbol_rate_hz,
        sps=sps
    )

    meta = {
        "pipeline": "fsk_frequency_centric",
        "estimated_cfo_hz": float(cfo_est),
        "tone_balance": float(balance),
        "timing_lock_score": float(timing_lock),
        "symbol_count": len(symbol_samples)
    }
    return centered_iq, symbol_samples, cfo_est, timing_lock, balance, meta
