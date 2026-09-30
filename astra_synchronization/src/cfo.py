"""
cfo.py
Carrier Frequency Offset (CFO) estimation, correction, and residual measurement.
Supports M-th power nonlinearity (PSK), 4th-power/spectral-symmetry (QAM),
and peak-pair frequency detection (FSK).
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np


def estimate_coarse_cfo_spectral(
    iq: np.ndarray,
    sample_rate_hz: float,
    max_search_fraction_fs: float = 0.20,
    fft_size: int = 8192
) -> Tuple[float, float]:
    """
    Estimate coarse carrier frequency offset via peak/center of FFT magnitude.
    Removes DC bin to prevent locking to residual local-oscillator leakage.
    Returns: (estimated_cfo_hz, peak_to_average_ratio)
    """
    n_samples = min(len(iq), fft_size)
    # Zero-mean DC removal
    centered = iq[:n_samples] - np.mean(iq[:n_samples])
    windowed = centered * np.blackman(n_samples)
    fft_out = np.fft.fftshift(np.fft.fft(windowed, n=fft_size))
    freqs = np.fft.fftshift(np.fft.fftfreq(fft_size, d=1.0 / sample_rate_hz))
    mag_sq = np.abs(fft_out) ** 2

    # Limit search range to [-max_search, +max_search]
    max_f = max_search_fraction_fs * sample_rate_hz
    mask = (freqs >= -max_f) & (freqs <= max_f)

    # Mask out DC bin and adjacent 4 bins
    center_bin = fft_size // 2
    dc_notch = 4
    mask[max(0, center_bin - dc_notch): min(fft_size, center_bin + dc_notch + 1)] = False

    if not np.any(mask):
        mask = np.ones_like(freqs, dtype=bool)

    valid_mag = mag_sq[mask]
    valid_freqs = freqs[mask]
    peak_idx = np.argmax(valid_mag)
    est_cfo = float(valid_freqs[peak_idx])

    # Quadratic interpolation around peak for sub-bin precision
    if 0 < peak_idx < len(valid_mag) - 1:
        alpha = valid_mag[peak_idx - 1]
        beta = valid_mag[peak_idx]
        gamma = valid_mag[peak_idx + 1]
        denom = alpha - 2.0 * beta + gamma
        if abs(denom) > 1e-12:
            delta = 0.5 * (alpha - gamma) / denom
            bin_width = freqs[1] - freqs[0]
            est_cfo += delta * bin_width

    avg_power = np.mean(valid_mag) + 1e-12
    peak_ratio = float(valid_mag[peak_idx] / avg_power)
    return est_cfo, peak_ratio


def estimate_coarse_cfo_mth_power(
    iq: np.ndarray,
    sample_rate_hz: float,
    m_order: int = 4,
    max_search_fraction_fs: float = 0.20,
    fft_size: int = 8192
) -> Tuple[float, float]:
    """
    Estimate CFO for M-PSK signals using M-th power nonlinearity.
    For BPSK (M=2) and QPSK (M=4), raising to M collapses modulation into a discrete tone at M * f_cfo.
    Crucially removes DC offset to avoid locking to the zero-frequency artifact.
    """
    n_samples = min(len(iq), fft_size)
    seg = iq[:n_samples]
    # Normalize segment amplitude
    rms = np.sqrt(np.mean(np.abs(seg)**2)) + 1e-8
    seg = seg / rms

    # M-th power
    raised = seg ** m_order
    # Zero-mean DC removal on the raised signal
    raised = raised - np.mean(raised)
    windowed = raised * np.blackman(n_samples)

    fft_out = np.fft.fftshift(np.fft.fft(windowed, n=fft_size))
    freqs = np.fft.fftshift(np.fft.fftfreq(fft_size, d=1.0 / sample_rate_hz))
    mag_sq = np.abs(fft_out) ** 2

    # Limit search range within Nyquist of the raised tone
    max_f = min(0.48 * sample_rate_hz, max_search_fraction_fs * sample_rate_hz * m_order)
    mask = (freqs >= -max_f) & (freqs <= max_f)

    # Mask out DC bin (center 6 bins)
    center_bin = fft_size // 2
    dc_notch = 6
    mask[max(0, center_bin - dc_notch): min(fft_size, center_bin + dc_notch + 1)] = False

    if not np.any(mask):
        mask = np.ones_like(freqs, dtype=bool)

    valid_mag = mag_sq[mask]
    valid_freqs = freqs[mask]
    peak_idx = np.argmax(valid_mag)
    peak_freq = float(valid_freqs[peak_idx])

    # Quadratic interpolation around peak
    if 0 < peak_idx < len(valid_mag) - 1:
        alpha = valid_mag[peak_idx - 1]
        beta = valid_mag[peak_idx]
        gamma = valid_mag[peak_idx + 1]
        denom = alpha - 2.0 * beta + gamma
        if abs(denom) > 1e-12:
            delta = 0.5 * (alpha - gamma) / denom
            bin_width = freqs[1] - freqs[0]
            peak_freq += delta * bin_width

    est_cfo = peak_freq / float(m_order)
    avg_power = np.mean(valid_mag) + 1e-12
    peak_ratio = float(valid_mag[peak_idx] / avg_power)
    
    # If no prominent tone exists outside DC, true CFO is near zero
    if peak_ratio < 3.0:
        est_cfo = 0.0

    return est_cfo, peak_ratio


def estimate_coarse_cfo_qam(
    iq: np.ndarray,
    sample_rate_hz: float,
    symbol_rate_hz: float,
    fft_size: int = 8192
) -> Tuple[float, float]:
    """
    Estimate coarse CFO for Rectangular QAM signals.
    Combines 4th-power nonlinearity with spectral energy center-of-mass estimation.
    """
    n_samples = min(len(iq), fft_size)
    seg = iq[:n_samples]
    rms = np.sqrt(np.mean(np.abs(seg)**2)) + 1e-8
    seg = seg / rms

    # 1. 4th power on normalized signal
    raised = (seg ** 4)
    raised = raised - np.mean(raised)
    windowed = raised * np.blackman(n_samples)

    fft_out = np.fft.fftshift(np.fft.fft(windowed, n=fft_size))
    freqs = np.fft.fftshift(np.fft.fftfreq(fft_size, d=1.0 / sample_rate_hz))
    mag_sq = np.abs(fft_out) ** 2

    # Search within 4 * symbol_rate
    max_f = min(0.45 * sample_rate_hz, 4.0 * symbol_rate_hz)
    mask = (freqs >= -max_f) & (freqs <= max_f)
    center_bin = fft_size // 2
    mask[max(0, center_bin - 6): min(fft_size, center_bin + 7)] = False

    valid_mag = mag_sq[mask]
    valid_freqs = freqs[mask]
    peak_idx = np.argmax(valid_mag)
    peak_ratio = float(valid_mag[peak_idx] / (np.mean(valid_mag) + 1e-12))

    if peak_ratio > 3.0:
        est_cfo = float(valid_freqs[peak_idx]) / 4.0
        return est_cfo, peak_ratio

    # 2. Fallback: Power spectrum band-edge center of mass
    fft_raw = np.fft.fftshift(np.fft.fft((seg - np.mean(seg)) * np.blackman(n_samples), n=fft_size))
    mag_raw = np.abs(fft_raw) ** 2
    # Threshold at 50% max power to find band edges
    th = 0.5 * np.max(mag_raw)
    passband = freqs[mag_raw >= th]
    if len(passband) > 10:
        est_cfo = float(0.5 * (np.min(passband) + np.max(passband)))
        return est_cfo, 2.0

    return 0.0, 1.0


def estimate_coarse_cfo_fsk(
    iq: np.ndarray,
    sample_rate_hz: float,
    symbol_rate_hz: float,
    fft_size: int = 8192
) -> Tuple[float, float]:
    """
    Estimate center carrier frequency offset for FSK waveforms via dual-peak tone symmetry.
    """
    n_samples = min(len(iq), fft_size)
    centered = iq[:n_samples] - np.mean(iq[:n_samples])
    windowed = centered * np.blackman(n_samples)
    fft_out = np.fft.fftshift(np.fft.fft(windowed, n=fft_size))
    freqs = np.fft.fftshift(np.fft.fftfreq(fft_size, d=1.0 / sample_rate_hz))
    mag_sq = np.abs(fft_out) ** 2

    # Find dominant spectral peaks
    # Smooth spectrum slightly to avoid noise ripples
    kernel = np.ones(5) / 5.0
    smooth_mag = np.convolve(mag_sq, kernel, mode="same")
    
    # Restrict search to passband
    max_f = min(0.40 * sample_rate_hz, 4.0 * symbol_rate_hz)
    mask = (freqs >= -max_f) & (freqs <= max_f)
    valid_mag = smooth_mag[mask]
    valid_freqs = freqs[mask]

    # Find two highest peaks with minimum separation of 0.2 * Rs
    peak1_idx = np.argmax(valid_mag)
    f1 = valid_freqs[peak1_idx]
    
    # Suppress peak 1 region
    min_sep = 0.3 * symbol_rate_hz
    mask2 = np.abs(valid_freqs - f1) > min_sep
    if np.any(mask2):
        peak2_idx = np.argmax(valid_mag[mask2])
        f2 = valid_freqs[mask2][peak2_idx]
        center_cfo = float(0.5 * (f1 + f2))
        conf = float(valid_mag[peak1_idx] / (np.mean(valid_mag) + 1e-12))
        return center_cfo, conf
    
    return float(f1), 2.0


def estimate_residual_cfo_autocorr(
    iq: np.ndarray,
    sample_rate_hz: float,
    max_lag: int = 4
) -> Tuple[float, float]:
    """
    Fine residual frequency offset estimation using Kay/Fitz autocorrelation at multiple lags.
    """
    if len(iq) < 32:
        return 0.0, 0.0

    estimates = []
    weights = []
    n = len(iq)

    for m in range(1, max_lag + 1):
        r_m = np.sum(iq[m:] * np.conj(iq[:n - m]))
        ang = np.angle(r_m)
        f_est = ang * sample_rate_hz / (2.0 * np.pi * m)
        w = float(np.abs(r_m))
        estimates.append(f_est)
        weights.append(w)

    total_w = sum(weights) + 1e-12
    fine_cfo = float(sum(e * w for e, w in zip(estimates, weights)) / total_w)
    confidence = float(min(1.0, weights[0] / (np.sum(np.abs(iq)**2) + 1e-12)))
    return fine_cfo, confidence


estimate_residual_cfo = estimate_residual_cfo_autocorr


def estimate_cfo(
    iq: np.ndarray,
    sample_rate_hz: float,
    symbol_rate_hz: float = 9600.0,
    modulation: str = "QPSK",
    modulation_family: str = "PSK",
    max_search_fraction_fs: float = 0.20
) -> Tuple[float, Dict[str, Any]]:
    """
    Authoritative modulation-aware coarse CFO estimator.
    """
    mod_upper = modulation.upper()
    fam_upper = modulation_family.upper()

    if "FSK" in mod_upper:
        cfo_est, conf = estimate_coarse_cfo_fsk(
            iq, sample_rate_hz, symbol_rate_hz=symbol_rate_hz
        )
        method = "fsk_dual_peak"
    elif "QAM" in mod_upper or fam_upper == "QAM":
        cfo_est, conf = estimate_coarse_cfo_qam(
            iq, sample_rate_hz, symbol_rate_hz=symbol_rate_hz
        )
        method = "qam_4th_power_bandedge"
    elif "8PSK" in mod_upper:
        # For 8PSK, 8th power can alias easily; 4th power or spectral edge is safer
        cfo_est, conf = estimate_coarse_cfo_mth_power(
            iq, sample_rate_hz, m_order=4, max_search_fraction_fs=max_search_fraction_fs
        )
        method = "psk_4th_power"
    elif "BPSK" in mod_upper:
        cfo_est, conf = estimate_coarse_cfo_mth_power(
            iq, sample_rate_hz, m_order=2, max_search_fraction_fs=max_search_fraction_fs
        )
        method = "bpsk_square"
    else:
        # Default QPSK / DQPSK / PSK
        cfo_est, conf = estimate_coarse_cfo_mth_power(
            iq, sample_rate_hz, m_order=4, max_search_fraction_fs=max_search_fraction_fs
        )
        method = "qpsk_4th_power"

    details = {
        "method": method,
        "estimated_cfo_hz": float(cfo_est),
        "normalized_cfo": float(cfo_est / sample_rate_hz),
        "confidence": float(conf)
    }
    return cfo_est, details


def correct_cfo(
    iq: np.ndarray,
    cfo_hz: float,
    sample_rate_hz: float,
    initial_phase_rad: float = 0.0
) -> np.ndarray:
    """
    Apply carrier frequency offset correction:
    y[n] = x[n] * exp(-j * (2*pi*f_cfo*n/Fs + phi0))
    """
    if abs(cfo_hz) < 1e-6 and abs(initial_phase_rad) < 1e-6:
        return iq.copy()

    n = np.arange(len(iq), dtype=np.float64)
    phase_vector = -2.0 * np.pi * cfo_hz * (n / float(sample_rate_hz)) - initial_phase_rad
    rotator = np.exp(1j * phase_vector).astype(np.complex64)
    return (iq * rotator).astype(np.complex64)
