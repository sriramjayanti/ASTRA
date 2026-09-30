"""
ASTRA DSP Feature Extractor Adapter.
Extracts 36 structured physical and statistical features from complex IQ bursts.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple
import numpy as np
from scipy import signal, stats
from scipy.signal import find_peaks

from .feature_schema import FEATURE_COLUMNS


def extract_dsp_feature_dict(
    iq: np.ndarray,
    sample_rate_hz: float = 192000.0,
) -> Dict[str, float]:
    """
    Extracts complete, normalized 36-feature dictionary from raw/preconditioned IQ samples.
    """
    x = np.asarray(iq)
    if len(x) < 32:
        return {col: float(np.nan) for col in FEATURE_COLUMNS}

    # Zero-mean DC removal
    x_zm = x - np.mean(x)
    pwr = np.mean(np.abs(x_zm) ** 2)
    rms = np.sqrt(max(1e-12, pwr))
    x_norm = x_zm / rms

    n = len(x_norm)
    dt = 1.0 / max(1.0, sample_rate_hz)

    # ==========================================
    # 1. Amplitude & Envelope Features
    # ==========================================
    env = np.abs(x_norm)
    mean_mag = float(np.mean(env))
    var_mag = float(np.var(env))
    skew_mag = float(stats.skew(env)) if len(env) > 2 else 0.0
    kurt_mag = float(stats.kurtosis(env)) if len(env) > 3 else 0.0
    peak_val = float(np.max(env))
    crest_factor = float(peak_val / (mean_mag + 1e-12))
    papr = float((peak_val ** 2) / (pwr + 1e-12))
    env_var = float(np.var(env ** 2))

    # ==========================================
    # 2. Phase & Frequency Features
    # ==========================================
    prod = x_norm[1:] * np.conj(x_norm[:-1])
    dphi = np.angle(prod)  # [-pi, +pi]
    mean_dphi = float(np.mean(dphi))
    var_dphi = float(np.var(dphi))
    
    # Circular variance
    r_bar = float(np.abs(np.mean(np.exp(1j * dphi))))
    circ_var = float(1.0 - r_bar)
    phase_conc = float(r_bar)

    # Instantaneous frequency in Hz
    f_inst = dphi * (sample_rate_hz / (2.0 * np.pi))
    f_mean = float(np.mean(f_inst))
    f_std = float(np.std(f_inst))
    f_kurt = float(stats.kurtosis(f_inst)) if len(f_inst) > 3 else 0.0

    # Dominant frequency states count (histogram peaks of inst-freq)
    f_hist, _ = np.histogram(f_inst, bins=32)
    f_peaks, _ = find_peaks(f_hist, prominence=np.max(f_hist) * 0.15) if len(f_hist) > 0 else ([], None)
    dom_states = float(len(f_peaks))

    # ==========================================
    # 3. Spectral Features (Welch PSD & FFT)
    # ==========================================
    seg_len = min(n, 1024)
    freqs, psd = signal.welch(x_norm, fs=sample_rate_hz, nperseg=seg_len, return_onesided=False)
    freqs = np.fft.fftshift(freqs)
    psd = np.fft.fftshift(psd)
    total_psd = np.sum(psd) + 1e-12

    # Centroid & Spread
    centroid = float(np.sum(freqs * psd) / total_psd)
    spread = float(np.sqrt(max(0.0, np.sum(((freqs - centroid) ** 2) * psd) / total_psd)))

    # Flatness (Wiener entropy)
    psd_pos = np.clip(psd, 1e-15, None)
    geom_mean = np.exp(np.mean(np.log(psd_pos)))
    arith_mean = np.mean(psd_pos)
    flatness = float(geom_mean / (arith_mean + 1e-15))
    psd_var = float(np.var(psd / total_psd))

    # 99% Occupied Bandwidth & 85% Rolloff
    cum_psd = np.cumsum(psd) / total_psd
    idx_low = np.searchsorted(cum_psd, 0.005)
    idx_high = np.searchsorted(cum_psd, 0.995)
    idx_roll = np.searchsorted(cum_psd, 0.85)
    obw_hz = float(abs(freqs[min(len(freqs)-1, idx_high)] - freqs[min(len(freqs)-1, idx_low)]))
    rolloff_hz = float(abs(freqs[min(len(freqs)-1, idx_roll)] - freqs[0]))

    # ==========================================
    # 4. IQ Statistical Moments & Higher-Order Cumulants
    # ==========================================
    i_part = np.real(x_norm)
    q_part = np.imag(x_norm)
    i_var = float(np.var(i_part))
    q_var = float(np.var(q_part))
    cov_iq = float(np.cov(i_part, q_part)[0, 1]) if len(i_part) > 1 else 0.0
    
    corr_iq = 0.0
    if (i_var * q_var) > 1e-12:
        corr_iq = float(cov_iq / np.sqrt(i_var * q_var))

    # Circularity coefficient rho = |E[x^2]| / E[|x|^2]
    e_x2 = np.mean(x_norm ** 2)
    e_abs2 = np.mean(np.abs(x_norm) ** 2)
    circularity = float(np.abs(e_x2) / (e_abs2 + 1e-12))

    # Higher-Order Cumulants:
    # C40 = E[x^4] - 3*(E[x^2])^2
    # C42 = E[|x|^4] - |E[x^2]|^2 - 2*(E[|x|^2])^2
    e_x4 = np.mean(x_norm ** 4)
    e_abs4 = np.mean(np.abs(x_norm) ** 4)
    c40 = complex(e_x4 - 3.0 * (e_x2 ** 2))
    c42 = complex(e_abs4 - (np.abs(e_x2) ** 2) - 2.0 * (e_abs2 ** 2))
    c40_mag = float(np.abs(c40))
    c42_mag = float(np.abs(c42))

    # ==========================================
    # 5. Temporal & Autocorrelation
    # ==========================================
    # Zero Crossing Rate
    zcr = float(np.mean(np.abs(np.diff(np.sign(i_part))) > 0))

    # Envelope Autocorrelation max peak
    env_c = env - np.mean(env)
    r_env = np.correlate(env_c, env_c, mode='full')
    r_half = r_env[len(env_c)-1 : len(env_c) + min(len(env_c), 256)]
    if np.max(np.abs(r_half)) > 1e-12:
        r_half = r_half / np.max(np.abs(r_half))

    peaks_ac, props_ac = find_peaks(r_half[2:], prominence=0.02) if len(r_half) > 2 else ([], None)
    if len(peaks_ac) > 0:
        max_idx = np.argmax(props_ac["prominences"])
        ac_lag = float(peaks_ac[max_idx] + 2)
        ac_score = float(props_ac["prominences"][max_idx])
    else:
        ac_lag = 0.0
        ac_score = 0.0

    # ==========================================
    # 6. Constellation & Ring Features
    # ==========================================
    amp_hist, bin_edges = np.histogram(env, bins=32)
    ring_peaks, _ = find_peaks(amp_hist, prominence=np.max(amp_hist) * 0.10) if len(amp_hist) > 0 else ([], None)
    ring_count = float(len(ring_peaks))
    cm_var = float(np.var(env))

    # Fast cluster count approximation via 2D spatial binning
    h2d, _, _ = np.histogram2d(i_part, q_part, bins=8)
    cluster_count_approx = float(np.sum(h2d > (np.max(h2d) * 0.15)))

    # ==========================================
    # 7. Signal Quality & CFO
    # ==========================================
    # M2M4 SNR Estimate
    if e_abs4 - 2.0 * (e_abs2 ** 2) > 0:
        snr_lin = np.sqrt(e_abs4 - 2.0 * (e_abs2 ** 2)) / (e_abs2 - np.sqrt(e_abs4 - 2.0 * (e_abs2 ** 2)) + 1e-12)
        snr_db = float(np.clip(10.0 * np.log10(max(1e-3, snr_lin)), -20.0, 50.0))
    else:
        p_sig = np.percentile(psd, 95)
        p_noise = np.percentile(psd, 15) + 1e-15
        snr_db = float(np.clip(10.0 * np.log10(p_sig / p_noise), -20.0, 50.0))

    # Clipping detection
    max_raw_amp = np.max(np.abs(x))
    clip_ratio = float(np.mean(np.abs(x) >= 0.99 * max_raw_amp))

    # CFO Magnitude
    cfo_hz = float(abs(freqs[np.argmax(psd)]))

    feature_dict = {
        "occupied_bandwidth_hz": obw_hz,
        "spectral_centroid_hz": centroid,
        "spectral_spread_hz": spread,
        "spectral_flatness": flatness,
        "spectral_rolloff_hz": rolloff_hz,
        "psd_variance": psd_var,
        "peak_to_average_power_ratio": papr,
        "mean_magnitude": mean_mag,
        "variance_magnitude": var_mag,
        "rms_amplitude": float(rms),
        "skewness_magnitude": skew_mag,
        "kurtosis_magnitude": kurt_mag,
        "crest_factor": crest_factor,
        "envelope_variance": env_var,
        "mean_phase_diff": mean_dphi,
        "variance_phase_diff": var_dphi,
        "circular_variance": circ_var,
        "phase_concentration": phase_conc,
        "inst_freq_mean_hz": f_mean,
        "inst_freq_std_hz": f_std,
        "inst_freq_kurtosis": f_kurt,
        "dominant_frequency_states": dom_states,
        "i_variance": i_var,
        "q_variance": q_var,
        "iq_covariance": cov_iq,
        "circularity_coefficient": circularity,
        "real_imag_correlation": corr_iq,
        "cumulant_c40_mag": c40_mag,
        "cumulant_c42_mag": c42_mag,
        "autocorr_max_peak_lag": ac_lag,
        "autocorr_max_peak_score": ac_score,
        "zero_crossing_rate": zcr,
        "constellation_cluster_count": cluster_count_approx,
        "radial_ring_count": ring_count,
        "constant_modulus_variance": cm_var,
        "estimated_snr_db": snr_db,
        "clipping_ratio": clip_ratio,
        "cfo_magnitude_hz": cfo_hz,
    }

    # Ensure all values are finite float
    for k, v in feature_dict.items():
        if not math.isfinite(v):
            feature_dict[k] = float(np.nan)

    return feature_dict
