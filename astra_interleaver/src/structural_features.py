"""
structural_features.py
Structural feature extraction from candidate bitstreams:
Autocorrelation, periodicity, binary entropy, run lengths, bit balance, and LLR telemetry.
"""

from typing import Dict, Any, Optional, Tuple
import numpy as np
from .models import StructuralFeatures


def compute_binary_entropy(bits: np.ndarray) -> float:
    """Computes Shannon binary entropy in bits [0.0, 1.0]."""
    if len(bits) == 0:
        return 0.0
    p1 = float(np.mean(bits == 1))
    p0 = 1.0 - p1
    if p0 <= 1e-12 or p1 <= 1e-12:
        return 0.0
    return float(-p0 * np.log2(p0) - p1 * np.log2(p1))


def compute_run_length_statistics(bits: np.ndarray) -> Tuple[float, int, float]:
    """
    Computes run-length statistics: (mean_run_length, max_run_length, run_length_entropy).
    """
    if len(bits) == 0:
        return 0.0, 0, 0.0
    if len(bits) == 1:
        return 1.0, 1, 0.0
        
    # Find positions where value changes
    diffs = np.diff(bits)
    change_indices = np.where(diffs != 0)[0] + 1
    
    if len(change_indices) == 0:
        # All bits identical
        return float(len(bits)), len(bits), 0.0
        
    runs = np.diff(np.concatenate(([0], change_indices, [len(bits)])))
    
    mean_run = float(np.mean(runs))
    max_run = int(np.max(runs))
    
    # Run-length distribution entropy
    counts = np.bincount(runs)
    probs = counts[counts > 0] / len(runs)
    rl_entropy = float(-np.sum(probs * np.log2(probs + 1e-12)))
    
    return mean_run, max_run, rl_entropy


def compute_autocorrelation_features(bits: np.ndarray, max_lags: int = 1024) -> Tuple[float, int, float]:
    """
    Computes FFT-based normalized autocorrelation of the NRZ stream (0 -> +1, 1 -> -1).
    Returns:
        (autocorrelation_peak, peak_lag, periodicity_score)
    """
    N = len(bits)
    if N < 8:
        return 0.0, 0, 0.0
        
    # Map bits to NRZ bipolar signal: 0 -> +1.0, 1 -> -1.0
    nrz = 1.0 - 2.0 * bits.astype(np.float32)
    nrz = nrz - np.mean(nrz)  # Zero-mean
    
    var = np.var(nrz)
    if var < 1e-9:
        return 0.0, 0, 0.0
        
    # FFT correlation
    n_fft = 2 ** int(np.ceil(np.log2(2 * N)))
    spectrum = np.fft.fft(nrz, n=n_fft)
    corr = np.fft.ifft(spectrum * np.conj(spectrum)).real[:N]
    
    # Normalize by lag 0
    norm_corr = corr / (corr[0] + 1e-12)
    
    limit_lag = min(max_lags, N // 2)
    if limit_lag <= 1:
        return 0.0, 0, 0.0
        
    search_range = norm_corr[1:limit_lag]
    abs_search = np.abs(search_range)
    
    peak_val = float(np.max(abs_search))
    peak_idx = int(np.argmax(abs_search) + 1)
    
    # Periodicity score: peak relative to mean background correlation
    mean_background = float(np.mean(abs_search)) + 1e-6
    periodicity_score = float(np.clip((peak_val - mean_background) / (1.0 - mean_background + 1e-6), 0.0, 1.0))
    
    return peak_val, peak_idx, periodicity_score


def extract_structural_features(
    hard_bits: np.ndarray,
    soft_llrs: Optional[np.ndarray] = None,
    remainder_fraction: float = 0.0,
    sync_patterns: Optional[np.ndarray] = None
) -> StructuralFeatures:
    """
    Extracts comprehensive structural evidence from a deinterleaved candidate stream.
    """
    if hard_bits is None or len(hard_bits) == 0:
        return StructuralFeatures()
        
    hard_bits = np.asarray(hard_bits, dtype=np.uint8)
    entropy = compute_binary_entropy(hard_bits)
    
    p1 = float(np.mean(hard_bits == 1))
    p0 = 1.0 - p1
    
    mean_run, max_run, rl_entropy = compute_run_length_statistics(hard_bits)
    ac_peak, ac_lag, periodicity = compute_autocorrelation_features(hard_bits)
    
    mean_abs_llr = 0.0
    low_conf_fraction = 0.0
    if soft_llrs is not None and len(soft_llrs) > 0:
        soft_clean = np.nan_to_num(soft_llrs, nan=0.0, posinf=100.0, neginf=-100.0)
        abs_llrs = np.abs(soft_clean)
        mean_abs_llr = float(np.mean(abs_llrs))
        low_conf_fraction = float(np.mean(abs_llrs < 1.0))
        
    sync_corr = 0.0
    if sync_patterns is not None and len(sync_patterns) > 0:
        # Cross-correlate with known sync pattern if provided
        pat_nrz = 1.0 - 2.0 * sync_patterns.astype(np.float32)
        stream_nrz = 1.0 - 2.0 * hard_bits.astype(np.float32)
        if len(stream_nrz) >= len(pat_nrz):
            xcorr = np.correlate(stream_nrz, pat_nrz, mode='valid')
            sync_corr = float(np.max(np.abs(xcorr))) / len(pat_nrz)
            
    return StructuralFeatures(
        binary_entropy=entropy,
        bit_balance_zero_fraction=p0,
        bit_balance_one_fraction=p1,
        mean_run_length=mean_run,
        max_run_length=max_run,
        run_length_entropy=rl_entropy,
        autocorrelation_peak=ac_peak,
        peak_lag=ac_lag,
        periodicity_score=periodicity,
        remainder_fraction=remainder_fraction,
        mean_abs_llr=mean_abs_llr,
        low_confidence_fraction=low_conf_fraction,
        known_sync_correlation=sync_corr
    )
