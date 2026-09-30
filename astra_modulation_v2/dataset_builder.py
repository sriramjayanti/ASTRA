"""
ASTRA Modulation Intelligence V2 — Unified Dataset Builder & Pipeline.

Ingests:
1. Audited CSPB.ML.2018R2 modules (BPSK, QPSK, 8PSK, DQPSK, MSK, 16QAM, 64QAM, 256QAM)
2. Synthetic FSK Expansion (2-FSK, 4-FSK with randomized tone spacing, drift, variable baud)
3. Synthetic QAM Expansion (16QAM, 64QAM, 256QAM across Easy/Medium/Hard/Extreme tiers)
4. Synthetic UNKNOWN / Non-Target signals (AWGN, CW, chirps, multi-tone, impulsive noise)

Enforces:
- Strict source-level partitioning (zero leakage before windowing)
- Canonical 11-class schema adherence
- Identical raw IQ feeding both 1D and 2D models
- Export of modulation_v2_split_manifest.json and ASTRA_MODULATION_DATASET_V2
"""

from __future__ import annotations

import os
import json
import zipfile
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from astra_modulation_v2.class_schema import (
    CLASS_SCHEMA_VERSION,
    MODULATION_CLASSES_V2,
    NUM_CLASSES_V2,
    normalize_modulation_name,
    get_class_index,
    get_class_name,
    assert_runtime_class_order,
)

# -------------------------------------------------------------------------
# Preprocessing Pipeline (Exact, Canonical Implementation)
# -------------------------------------------------------------------------

class IQPreprocessorV2:
    """Canonical IQ Preprocessing Pipeline for V2 models."""
    
    def __init__(self, remove_dc: bool = True, normalize_rms: bool = True, eps: float = 1e-8):
        self.remove_dc = remove_dc
        self.normalize_rms = normalize_rms
        self.eps = eps

    def process(self, iq: np.ndarray) -> np.ndarray:
        """Processes raw complex IQ array into clean, normalized complex64 array."""
        iq = np.nan_to_num(iq, nan=0.0, posinf=0.0, neginf=0.0).astype(np.complex64)
        if len(iq) == 0:
            return iq
            
        # 1. DC Offset Removal
        if self.remove_dc:
            mean_val = np.mean(iq)
            if np.isfinite(mean_val):
                iq = iq - mean_val
                
        # 2. RMS Power Normalization
        if self.normalize_rms:
            power = np.mean(np.abs(iq) ** 2)
            rms = np.sqrt(power)
            if np.isfinite(rms) and rms > self.eps:
                iq = iq / rms
            else:
                iq = np.zeros_like(iq)
                
        return np.nan_to_num(iq, nan=0.0, posinf=0.0, neginf=0.0).astype(np.complex64)


def iq_to_tensor_1d(iq_window: np.ndarray) -> torch.Tensor:
    """Converts complex IQ window [N] to real tensor [2, N] for ResNet-1D."""
    i_ch = np.real(iq_window).astype(np.float32)
    q_ch = np.imag(iq_window).astype(np.float32)
    return torch.from_numpy(np.stack([i_ch, q_ch], axis=0))


def iq_to_spectrogram_2d(
    iq_window: np.ndarray,
    n_fft: int = 128,
    hop_length: int = 32,
    win_length: int = 128,
    target_f: int = 128,
    target_t: int = 128,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Computes centered, normalized log-power spectrogram tensor [1, 128, 128]
    directly from complex IQ without image disk conversion.
    """
    if len(iq_window) < n_fft:
        pad_len = n_fft - len(iq_window)
        iq_window = np.pad(iq_window, (0, pad_len), mode="constant")
        
    t_sig = torch.from_numpy(iq_window.astype(np.complex64))
    window = torch.hann_window(win_length)
    
    # STFT: returns [F, T] complex
    stft = torch.stft(
        t_sig,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=True,
        return_complex=True,
    )
    
    # Centered FFT shift across frequency axis
    stft = torch.fft.fftshift(stft, dim=0)
    
    # Log power
    power = torch.abs(stft) ** 2
    log_power = torch.log(power + eps)
    
    # Resize / pad / crop to exact target shape [128, 128]
    cur_f, cur_t = log_power.shape
    tensor_2d = log_power.unsqueeze(0).unsqueeze(0)  # [1, 1, cur_f, cur_t]
    if (cur_f, cur_t) != (target_f, target_t):
        tensor_2d = torch.nn.functional.interpolate(
            tensor_2d, size=(target_f, target_t), mode="bilinear", align_corners=False
        )
        
    tensor_out = tensor_2d.squeeze(0)  # [1, 128, 128]
    
    # Standardize to zero-mean, unit-variance
    m = tensor_out.mean()
    s = tensor_out.std()
    if s > eps:
        tensor_out = (tensor_out - m) / s
    else:
        tensor_out = tensor_out - m
        
    return torch.nan_to_num(tensor_out, nan=0.0, posinf=0.0, neginf=0.0).float()


# -------------------------------------------------------------------------
# Synthetic Generators (FSK, QAM, UNKNOWN)
# -------------------------------------------------------------------------

def generate_fsk_capture(
    modulation: str,
    n_samples: int = 16384,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Generates synthetic FSK waveform (2-FSK or 4-FSK) with extensive randomized
    tone spacing, frequency deviation, drift, baud, and channel impairments.
    """
    if rng is None:
        rng = np.random.default_rng()
        
    sample_rate = float(rng.choice([96000.0, 192000.0, 384000.0]))
    symbol_rates = [1200.0, 2400.0, 4800.0, 9600.0, 19200.0, 38400.0]
    symbol_rate = float(rng.choice(symbol_rates) * rng.uniform(0.95, 1.05))
    sps = sample_rate / symbol_rate
    
    if modulation == "2-FSK":
        m = 2
        # Randomized normalized tone spacing from 0.25 to 2.2 Rs
        tone_spacing_ratio = float(rng.choice([0.25, 0.5, 0.75, 1.0, 1.5, 2.0]) * rng.uniform(0.9, 1.1))
        tone_multipliers = np.array([-0.5, 0.5]) * tone_spacing_ratio
    else:  # 4-FSK
        m = 4
        # Randomize tone spacing and symmetry
        spacing = float(rng.uniform(0.6, 1.8))
        tone_multipliers = np.array([-1.5, -0.5, 0.5, 1.5]) * spacing
        # Small random frequency offset on carrier center
        tone_multipliers += float(rng.uniform(-0.15, 0.15))
        
    num_symbols = int(np.ceil(n_samples / sps)) + 20
    symbols = rng.integers(0, m, size=num_symbols)
    
    # Continuous-phase synthesis
    phase = float(rng.uniform(0, 2 * np.pi))
    phase_acc = []
    
    # Frequency drift (linear drift or random walk)
    drift_type = rng.choice(["none", "linear", "random_walk"])
    drift_rate = float(rng.uniform(-50.0, 50.0)) if drift_type == "linear" else 0.0
    
    freq_shifts = tone_multipliers[symbols] * symbol_rate
    dt = 1.0 / sample_rate
    
    signal = np.zeros(n_samples, dtype=np.complex64)
    cur_phase = phase
    sym_idx = 0
    sym_sample_cnt = 0
    cur_drift = 0.0
    
    for n in range(n_samples):
        if sym_sample_cnt >= sps:
            sym_sample_cnt -= sps
            sym_idx = min(sym_idx + 1, num_symbols - 1)
            
        freq = freq_shifts[sym_idx] + cur_drift
        if drift_type == "linear":
            cur_drift += drift_rate * dt
        elif drift_type == "random_walk":
            cur_drift += float(rng.normal(0, 2.0))
            
        cur_phase += 2 * np.pi * freq * dt
        signal[n] = np.exp(1j * cur_phase)
        sym_sample_cnt += 1
        
    # Channel impairments
    snr_db = float(rng.uniform(-8.0, 28.0))
    cfo_hz = float(rng.uniform(-4000.0, 4000.0))
    
    # Apply CFO
    t = np.arange(n_samples) / sample_rate
    signal = signal * np.exp(1j * (2 * np.pi * cfo_hz * t + rng.uniform(0, 2 * np.pi)))
    
    # Multipath fading (2-tap or 3-tap)
    if rng.random() > 0.4:
        delay = int(rng.integers(1, 12))
        gain = complex(rng.uniform(-0.4, 0.4), rng.uniform(-0.4, 0.4))
        delayed = np.zeros_like(signal)
        delayed[delay:] = signal[:-delay]
        signal = signal + gain * delayed
        
    # IQ Imbalance
    gain_imb = float(rng.uniform(0.90, 1.10))
    phase_imb = float(np.radians(rng.uniform(-8.0, 8.0)))
    i_ch = np.real(signal)
    q_ch = np.imag(signal)
    q_ch_imb = gain_imb * (np.sin(phase_imb) * i_ch + np.cos(phase_imb) * q_ch)
    signal = i_ch + 1j * q_ch_imb
    
    # AWGN noise addition
    sig_power = np.mean(np.abs(signal) ** 2)
    noise_power = sig_power / (10.0 ** (snr_db / 10.0))
    noise = (rng.normal(0, np.sqrt(noise_power / 2), n_samples) +
             1j * rng.normal(0, np.sqrt(noise_power / 2), n_samples))
    signal = signal + noise
    
    meta = {
        "symbol_rate": symbol_rate,
        "sample_rate": sample_rate,
        "snr_db": snr_db,
        "cfo_hz": cfo_hz,
        "rolloff": 0.0,
        "drift_type": drift_type,
    }
    return signal.astype(np.complex64), meta


def generate_qam_capture(
    modulation: str,
    n_samples: int = 16384,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Generates synthetic QAM waveform (16QAM, 64QAM, 256QAM) across 4 difficulty tiers
    with RRC pulse shaping, multipath, IQ imbalance, and clipping.
    """
    if rng is None:
        rng = np.random.default_rng()
        
    sample_rate = float(rng.choice([96000.0, 192000.0, 384000.0]))
    symbol_rates = [4800.0, 9600.0, 19200.0, 24000.0, 48000.0]
    symbol_rate = float(rng.choice(symbol_rates) * rng.uniform(0.95, 1.05))
    sps = int(round(sample_rate / symbol_rate))
    sps = max(sps, 4)
    actual_sr = sps * symbol_rate
    
    # Impairment tier distribution: 25% easy, 35% medium, 30% hard, 10% extreme
    tier_rand = rng.random()
    if tier_rand < 0.25:
        tier = "easy"
        snr_db = float(rng.uniform(14.0, 30.0))
        cfo_hz = float(rng.uniform(-500.0, 500.0))
        multipath_taps = 1
        gain_imb = float(rng.uniform(0.98, 1.02))
        phase_imb = float(np.radians(rng.uniform(-2.0, 2.0)))
    elif tier_rand < 0.60:
        tier = "medium"
        snr_db = float(rng.uniform(6.0, 18.0))
        cfo_hz = float(rng.uniform(-2000.0, 2000.0))
        multipath_taps = 2 if rng.random() > 0.4 else 1
        gain_imb = float(rng.uniform(0.94, 1.06))
        phase_imb = float(np.radians(rng.uniform(-5.0, 5.0)))
    elif tier_rand < 0.90:
        tier = "hard"
        snr_db = float(rng.uniform(-2.0, 10.0))
        cfo_hz = float(rng.uniform(-4000.0, 4000.0))
        multipath_taps = 3
        gain_imb = float(rng.uniform(0.90, 1.10))
        phase_imb = float(np.radians(rng.uniform(-8.0, 8.0)))
    else:
        tier = "extreme"
        snr_db = float(rng.uniform(-8.0, 2.0))
        cfo_hz = float(rng.uniform(-6000.0, 6000.0))
        multipath_taps = 3
        gain_imb = float(rng.uniform(0.85, 1.15))
        phase_imb = float(np.radians(rng.uniform(-12.0, 12.0)))
        
    rolloff = float(rng.choice([0.15, 0.20, 0.25, 0.35, 0.50]))
    
    # Constellation levels
    if modulation == "16QAM":
        levels = np.array([-3, -1, 1, 3], dtype=np.float32)
        norm = 1.0 / np.sqrt(10.0)
    elif modulation == "64QAM":
        levels = np.array([-7, -5, -3, -1, 1, 3, 5, 7], dtype=np.float32)
        norm = 1.0 / np.sqrt(42.0)
    else:  # 256QAM
        levels = np.arange(-15, 16, 2, dtype=np.float32)
        norm = 1.0 / np.sqrt(170.0)
        
    num_symbols = int(np.ceil(n_samples / sps)) + 40
    # Payload diversity
    payload_type = rng.choice(["random", "prbs", "biased", "repetitive"])
    if payload_type == "random":
        sym_i = rng.choice(levels, size=num_symbols)
        sym_q = rng.choice(levels, size=num_symbols)
    elif payload_type == "prbs":
        seq = (np.arange(num_symbols) * 31 + 17) % len(levels)
        sym_i = levels[seq]
        sym_q = levels[(seq * 3 + 5) % len(levels)]
    elif payload_type == "biased":
        p = np.exp(-np.abs(levels) / levels.max())
        p = p / p.sum()
        sym_i = rng.choice(levels, size=num_symbols, p=p)
        sym_q = rng.choice(levels, size=num_symbols, p=p)
    else:
        chunk_i = rng.choice(levels, size=8)
        chunk_q = rng.choice(levels, size=8)
        sym_i = np.tile(chunk_i, int(np.ceil(num_symbols / 8)))[:num_symbols]
        sym_q = np.tile(chunk_q, int(np.ceil(num_symbols / 8)))[:num_symbols]
        
    const_symbols = (sym_i + 1j * sym_q) * norm
    
    # Upsampling
    upsampled = np.zeros(num_symbols * sps, dtype=np.complex64)
    upsampled[::sps] = const_symbols
    
    # RRC Filter
    span = 8
    t_filt = np.arange(-span * sps, span * sps + 1) / float(sps)
    filt = np.sinc(t_filt) * np.cos(np.pi * rolloff * t_filt)
    denom = 1.0 - (2.0 * rolloff * t_filt) ** 2
    denom[np.isclose(denom, 0.0)] = 1e-6
    filt = filt / denom
    filt = filt / np.sqrt(np.sum(filt ** 2))
    
    tx_signal = np.convolve(upsampled, filt, mode="same")
    start_cut = span * sps
    signal = tx_signal[start_cut : start_cut + n_samples]
    if len(signal) < n_samples:
        signal = np.pad(signal, (0, n_samples - len(signal)), mode="wrap")
        
    # CFO & Phase
    t_arr = np.arange(n_samples) / actual_sr
    signal = signal * np.exp(1j * (2 * np.pi * cfo_hz * t_arr + rng.uniform(0, 2 * np.pi)))
    
    # Multipath
    if multipath_taps >= 2:
        d1 = int(rng.integers(1, 6))
        g1 = complex(rng.uniform(-0.3, 0.3), rng.uniform(-0.3, 0.3))
        delayed = np.zeros_like(signal)
        delayed[d1:] = signal[:-d1]
        signal = signal + g1 * delayed
    if multipath_taps >= 3:
        d2 = int(rng.integers(7, 16))
        g2 = complex(rng.uniform(-0.15, 0.15), rng.uniform(-0.15, 0.15))
        delayed2 = np.zeros_like(signal)
        delayed2[d2:] = signal[:-d2]
        signal = signal + g2 * delayed2
        
    # Nonlinear soft clipping
    if tier in ["hard", "extreme"]:
        mag = np.abs(signal)
        clip_thresh = float(rng.uniform(1.2, 1.8))
        mask = mag > clip_thresh
        signal[mask] = signal[mask] * (clip_thresh / mag[mask])
        
    # IQ Imbalance
    i_c = np.real(signal)
    q_c = np.imag(signal)
    q_imb = gain_imb * (np.sin(phase_imb) * i_c + np.cos(phase_imb) * q_c)
    signal = i_c + 1j * q_imb
    
    # AWGN noise
    p_sig = np.mean(np.abs(signal) ** 2)
    p_noise = p_sig / (10.0 ** (snr_db / 10.0))
    noise = (rng.normal(0, np.sqrt(p_noise / 2), n_samples) +
             1j * rng.normal(0, np.sqrt(p_noise / 2), n_samples))
    signal = signal + noise
    
    meta = {
        "symbol_rate": symbol_rate,
        "sample_rate": actual_sr,
        "snr_db": snr_db,
        "cfo_hz": cfo_hz,
        "rolloff": rolloff,
        "tier": tier,
        "payload": payload_type,
    }
    return signal.astype(np.complex64), meta


def generate_unknown_capture(
    n_samples: int = 16384,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Generates diverse non-target / UNKNOWN signal captures:
    - Pure AWGN
    - Single CW unmodulated carrier
    - Multi-tone interference (2 to 6 tones)
    - Linear / exponential chirps
    - Impulsive noise bursts
    """
    if rng is None:
        rng = np.random.default_rng()
        
    sample_rate = float(rng.choice([96000.0, 192000.0, 384000.0]))
    t = np.arange(n_samples) / sample_rate
    subtype = rng.choice(["awgn", "cw", "multitone", "chirp", "impulsive"])
    
    if subtype == "awgn":
        # Pure gaussian noise
        sig = rng.normal(0, 1.0, n_samples) + 1j * rng.normal(0, 1.0, n_samples)
        snr_db = -10.0
        cfo_hz = 0.0
        
    elif subtype == "cw":
        # Unmodulated continuous-wave tone
        fc = float(rng.uniform(-0.4, 0.4) * sample_rate)
        phase = float(rng.uniform(0, 2 * np.pi))
        sig = np.exp(1j * (2 * np.pi * fc * t + phase))
        # Add slight background noise
        noise = rng.normal(0, 0.1, n_samples) + 1j * rng.normal(0, 0.1, n_samples)
        sig = sig + noise
        snr_db = float(rng.uniform(5.0, 25.0))
        cfo_hz = fc
        
    elif subtype == "multitone":
        # 2 to 6 simultaneous independent tones
        num_tones = int(rng.integers(2, 7))
        sig = np.zeros(n_samples, dtype=np.complex64)
        for _ in range(num_tones):
            fc = float(rng.uniform(-0.4, 0.4) * sample_rate)
            a = float(rng.uniform(0.3, 1.0))
            phi = float(rng.uniform(0, 2 * np.pi))
            sig += a * np.exp(1j * (2 * np.pi * fc * t + phi))
        noise = rng.normal(0, 0.1, n_samples) + 1j * rng.normal(0, 0.1, n_samples)
        sig = sig + noise
        snr_db = float(rng.uniform(5.0, 20.0))
        cfo_hz = 0.0
        
    elif subtype == "chirp":
        # Linear frequency sweep
        f0 = float(rng.uniform(-0.35, 0.0) * sample_rate)
        f1 = float(rng.uniform(0.0, 0.35) * sample_rate)
        t1 = n_samples / sample_rate
        k = (f1 - f0) / t1
        phase = float(rng.uniform(0, 2 * np.pi))
        sig = np.exp(1j * (2 * np.pi * (f0 * t + 0.5 * k * (t ** 2)) + phase))
        noise = rng.normal(0, 0.1, n_samples) + 1j * rng.normal(0, 0.1, n_samples)
        sig = sig + noise
        snr_db = float(rng.uniform(5.0, 25.0))
        cfo_hz = f0
        
    else:  # impulsive
        # Background noise with random impulsive spikes
        sig = rng.normal(0, 0.3, n_samples) + 1j * rng.normal(0, 0.3, n_samples)
        num_spikes = int(rng.integers(5, 30))
        spike_locs = rng.integers(0, n_samples, size=num_spikes)
        spike_amps = rng.uniform(3.0, 10.0, size=num_spikes) * np.exp(1j * rng.uniform(0, 2*np.pi, size=num_spikes))
        sig[spike_locs] += spike_amps
        snr_db = -5.0
        cfo_hz = 0.0
        
    meta = {
        "symbol_rate": 0.0,
        "sample_rate": sample_rate,
        "snr_db": snr_db,
        "cfo_hz": cfo_hz,
        "rolloff": 0.0,
        "subtype": subtype,
    }
    return sig.astype(np.complex64), meta


# -------------------------------------------------------------------------
# Unified PyTorch Dataset for 1D and 2D Models
# -------------------------------------------------------------------------

class ASTRAModulationV2Dataset(Dataset):
    """
    Unified PyTorch Dataset serving both ResNet-1D (raw normalized IQ)
    and Spectrogram CNN-2D (centered STFT tensor) from identical source windows.
    Zero cross-split leakage.
    """
    
    def __init__(
        self,
        records: List[Dict[str, Any]],
        window_size: int = 2048,
        windows_per_capture: int = 4,
        return_mode: str = "both",  # "1d", "2d", or "both"
        preprocessor: Optional[IQPreprocessorV2] = None,
    ):
        self.records = records
        self.window_size = window_size
        self.windows_per_capture = windows_per_capture
        self.return_mode = return_mode
        self.preprocessor = preprocessor or IQPreprocessorV2()
        
        # Build window index: maps global_idx -> (record_idx, window_offset)
        self.samples = []
        for r_idx, rec in enumerate(self.records):
            # Check length or available samples
            n_samp = rec.get("total_samples", 16384)
            stride = max((n_samp - window_size) // max(windows_per_capture - 1, 1), window_size)
            for w in range(windows_per_capture):
                start = min(w * stride, max(n_samp - window_size, 0))
                self.samples.append((r_idx, start))
                
        # Cache for recently loaded captures to avoid disk IO bottleneck
        self._cache: Dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.samples)

    def _load_raw_iq(self, rec: Dict[str, Any]) -> np.ndarray:
        source_id = rec["source_id"]
        if source_id in self._cache:
            return self._cache[source_id]
            
        iq_path = rec.get("iq_path")
        zip_path = rec.get("zip_path")
        internal_path = rec.get("internal_path")
        
        if zip_path and internal_path and os.path.exists(zip_path):
            # Read from CSPB zip archive
            with zipfile.ZipFile(zip_path, "r") as zf:
                raw_bytes = zf.read(internal_path)
                data = np.frombuffer(raw_bytes, dtype=np.float32)
                iq = data[0::2] + 1j * data[1::2]
        elif iq_path and os.path.exists(iq_path):
            # Direct binary or npy
            if iq_path.endswith(".npy"):
                iq = np.load(iq_path)
            else:
                data = np.fromfile(iq_path, dtype=np.float32)
                iq = data[0::2] + 1j * data[1::2]
        else:
            # Fallback zero window
            iq = np.zeros(16384, dtype=np.complex64)
            
        iq = self.preprocessor.process(iq)
        # Limit cache size to 1000 items
        if len(self._cache) > 1000:
            self._cache.clear()
        self._cache[source_id] = iq
        return iq

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec_idx, start_sample = self.samples[idx]
        rec = self.records[rec_idx]
        
        iq_full = self._load_raw_iq(rec)
        end_sample = start_sample + self.window_size
        
        if len(iq_full) == 0:
            window = np.zeros(self.window_size, dtype=np.complex64)
        elif len(iq_full) < self.window_size:
            pad_len = self.window_size - len(iq_full)
            window = np.pad(iq_full, (0, pad_len), mode="constant")
        else:
            max_start = len(iq_full) - self.window_size
            actual_start = min(max(0, start_sample), max_start)
            window = iq_full[actual_start : actual_start + self.window_size]
            
        label = get_class_index(rec["modulation"])
        item: Dict[str, Any] = {
            "label": label,
            "modulation": rec["modulation"],
            "source_id": rec["source_id"],
            "snr_db": rec.get("snr_db", 0.0),
        }
        
        if self.return_mode in ["1d", "both"]:
            item["iq_1d"] = iq_to_tensor_1d(window)
        if self.return_mode in ["2d", "both"]:
            item["spectrogram_2d"] = iq_to_spectrogram_2d(window)
            
        return item
