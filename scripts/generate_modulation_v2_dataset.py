"""
ASTRA Production Modulation V2 Synthetic Dataset Generator.
Synthesizes multi-parametric, highly diverse, balanced datasets across:
- 2-FSK, 4-FSK (Continuous-phase & discontinuous, variable tone spacing 0.25-2.5 Rs, frequency drift)
- BPSK, QPSK, 8PSK, DQPSK, MSK
- 16QAM, 64QAM, 256QAM (Rich SNR curriculum, IQ imbalance, soft clipping, multipath, roll-off)

Strictly enforces:
1. Zero Data Leakage (grouped by source capture before window slicing)
2. Canonical MODULATION_CLASSES_V2 mapping
3. Balanced difficulty distribution
4. Comprehensive metadata sidecars & split manifests
5. Dedicated OOD (Out-Of-Distribution) evaluation test set
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import scipy.signal as signal

# Ensure workspace root is in sys.path
workspace_root = Path(__file__).resolve().parent.parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

from astra_config.classes import (
    TRAINED_MODULATION_CLASSES_V2,
    MODULATION_CLASSES_V2,
    normalize_modulation_name,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("MOD_V2_GEN")


# =============================================================================
# 1. Payload Generator: Structured, Biased, and Pseudo-Random Sequences
# =============================================================================

def generate_varied_bits(num_bits: int, pattern_type: str, rng: np.random.Generator) -> np.ndarray:
    """Generates non-uniform bit distributions to prevent symbol sequence memorization."""
    if pattern_type == "random":
        return rng.integers(0, 2, size=num_bits, dtype=np.uint8)
    
    elif pattern_type == "biased":
        p1 = rng.uniform(0.15, 0.85)
        return (rng.uniform(0.0, 1.0, size=num_bits) < p1).astype(np.uint8)
    
    elif pattern_type == "alternating":
        pat = rng.choice(["01", "0011", "11110000", "10110011"])
        pat_arr = np.array([int(b) for b in pat], dtype=np.uint8)
        reps = (num_bits // len(pat_arr)) + 1
        return np.tile(pat_arr, reps)[:num_bits]
    
    elif pattern_type == "prbs":
        # Galois LFSR PN9 simulation (x^9 + x^5 + 1)
        state = int(rng.integers(1, 511))
        bits = np.zeros(num_bits, dtype=np.uint8)
        for i in range(num_bits):
            feedback = ((state >> 8) ^ (state >> 4)) & 1
            bits[i] = state & 1
            state = ((state << 1) | feedback) & 0x1FF
        return bits
    
    elif pattern_type == "framed":
        # Preamble (0xEB90) + Frame Counter + Random Payload + Parity
        preamble = np.array([1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0], dtype=np.uint8)
        frame_len = 256
        frames = []
        seq = 0
        rem = num_bits
        while rem > 0:
            seq_bits = np.array([int(b) for b in f"{seq & 0xFFFF:016b}"], dtype=np.uint8)
            payload_len = max(8, min(frame_len - 32, rem - 32))
            payload = rng.integers(0, 2, size=payload_len, dtype=np.uint8)
            parity = np.array([np.sum(payload) % 2], dtype=np.uint8)
            f = np.concatenate([preamble, seq_bits, payload, parity])
            frames.append(f)
            seq += 1
            rem -= len(f)
        return np.concatenate(frames)[:num_bits]
    
    elif pattern_type == "fsk_dotting":
        # Alternating tones for clock synchronization (010101... or 00110011...)
        pattern_str = rng.choice(["01", "10", "0011", "1100", "0110"])
        pat_arr = np.array([int(b) for b in pattern_str], dtype=np.uint8)
        reps = (num_bits // len(pat_arr)) + 1
        return np.tile(pat_arr, reps)[:num_bits]

    elif pattern_type == "fsk_sync_burst":
        # DMR 24-bit frame sync pattern or Barker-13 followed by telemetry payload
        dmr_sync = np.array([0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1], dtype=np.uint8)
        payload = rng.integers(0, 2, size=max(0, num_bits - len(dmr_sync)), dtype=np.uint8)
        return np.concatenate([dmr_sync, payload])[:num_bits]

    elif pattern_type == "tone_dwell":
        # Sustained tone holding simulating unmodulated squelch or long run telemetry
        dwell_len = rng.integers(40, 120)
        dwell_val = rng.integers(0, 2)
        dwell_bits = np.full(dwell_len, dwell_val, dtype=np.uint8)
        rem_bits = rng.integers(0, 2, size=max(0, num_bits - dwell_len), dtype=np.uint8)
        return np.concatenate([dwell_bits, rem_bits])[:num_bits]

    else:
        return rng.integers(0, 2, size=num_bits, dtype=np.uint8)


# =============================================================================
# 2. Pulse Shaping Filter Synthesis (RRC, RC, Gaussian)
# =============================================================================

def gaussian_filter_freq(freq: np.ndarray, sps: int, bt: float = 0.5) -> np.ndarray:
    """Applies Gaussian pulse shaping to instantaneous frequency deviation for GFSK."""
    span = 2
    L = span * sps
    t = np.arange(-L, L + 1) / sps
    sigma = np.sqrt(np.log(2.0)) / (2.0 * np.pi * max(1e-4, bt))
    h = np.exp(- (t ** 2) / (2.0 * (sigma ** 2)))
    h = h / np.sum(h)
    out = np.convolve(freq, h, mode='full')
    start = L
    return out[start : start + len(freq)]


def rc_filter_freq(freq: np.ndarray, sps: int, beta: float = 0.35) -> np.ndarray:
    """Applies Raised Cosine pulse shaping to instantaneous frequency deviation."""
    span = 2
    L = span * sps
    t = np.arange(-L, L + 1) / sps
    denom = 1.0 - (2.0 * beta * t) ** 2
    denom[np.abs(denom) < 1e-8] = 1e-8
    h = np.sinc(t) * np.cos(np.pi * beta * t) / denom
    h = h / np.sum(h)
    out = np.convolve(freq, h, mode='full')
    start = L
    return out[start : start + len(freq)]


def rrc_filter(sps: int, beta: float, span_symbols: int = 8) -> np.ndarray:
    """Synthesizes a Root-Raised-Cosine (RRC) pulse-shaping filter impulse response."""
    N = span_symbols * sps
    t = np.arange(-N / 2, N / 2 + 1) / sps
    h = np.zeros(len(t), dtype=np.float64)

    for i, ti in enumerate(t):
        if abs(ti) < 1e-8:
            h[i] = 1.0 + beta * (4.0 / np.pi - 1.0)
        elif abs(abs(ti) - 1.0 / (4.0 * beta)) < 1e-8:
            h[i] = (beta / np.sqrt(2.0)) * (
                (1.0 + 2.0 / np.pi) * np.sin(np.pi / (4.0 * beta))
                + (1.0 - 2.0 / np.pi) * np.cos(np.pi / (4.0 * beta))
            )
        else:
            num = np.sin(np.pi * ti * (1.0 - beta)) + 4.0 * beta * ti * np.cos(np.pi * ti * (1.0 + beta))
            den = np.pi * ti * (1.0 - (4.0 * beta * ti) ** 2)
            h[i] = num / den

    # Unit energy normalization
    energy = np.sum(h ** 2)
    if energy > 0:
        h = h / np.sqrt(energy)
    return h.astype(np.float32)


# =============================================================================
# 3. Waveform Synthesis: FSK (2-FSK, 4-FSK), PSK, and Dense QAM
# =============================================================================

def synthesize_fsk_clean(
    bits: np.ndarray,
    M: int,
    baud: float,
    sample_rate: float,
    tone_spacing_ratio: float,
    continuous_phase: bool,
    frequency_shift_ratio: float,
    drift_rate_hz_per_sec: float,
    initial_phase: float,
    pulse_shape: str = "rect",
    gaussian_bt: float = 0.5,
    rc_beta: float = 0.35,
    asymmetric_perturb: float = 0.0,
    tone_droop: float = 0.0,
    pa_chirp_rate: float = 0.0,
    hum_freq_hz: float = 0.0,
    hum_dev_hz: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Synthesizes realistic M-FSK / GFSK with:
    - Tone spacing variation & non-ideal asymmetric VCO perturbation
    - Pulse shaping: Rectangular, Gaussian (GFSK), or Raised Cosine
    - Continuous phase (CPFSK) vs discontinuous phase
    - Carrier offset across [-0.20 Fs, +0.20 Fs]
    - Dynamic linear drift, quadratic PA chirp, and mains FM hum
    - Channel bandpass edge tone droop / attenuation
    """
    if rng is None:
        rng = np.random.default_rng(42)

    k = int(math.log2(M))
    # Pad bits to multiple of k
    pad = (k - (len(bits) % k)) % k
    if pad > 0:
        bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
    
    # Group bits into symbols
    syms = np.zeros(len(bits) // k, dtype=np.int64)
    for i in range(k):
        syms = (syms << 1) | bits[i::k]
    
    num_symbols = len(syms)
    sps = int(round(sample_rate / baud))
    sps = max(2, sps)
    actual_sample_rate = sps * baud

    # Base tone frequencies relative to center
    delta_f = tone_spacing_ratio * baud
    if M == 2:
        tone_offsets = np.array([-0.5, 0.5], dtype=np.float64) * delta_f
    elif M == 4:
        # Standard Gray-coded 4-FSK tone grid [-1.5, -0.5, +0.5, +1.5]
        tone_offsets = np.array([-1.5, -0.5, 0.5, 1.5], dtype=np.float64) * delta_f
        gray_map = {0: 0, 1: 1, 3: 2, 2: 3}
        syms = np.array([gray_map.get(s, 0) for s in syms], dtype=np.int64)
    else:
        raise ValueError(f"Unsupported FSK M={M}")

    # Physical VCO non-linearity: asymmetric perturbation
    if asymmetric_perturb > 0.0:
        perturb = rng.uniform(-asymmetric_perturb, asymmetric_perturb, size=len(tone_offsets))
        tone_offsets = tone_offsets * (1.0 + perturb)

    # Carrier shift across [-0.20 Fs, +0.20 Fs]
    carrier_shift_hz = frequency_shift_ratio * actual_sample_rate
    symbol_freqs = tone_offsets[syms] + carrier_shift_hz

    # Raw frequency deviation per sample
    freq_per_sample = np.repeat(symbol_freqs, sps)
    total_samples = len(freq_per_sample)
    t = np.arange(total_samples) / actual_sample_rate

    # Apply frequency domain pulse shaping (GFSK / RC-FSK)
    if pulse_shape == "gaussian":
        inst_freq = gaussian_filter_freq(freq_per_sample, sps=sps, bt=gaussian_bt)
    elif pulse_shape == "rc":
        inst_freq = rc_filter_freq(freq_per_sample, sps=sps, beta=rc_beta)
    else:
        inst_freq = freq_per_sample.copy()

    # Linear frequency drift: drift_rate * t
    if drift_rate_hz_per_sec != 0.0:
        inst_freq = inst_freq + drift_rate_hz_per_sec * t

    # Power amplifier warm-up chirp: quadratic frequency drift
    if pa_chirp_rate != 0.0:
        inst_freq = inst_freq + pa_chirp_rate * (t ** 2)

    # Power supply mains FM hum
    if hum_freq_hz > 0.0 and hum_dev_hz > 0.0:
        inst_freq = inst_freq + hum_dev_hz * np.sin(2.0 * np.pi * hum_freq_hz * t)

    dt = 1.0 / actual_sample_rate
    if continuous_phase:
        # Phase integration: phi(t) = phi0 + 2*pi * integral(inst_freq)
        phase = initial_phase + 2.0 * np.pi * np.cumsum(inst_freq * dt)
    else:
        # Discontinuous phase resets at symbol boundaries
        m_idx = np.tile(np.arange(sps), num_symbols)
        phase = initial_phase + 2.0 * np.pi * inst_freq * (m_idx * dt)

    iq = np.exp(1j * phase).astype(np.complex64)

    # Tone droop: frequency-dependent attenuation across analog channel band
    if tone_droop > 0.0:
        max_dev = np.max(np.abs(tone_offsets)) + 1e-6
        amp_sag = 1.0 - tone_droop * np.clip(np.abs(inst_freq - carrier_shift_hz) / max_dev, 0.0, 1.0)
        iq = iq * amp_sag.astype(np.float32)

    # Unit power normalization
    power = np.mean(np.abs(iq) ** 2)
    if power > 0:
        iq = (iq / np.sqrt(power)).astype(np.complex64)

    meta = {
        "modulation": f"{M}-FSK",
        "M": M,
        "baud": baud,
        "sps": sps,
        "tone_spacing_ratio": tone_spacing_ratio,
        "tone_spacing_hz": float(delta_f),
        "carrier_shift_hz": float(carrier_shift_hz),
        "continuous_phase": continuous_phase,
        "drift_rate_hz_per_sec": drift_rate_hz_per_sec,
        "pulse_shape": pulse_shape,
        "gaussian_bt": float(gaussian_bt) if pulse_shape == "gaussian" else None,
        "tone_droop": float(tone_droop),
    }
    return iq, meta



def get_qam_grid(M: int) -> Tuple[np.ndarray, float, int]:
    """Generates standard 2D Gray-coded square QAM constellation symbols and norm factor."""
    k = int(math.log2(M))
    k_axis = k // 2
    side = int(math.sqrt(M))
    levels = np.arange(-side + 1, side, 2.0, dtype=np.float64)
    # Energy: sum of squares
    mean_energy = np.mean(np.tile(levels**2, side) + np.repeat(levels**2, side))
    norm_factor = 1.0 / np.sqrt(mean_energy)

    # 1D Gray coding
    def to_gray(val: int) -> int:
        return val ^ (val >> 1)
    gray_levels = {to_gray(i): levels[i] for i in range(side)}

    grid = np.zeros(M, dtype=np.complex64)
    for bi in range(side):
        for bq in range(side):
            sym_int = (bi << k_axis) | bq
            grid[sym_int] = complex(gray_levels[bi] * norm_factor, gray_levels[bq] * norm_factor)
    return grid, norm_factor, k


def synthesize_qam_psk_clean(
    bits: np.ndarray,
    mod_name: str,
    baud: float,
    sample_rate: float,
    rolloff: float,
    constellation_phase_rad: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Synthesizes RRC-pulse-shaped PSK or dense QAM baseband IQ with phase rotation."""
    norm_mod = normalize_modulation_name(mod_name)
    sps = int(round(sample_rate / baud))
    sps = max(2, sps)

    if norm_mod == "BPSK":
        syms = (2.0 * bits[:len(bits) - (len(bits) % 1)] - 1.0).astype(np.complex64)
    elif norm_mod == "QPSK":
        k = 2
        bits_adj = bits[:len(bits) - (len(bits) % k)]
        b0 = bits_adj[0::2]
        b1 = bits_adj[1::2]
        syms = ((2.0 * b0 - 1.0) + 1j * (2.0 * b1 - 1.0)) / np.sqrt(2.0)
    elif norm_mod == "8PSK":
        k = 3
        bits_adj = bits[:len(bits) - (len(bits) % k)]
        sym_int = (bits_adj[0::3] << 2) | (bits_adj[1::3] << 1) | bits_adj[2::3]
        # Gray encoded 8-PSK phase angles
        angles = np.array([0, 1, 3, 2, 7, 6, 4, 5], dtype=np.float64) * (2.0 * np.pi / 8.0)
        syms = np.exp(1j * angles[sym_int]).astype(np.complex64)
    elif norm_mod == "DQPSK":
        k = 2
        bits_adj = bits[:len(bits) - (len(bits) % k)]
        d_phase_map = {0: 0.0, 1: np.pi/2.0, 2: -np.pi/2.0, 3: np.pi}
        sym_int = (bits_adj[0::2] << 1) | bits_adj[1::2]
        d_phases = np.array([d_phase_map[int(s)] for s in sym_int])
        cum_phase = np.cumsum(d_phases)
        syms = np.exp(1j * cum_phase).astype(np.complex64)
    elif norm_mod == "MSK":
        # Continuous phase frequency shift h=0.5
        b = 2.0 * bits - 1.0
        phase = np.cumsum(b * (np.pi / (2.0 * sps)))
        iq = np.exp(1j * phase).astype(np.complex64)
        return iq, {"modulation": "MSK", "baud": baud, "sps": sps}
    elif norm_mod in ("16QAM", "64QAM", "256QAM"):
        M = int(norm_mod.replace("QAM", ""))
        grid, _, k = get_qam_grid(M)
        bits_adj = bits[:len(bits) - (len(bits) % k)]
        sym_int = np.zeros(len(bits_adj) // k, dtype=np.int64)
        for i in range(k):
            sym_int = (sym_int << 1) | bits_adj[i::k]
        syms = grid[sym_int]
    else:
        raise ValueError(f"Unsupported modulation: {norm_mod}")

    # Apply constellation phase rotation
    if constellation_phase_rad != 0.0:
        syms = syms * np.exp(1j * constellation_phase_rad)

    # Upsample by SPS
    upsampled = np.zeros(len(syms) * sps, dtype=np.complex64)
    upsampled[::sps] = syms

    # RRC pulse shaping
    h_rrc = rrc_filter(sps=sps, beta=rolloff, span_symbols=8)
    iq = signal.convolve(upsampled, h_rrc, mode="same").astype(np.complex64)

    # Unit power normalization
    power = np.mean(np.abs(iq) ** 2)
    if power > 0:
        iq = (iq / np.sqrt(power)).astype(np.complex64)

    meta = {
        "modulation": norm_mod,
        "baud": baud,
        "sps": sps,
        "rolloff": rolloff,
        "constellation_phase_rad": constellation_phase_rad,
    }
    return iq, meta


# =============================================================================
# 4. Realistic Channel Impairment Pipeline
# =============================================================================

def apply_comprehensive_impairments(
    iq: np.ndarray,
    snr_db: float,
    cfo_hz: float,
    sample_rate: float,
    gain_imbalance: float,
    phase_imbalance_deg: float,
    soft_clipping_sat: Optional[float],
    multipath_taps: Optional[List[Tuple[int, complex]]],
    dc_offset: complex,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Applies realistic RF channel physics: multipath, IQ imbalance, clipping, CFO, and AWGN."""
    impaired = iq.copy()

    # 1. Multipath FIR channel
    if multipath_taps and len(multipath_taps) > 1:
        max_del = max(t[0] for t in multipath_taps)
        cir = np.zeros(max_del + 1, dtype=np.complex64)
        for delay, amp in multipath_taps:
            cir[delay] += amp
        # Normalize channel energy
        cir = cir / np.sqrt(np.sum(np.abs(cir) ** 2))
        impaired = signal.convolve(impaired, cir, mode="same").astype(np.complex64)

    # 2. IQ Imbalance (Gain & Phase mismatch in quadrature mixer)
    if gain_imbalance != 0.0 or phase_imbalance_deg != 0.0:
        i_comp = np.real(impaired) * (1.0 + gain_imbalance / 2.0)
        q_raw = np.imag(impaired) * (1.0 - gain_imbalance / 2.0)
        phi_rad = np.radians(phase_imbalance_deg)
        q_comp = q_raw * np.cos(phi_rad) + i_comp * np.sin(phi_rad)
        impaired = (i_comp + 1j * q_comp).astype(np.complex64)

    # 3. Soft Clipping / SSPA Power Amplifier Non-Linearity (Rapp model)
    if soft_clipping_sat is not None and soft_clipping_sat > 0:
        v_sat = soft_clipping_sat
        p = 2.0
        mag = np.abs(impaired)
        gain_scale = 1.0 / (1.0 + (mag / v_sat) ** (2.0 * p)) ** (1.0 / (2.0 * p))
        impaired = impaired * gain_scale

    # 4. DC Offset
    if abs(dc_offset) > 0:
        impaired = impaired + dc_offset

    # 5. Carrier Frequency Offset (CFO)
    if cfo_hz != 0.0:
        t = np.arange(len(impaired)) / sample_rate
        impaired = impaired * np.exp(1j * 2.0 * np.pi * cfo_hz * t)

    # 6. AWGN (Noise addition based on realized SNR)
    sig_power = float(np.mean(np.abs(impaired) ** 2))
    if sig_power > 0:
        snr_linear = 10.0 ** (snr_db / 10.0)
        noise_power = sig_power / snr_linear
        noise_std = np.sqrt(noise_power / 2.0)
        noise = (rng.normal(0, noise_std, len(impaired)) + 1j * rng.normal(0, noise_std, len(impaired))).astype(np.complex64)
        impaired = impaired + noise

    meta = {
        "snr_db": float(snr_db),
        "cfo_hz": float(cfo_hz),
        "gain_imbalance": float(gain_imbalance),
        "phase_imbalance_deg": float(phase_imbalance_deg),
        "soft_clipping_sat": float(soft_clipping_sat) if soft_clipping_sat else None,
        "has_multipath": bool(multipath_taps and len(multipath_taps) > 1),
        "dc_offset_mag": float(abs(dc_offset)),
    }
    return impaired.astype(np.complex64), meta


# =============================================================================
# 5. Master Dataset Generation & Anti-Leakage Partitioning
# =============================================================================

def generate_v2_signal_capture(
    source_id: str,
    modulation: str,
    difficulty: str,
    seed: int,
    sample_rate: float = 192000.0,
    target_samples: int = 16384,
    is_ood: bool = False,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Synthesizes a single high-diversity signal burst with full parameter tracking."""
    rng = np.random.default_rng(seed)
    norm_mod = normalize_modulation_name(modulation)

    # 1. Parameter sampling based on difficulty & OOD flag
    if is_ood:
        # Held-out parameter regimes
        baud_candidates = [3200.0, 6000.0, 12800.0, 25600.0]
        baud = float(rng.choice(baud_candidates))
        snr_db = float(rng.choice([-8.0, -3.0, 3.0, 8.0, 22.0]))
        fsk_tone_spacing = float(rng.choice([0.35, 0.85, 1.35, 2.25]))
        rolloff = float(rng.choice([0.15, 0.45]))
    else:
        # Standard in-distribution curriculum
        baud_candidates = [1200.0, 2400.0, 4800.0, 9600.0, 19200.0, 38400.0]
        baud = float(rng.choice(baud_candidates))
        
        # Difficulty tiers
        if difficulty == "clean":
            snr_db = float(rng.uniform(18.0, 30.0))
        elif difficulty == "easy":
            snr_db = float(rng.uniform(12.0, 20.0))
        elif difficulty == "medium":
            snr_db = float(rng.uniform(3.0, 12.0))
        elif difficulty == "hard":
            snr_db = float(rng.uniform(-4.0, 4.0))
        else:  # extreme
            snr_db = float(rng.uniform(-10.0, -2.0))

        # FSK tone spacing: candidates + continuous random
        if rng.uniform() < 0.6:
            fsk_tone_spacing = float(rng.choice([0.25, 0.5, 0.75, 1.0, 1.5, 2.0]))
        else:
            fsk_tone_spacing = float(rng.uniform(0.3, 2.2))

        # RRC roll-off candidates + continuous
        if rng.uniform() < 0.7:
            rolloff = float(rng.choice([0.10, 0.20, 0.25, 0.35, 0.50]))
        else:
            rolloff = float(rng.uniform(0.12, 0.48))

    # Pattern variation: FSK receives targeted synchronization and dwell patterns
    if "FSK" in norm_mod:
        pattern = rng.choice(["random", "fsk_dotting", "fsk_sync_burst", "tone_dwell", "biased", "prbs", "framed"])
    else:
        pattern = rng.choice(["random", "biased", "alternating", "prbs", "framed"])
    num_bits = max(4096, int(target_samples * 2))
    bits = generate_varied_bits(num_bits, pattern_type=str(pattern), rng=rng)

    # 2. Modulation Synthesis
    if "FSK" in norm_mod:
        M = 4 if "4" in norm_mod else 2
        continuous = bool(rng.uniform() < 0.85)  # 85% continuous phase (CPFSK / GFSK)
        # Pulse shaping distribution: 50% Gaussian (GFSK), 25% Raised Cosine, 25% Rectangular
        p_shape = rng.choice(["gaussian", "rc", "rect"], p=[0.50, 0.25, 0.25])
        bt_val = float(rng.choice([0.25, 0.30, 0.40, 0.50, 0.70]))
        beta_val = float(rng.choice([0.20, 0.25, 0.35, 0.50]))

        # Physical VCO imperfections
        asym = float(rng.uniform(0.01, 0.06)) if difficulty in ("medium", "hard", "extreme") else 0.0
        droop = float(rng.uniform(0.04, 0.20)) if difficulty in ("medium", "hard", "extreme") else 0.0
        
        # CFO shifted across [-0.15 Fs, +0.15 Fs]
        freq_shift_ratio = float(rng.uniform(-0.15, 0.15)) if difficulty in ("medium", "hard", "extreme") else 0.0
        drift_rate = float(rng.uniform(-300.0, 300.0)) if difficulty in ("hard", "extreme") else 0.0
        pa_chirp = float(rng.uniform(-40.0, 40.0)) if difficulty in ("hard", "extreme") and rng.uniform() < 0.35 else 0.0
        
        # Mains power supply ripple hum (50 Hz or 60 Hz or 100 Hz FM)
        hum_f = float(rng.choice([50.0, 60.0, 100.0])) if difficulty in ("hard", "extreme") and rng.uniform() < 0.35 else 0.0
        hum_d = float(rng.uniform(15.0, 45.0)) if hum_f > 0.0 else 0.0
        phase0 = float(rng.uniform(0.0, 2.0 * np.pi))

        clean_iq, mod_meta = synthesize_fsk_clean(
            bits=bits,
            M=M,
            baud=baud,
            sample_rate=sample_rate,
            tone_spacing_ratio=fsk_tone_spacing,
            continuous_phase=continuous,
            frequency_shift_ratio=freq_shift_ratio,
            drift_rate_hz_per_sec=drift_rate,
            initial_phase=phase0,
            pulse_shape=str(p_shape),
            gaussian_bt=bt_val,
            rc_beta=beta_val,
            asymmetric_perturb=asym,
            tone_droop=droop,
            pa_chirp_rate=pa_chirp,
            hum_freq_hz=hum_f,
            hum_dev_hz=hum_d,
            rng=rng,
        )
    else:
        phase0 = float(rng.uniform(0.0, 2.0 * np.pi))
        clean_iq, mod_meta = synthesize_qam_psk_clean(
            bits=bits,
            mod_name=norm_mod,
            baud=baud,
            sample_rate=sample_rate,
            rolloff=rolloff,
            constellation_phase_rad=phase0,
            rng=rng,
        )

    # 3. Channel Impairments Selection
    if difficulty == "clean":
        cfo = float(rng.uniform(-50.0, 50.0))
        gain_imb = 0.0
        phase_imb = 0.0
        clipping = None
        multipath = None
        dc = 0j
    elif difficulty == "easy":
        cfo = float(rng.uniform(-250.0, 250.0))
        gain_imb = float(rng.uniform(-0.03, 0.03))
        phase_imb = float(rng.uniform(-2.0, 2.0))
        clipping = None
        multipath = [(0, 1.0 + 0j)]
        dc = complex(rng.uniform(-0.02, 0.02), rng.uniform(-0.02, 0.02))
    elif difficulty == "medium":
        cfo = float(rng.uniform(-1500.0, 1500.0))
        gain_imb = float(rng.uniform(-0.08, 0.08))
        phase_imb = float(rng.uniform(-5.0, 5.0))
        clipping = float(rng.uniform(1.2, 1.8)) if rng.uniform() < 0.4 else None
        multipath = [(0, 1.0 + 0j), (int(rng.integers(2, 6)), rng.uniform(0.15, 0.35) * np.exp(1j * rng.uniform(0, 2*np.pi)))]
        dc = complex(rng.uniform(-0.05, 0.05), rng.uniform(-0.05, 0.05))
    else:  # hard & extreme
        cfo = float(rng.uniform(-5000.0, 5000.0))
        gain_imb = float(rng.uniform(-0.15, 0.15))
        phase_imb = float(rng.uniform(-10.0, 10.0))
        clipping = float(rng.uniform(0.85, 1.3))
        multipath = [
            (0, 1.0 + 0j),
            (int(rng.integers(2, 5)), rng.uniform(0.3, 0.5) * np.exp(1j * rng.uniform(0, 2*np.pi))),
            (int(rng.integers(6, 12)), rng.uniform(0.15, 0.3) * np.exp(1j * rng.uniform(0, 2*np.pi))),
        ]
        dc = complex(rng.uniform(-0.08, 0.08), rng.uniform(-0.08, 0.08))

    impaired_iq, chan_meta = apply_comprehensive_impairments(
        iq=clean_iq[:target_samples],
        snr_db=snr_db,
        cfo_hz=cfo,
        sample_rate=sample_rate,
        gain_imbalance=gain_imb,
        phase_imbalance_deg=phase_imb,
        soft_clipping_sat=clipping,
        multipath_taps=multipath,
        dc_offset=dc,
        rng=rng,
    )

    # Pad or truncate to exact target_samples
    if len(impaired_iq) < target_samples:
        pad_len = target_samples - len(impaired_iq)
        impaired_iq = np.pad(impaired_iq, (0, pad_len), mode="constant")
    else:
        impaired_iq = impaired_iq[:target_samples]

    metadata = {
        "source_id": source_id,
        "seed": seed,
        "modulation": norm_mod,
        "difficulty": difficulty,
        "pattern_type": pattern,
        "sample_rate": sample_rate,
        "is_ood": is_ood,
        **mod_meta,
        **chan_meta,
    }
    return impaired_iq, metadata


def build_v2_dataset(
    output_dir: Path,
    samples_per_class: int = 300,
    ood_samples_per_class: int = 60,
    fsk_samples_per_class: Optional[int] = None,
    fsk_ood_per_class: Optional[int] = None,
    master_seed: int = 42,
) -> Dict[str, Any]:
    """
    Builds the complete ASTRA_MODULATION_SYNTHETIC_V2 dataset with strict anti-leakage source grouping.
    Supports enriched signal volume and diversity patterns for FSK classes.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir = output_dir / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    captures_dir = output_dir / "captures"
    captures_dir.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, Any]] = []
    classes = TRAINED_MODULATION_CLASSES_V2
    difficulty_levels = ["clean", "easy", "medium", "hard", "extreme"]
    diff_weights = [0.20, 0.30, 0.30, 0.15, 0.05]

    logger.info("Generating ASTRA_MODULATION_SYNTHETIC_V2 with %d classes...", len(classes))
    rng = np.random.default_rng(master_seed)
    source_counter = 0

    # 1. Generate In-Distribution Signals (Train 70%, Val 15%, Test 15% partitioned at SOURCE level)
    for cls in classes:
        is_fsk = "FSK" in cls
        n_cls_samples = fsk_samples_per_class if (is_fsk and fsk_samples_per_class) else samples_per_class
        n_tr = int(n_cls_samples * 0.70)
        n_va = int(n_cls_samples * 0.15)
        splits = ["train"] * n_tr + ["validation"] * n_va + ["test"] * (n_cls_samples - n_tr - n_va)
        rng.shuffle(splits)

        for i in range(n_cls_samples):
            source_counter += 1
            src_id = f"src_{source_counter:06d}"
            split_assign = splits[i]
            diff = rng.choice(difficulty_levels, p=diff_weights)
            seed = int(master_seed + source_counter * 10007) & 0x7FFFFFFF

            iq_data, meta = generate_v2_signal_capture(
                source_id=src_id,
                modulation=cls,
                difficulty=str(diff),
                seed=seed,
                is_ood=False,
            )

            # Save raw float32 IQ capture
            cap_filename = f"{src_id}_{cls.replace('-', '').lower()}.iq"
            cap_path = captures_dir / cap_filename
            iq_data.tofile(cap_path)

            meta["split"] = split_assign
            meta["file_path"] = str(cap_path)
            meta["sample_count"] = len(iq_data)
            records.append(meta)

    # 2. Generate OOD Evaluation Signals
    for cls in classes:
        is_fsk = "FSK" in cls
        n_cls_ood = fsk_ood_per_class if (is_fsk and fsk_ood_per_class) else ood_samples_per_class
        for i in range(n_cls_ood):
            source_counter += 1
            src_id = f"src_ood_{source_counter:06d}"
            diff = rng.choice(["medium", "hard", "extreme"])
            seed = int(master_seed + source_counter * 10007 + 99999) & 0x7FFFFFFF

            iq_data, meta = generate_v2_signal_capture(
                source_id=src_id,
                modulation=cls,
                difficulty=str(diff),
                seed=seed,
                is_ood=True,
            )

            cap_filename = f"{src_id}_{cls.replace('-', '').lower()}.iq"
            cap_path = captures_dir / cap_filename
            iq_data.tofile(cap_path)

            meta["split"] = "ood_test"
            meta["file_path"] = str(cap_path)
            meta["sample_count"] = len(iq_data)
            records.append(meta)

    # Save manifests
    import pandas as pd
    df = pd.DataFrame(records)
    all_manifest = manifest_dir / "all.csv"
    df.to_csv(all_manifest, index=False)
    df[df["split"] == "train"].to_csv(manifest_dir / "train.csv", index=False)
    df[df["split"] == "validation"].to_csv(manifest_dir / "val.csv", index=False)
    df[df["split"] == "test"].to_csv(manifest_dir / "test.csv", index=False)
    df[df["split"] == "ood_test"].to_csv(manifest_dir / "ood.csv", index=False)

    summary = {
        "dataset_name": "ASTRA_MODULATION_SYNTHETIC_V2",
        "total_source_captures": len(df),
        "classes": classes,
        "split_counts": df["split"].value_counts().to_dict(),
        "class_counts": df["modulation"].value_counts().to_dict(),
        "manifest_path": str(all_manifest),
    }

    with open(output_dir / "dataset_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("Dataset V2 Generation Complete: %d sources generated.", len(df))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASTRA V2 Synthetic Dataset Builder")
    parser.add_argument("--output", type=str, default="datasets/ASTRA_MODULATION_SYNTHETIC_V2")
    parser.add_argument("--samples-per-class", type=int, default=300)
    parser.add_argument("--ood-per-class", type=int, default=60)
    parser.add_argument("--fsk-samples-per-class", type=int, default=600)
    parser.add_argument("--fsk-ood-per-class", type=int, default=120)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    out_p = Path(args.output)
    build_v2_dataset(
        output_dir=out_p,
        samples_per_class=args.samples_per_class,
        ood_samples_per_class=args.ood_per_class,
        fsk_samples_per_class=args.fsk_samples_per_class,
        fsk_ood_per_class=args.fsk_ood_per_class,
        master_seed=args.seed,
    )

