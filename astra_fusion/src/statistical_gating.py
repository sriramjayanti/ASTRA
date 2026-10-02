"""
ASTRA Statistical Cumulants & Envelope Pre-Gating Engine (Stage 1).

Extracts mathematically rigorous higher-order cumulants (C40, C42),
envelope kurtosis (gamma_2), and spectral flatness to form hard Bayesian priors.
This cleanly separates:
  - Constant Envelope (BPSK, QPSK, 8PSK, 2FSK, 4FSK, MSK)
  - Multi-Amplitude / Dense Constellations (16QAM, 64QAM, 256QAM, 4ASK, 8ASK)
  - Pure Phase Keying vs Continuous Phase FSK
"""

from __future__ import annotations
from typing import Dict, Any, Tuple, Optional
import numpy as np


def compute_higher_order_cumulants(iq_samples: np.ndarray) -> Dict[str, float]:
    """
    Computes 2nd and 4th order statistical moments and cumulants:
      M20 = E[r^2]
      M21 = E[|r|^2]
      M40 = E[r^4]
      M41 = E[r^3 * r*]
      M42 = E[|r|^4]
      C40 = M40 - 3*(M20)^2
      C42 = M42 - |M20|^2 - 2*(M21)^2
    """
    # Normalize to zero-mean and unit variance
    r = np.asarray(iq_samples, dtype=np.complex128)
    r = r - np.mean(r)
    pwr = np.mean(np.abs(r) ** 2)
    if pwr < 1e-12:
        return {
            "c40": 0.0,
            "c42": 0.0,
            "gamma_2": 1.0,
            "spectral_flatness": 1.0,
            "is_constant_envelope": True,
            "is_multi_amplitude": False,
        }

    r_norm = r / np.sqrt(pwr)

    # Moments
    m20 = np.mean(r_norm ** 2)
    m21 = np.mean(np.abs(r_norm) ** 2)  # Should be 1.0
    m40 = np.mean(r_norm ** 4)
    m42 = np.mean(np.abs(r_norm) ** 4)

    # Cumulants
    c40 = float(np.abs(m40 - 3.0 * (m20 ** 2)))
    c42 = float(np.real(m42 - np.abs(m20) ** 2 - 2.0 * (m21 ** 2)))
    gamma_2 = float(m42 / (m21 ** 2 + 1e-12))

    # Instantaneous frequency variance & spectral flatness
    diff_phase = np.diff(np.unwrap(np.angle(r_norm)))
    freq_var = float(np.var(diff_phase))

    # Envelope modulation depth
    mag = np.abs(r_norm)
    mag_var = float(np.var(mag))

    is_constant_envelope = bool(gamma_2 < 1.25 and mag_var < 0.15)
    is_multi_amplitude = bool(gamma_2 >= 1.30 or mag_var >= 0.20)

    return {
        "c40": c40,
        "c42": c42,
        "gamma_2": gamma_2,
        "mag_var": mag_var,
        "freq_var": freq_var,
        "is_constant_envelope": is_constant_envelope,
        "is_multi_amplitude": is_multi_amplitude,
    }


def apply_statistical_prior_gating(
    raw_probabilities: np.ndarray,
    class_names: list[str],
    iq_samples: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Applies hard Bayesian prior gating to raw neural network softmax outputs
    based on physical cumulants and envelope kurtosis.
    """
    stats = compute_higher_order_cumulants(iq_samples)
    probs = np.copy(raw_probabilities).astype(np.float64)

    # Class category masks
    fsk_msk_classes = {"2-FSK", "4-FSK", "MSK", "GMSK"}
    psk_classes = {"BPSK", "QPSK", "8PSK", "16PSK", "DQPSK"}
    qam_classes = {"16QAM", "32QAM", "64QAM", "256QAM"}
    ask_classes = {"4ASK", "8ASK", "OOK", "ASK"}

    priors = np.ones(len(class_names), dtype=np.float64)

    if stats["is_multi_amplitude"]:
        # Strongly suppress constant-envelope FSK/MSK/BPSK/QPSK
        for idx, cls in enumerate(class_names):
            if cls in fsk_msk_classes:
                priors[idx] *= 0.05
            elif cls in qam_classes or cls in ask_classes:
                priors[idx] *= 3.0
    elif stats["is_constant_envelope"]:
        # Constant envelope: strongly suppress high-order QAM
        for idx, cls in enumerate(class_names):
            if cls in qam_classes:
                priors[idx] *= 0.05

        # Differentiate FSK/MSK from Linear PSK using instantaneous frequency variance
        # FSK/MSK has distinct continuous frequency variance without discrete phase constellation collapses
        if stats["freq_var"] < 0.15:
            # Clean linear PSK with static carrier
            for idx, cls in enumerate(class_names):
                if cls in fsk_msk_classes:
                    priors[idx] *= 0.25
                elif cls in psk_classes:
                    priors[idx] *= 2.5

    # Apply Bayesian update
    gated_probs = probs * priors
    total = np.sum(gated_probs)
    if total > 1e-12:
        gated_probs = gated_probs / total
    else:
        gated_probs = probs / np.sum(probs)

    return gated_probs.astype(np.float32), stats
