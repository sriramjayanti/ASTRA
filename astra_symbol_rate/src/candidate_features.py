"""
ASTRA Candidate Feature Extraction and Schema Definition.
Extracts normalized numerical features per candidate hypothesis for XGBoost ranking.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np

from .models import DSPRateEvidence, SymbolRateCandidate

FEATURE_SCHEMA_VERSION = "1.0.0"

# Fixed, deterministic feature column order
FEATURE_COLUMNS: List[str] = [
    "candidate_rate_hz",
    "candidate_normalized",
    "samples_per_symbol",
    "autocorr_score",
    "power_autocorr_score",
    "cyclostationary_score",
    "bandwidth_consistency",
    "spectral_peak_score",
    "instant_freq_score",
    "harmonic_consistency",
    "support_count",
    "is_standard_grid",
    "snr_db_estimate",
    "clipping_ratio",
    "spectral_flatness",
    "cfo_magnitude_hz",
    "mod_fsk_family_prob",
    "mod_psk_family_prob",
    "mod_qam_family_prob",
    "mod_top1_confidence",
]


def extract_candidate_feature_dict(
    candidate: SymbolRateCandidate,
    evidence: DSPRateEvidence,
    sample_rate_hz: float,
    modulation_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """
    Constructs a complete feature dictionary for a single candidate rate.
    """
    rs = candidate.rate_hz
    fs = max(1.0, sample_rate_hz)
    sps = fs / max(1.0, rs)

    # 1. Autocorrelation evidence
    autocorr_score = 0.0
    for p in evidence.autocorr_peaks:
        if abs(p["rate_hz"] - rs) / max(1.0, rs) <= 0.03:
            autocorr_score = max(autocorr_score, float(p.get("score", 0.0)))

    # 2. Power Autocorrelation evidence
    pwr_autocorr_score = 0.0
    for p in evidence.power_autocorr_peaks:
        if abs(p["rate_hz"] - rs) / max(1.0, rs) <= 0.03:
            pwr_autocorr_score = max(pwr_autocorr_score, float(p.get("score", 0.0)))

    # 3. Cyclostationary evidence
    cyclo_score = 0.0
    for p in evidence.cyclostationary_peaks:
        if abs(p["rate_hz"] - rs) / max(1.0, rs) <= 0.03:
            cyclo_score = max(cyclo_score, float(p.get("score", 0.0)))

    # 4. Bandwidth Consistency
    bw_consistency = 0.0
    obw = max(1.0, evidence.occupied_bandwidth_hz)
    # Check if OBW is roughly in [0.8*Rs, 2.0*Rs]
    ratio = obw / max(1.0, rs)
    if 0.9 <= ratio <= 2.2:
        bw_consistency = float(np.clip(1.0 - abs(ratio - 1.25) / 1.0, 0.0, 1.0))

    # 5. Spectral Peak Spacing
    spectral_score = 0.0
    for p in evidence.spectral_peak_spacings:
        if abs(p["rate_hz"] - rs) / max(1.0, rs) <= 0.03:
            spectral_score = max(spectral_score, float(p.get("score", 0.0)))

    # 6. Instantaneous Frequency
    if_score = 0.0
    for p in evidence.if_peaks:
        if abs(p["rate_hz"] - rs) / max(1.0, rs) <= 0.03:
            if_score = max(if_score, float(p.get("score", 0.0)))

    # 7. Harmonic Consistency
    harmonic_consistency = 1.0 if candidate.harmonic_ratio is not None else 0.0

    # 8. Multi-Source Support Count
    support_count = float(len(candidate.supported_by))

    # 9. Standard grid proximity
    is_standard = 1.0 if (candidate.snap_distance_percent is not None and candidate.snap_distance_percent <= 3.0) else 0.0

    # 10. Modulation Context (Optional)
    fsk_prob = 0.0
    psk_prob = 0.0
    qam_prob = 0.0
    top1_conf = 0.0

    if modulation_context:
        probs = modulation_context.get("probabilities", {})
        if probs:
            fsk_prob = float(probs.get("2-FSK", 0.0) + probs.get("4-FSK", 0.0) + probs.get("2fsk", 0.0) + probs.get("4fsk", 0.0))
            psk_prob = float(probs.get("BPSK", 0.0) + probs.get("QPSK", 0.0) + probs.get("8PSK", 0.0) + probs.get("bpsk", 0.0) + probs.get("qpsk", 0.0) + probs.get("8psk", 0.0))
            qam_prob = float(probs.get("16-QAM", 0.0) + probs.get("64-QAM", 0.0) + probs.get("16qam", 0.0) + probs.get("64qam", 0.0) + probs.get("256qam", 0.0))
        top1_conf = float(modulation_context.get("confidence", 0.0))

    feature_dict = {
        "candidate_rate_hz": float(rs),
        "candidate_normalized": float(rs / fs),
        "samples_per_symbol": float(sps),
        "autocorr_score": float(autocorr_score),
        "power_autocorr_score": float(pwr_autocorr_score),
        "cyclostationary_score": float(cyclo_score),
        "bandwidth_consistency": float(bw_consistency),
        "spectral_peak_score": float(spectral_score),
        "instant_freq_score": float(if_score),
        "harmonic_consistency": float(harmonic_consistency),
        "support_count": float(support_count),
        "is_standard_grid": float(is_standard),
        "snr_db_estimate": float(evidence.estimated_snr_db),
        "clipping_ratio": float(evidence.clipping_ratio),
        "spectral_flatness": float(evidence.spectral_flatness),
        "cfo_magnitude_hz": float(abs(evidence.cfo_estimate_hz)),
        "mod_fsk_family_prob": float(fsk_prob),
        "mod_psk_family_prob": float(psk_prob),
        "mod_qam_family_prob": float(qam_prob),
        "mod_top1_confidence": float(top1_conf),
    }

    # Ensure all values are finite float32
    for k, v in feature_dict.items():
        if not math.isfinite(v):
            feature_dict[k] = 0.0

    return feature_dict


def build_candidate_feature_matrix(
    candidates: Sequence[SymbolRateCandidate],
    evidence: DSPRateEvidence,
    sample_rate_hz: float,
    modulation_context: Optional[Dict[str, Any]] = None,
) -> Tuple[np.ndarray, List[Dict[str, float]]]:
    """
    Returns (feature_matrix [N_candidates, N_features], list_of_feature_dicts).
    """
    rows: List[List[float]] = []
    dicts: List[Dict[str, float]] = []

    for cand in candidates:
        feat_dict = extract_candidate_feature_dict(
            candidate=cand,
            evidence=evidence,
            sample_rate_hz=sample_rate_hz,
            modulation_context=modulation_context,
        )
        cand.features = feat_dict
        dicts.append(feat_dict)
        row = [feat_dict[col] for col in FEATURE_COLUMNS]
        rows.append(row)

    if not rows:
        return np.zeros((0, len(FEATURE_COLUMNS)), dtype=np.float32), []

    return np.asarray(rows, dtype=np.float32), dicts
