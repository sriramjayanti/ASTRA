"""
ASTRA Symbol-Rate Dataset Builder.
Generates candidate feature matrices with signal-level group IDs to prevent data leakage.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np

from .autocorrelation import extract_autocorr_evidence
from .bandwidth import extract_bandwidth_evidence
from .candidate_features import FEATURE_COLUMNS, build_candidate_feature_matrix
from .candidate_generator import generate_candidates
from .cyclostationary import extract_cyclostationary_evidence
from .instantaneous_frequency import extract_instantaneous_frequency_evidence
from .models import DSPRateEvidence, SymbolRateCandidate
from .preprocessing import compute_cfo_estimate, compute_snr_estimate, preprocess_iq
from .spectral import extract_spectral_evidence
from .utils import generate_synthetic_signal

logger = logging.getLogger("astra_symbol_rate.dataset_builder")


def extract_dsp_evidence(
    iq: np.ndarray,
    sample_rate_hz: float,
    config: Optional[Dict[str, Any]] = None,
) -> DSPRateEvidence:
    """
    Runs all independent DSP evidence estimators on a preprocessed IQ burst.
    """
    proc_iq, meta = preprocess_iq(iq, sample_rate_hz)
    snr_db = compute_snr_estimate(proc_iq)
    cfo_hz = compute_cfo_estimate(proc_iq, sample_rate_hz)

    # 1. Autocorrelation & Power Autocorrelation
    ac_peaks, pwr_peaks = extract_autocorr_evidence(proc_iq, sample_rate_hz)

    # 2. Instantaneous Frequency
    if_peaks = extract_instantaneous_frequency_evidence(proc_iq, sample_rate_hz)

    # 3. Spectral Evidence
    spectral_meta = extract_spectral_evidence(proc_iq, sample_rate_hz)

    # 4. Bandwidth Evidence
    bw_meta = extract_bandwidth_evidence(proc_iq, sample_rate_hz)

    # 5. Cyclostationary Evidence
    cyclo_peaks = extract_cyclostationary_evidence(proc_iq, sample_rate_hz)

    return DSPRateEvidence(
        autocorr_peaks=ac_peaks,
        power_autocorr_peaks=pwr_peaks,
        if_peaks=if_peaks,
        spectral_peak_spacings=spectral_meta.get("spectral_peak_spacings", []),
        bandwidth_candidates=bw_meta.get("bandwidth_candidates", []),
        cyclostationary_peaks=cyclo_peaks,
        occupied_bandwidth_hz=float(bw_meta.get("occupied_bandwidth_hz", 0.0)),
        estimated_snr_db=float(snr_db),
        cfo_estimate_hz=float(cfo_hz),
        spectral_flatness=float(spectral_meta.get("spectral_flatness", 0.0)),
        spectral_centroid_hz=float(spectral_meta.get("spectral_centroid_hz", 0.0)),
        clipping_ratio=float(meta.get("clipping_ratio", 0.0)),
        sample_rate_hz=float(sample_rate_hz),
    )


def build_candidate_dataset_from_signals(
    signals_data: Sequence[Tuple[np.ndarray, Dict[str, Any]]],
    tolerance_percent: float = 2.0,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Transforms a collection of (iq_array, metadata) tuples into a tabular dataset.
    Preserves strict signal-level group IDs for leakage-free validation splits.
    """
    feature_rows: List[List[float]] = []
    labels: List[int] = []
    groups: List[str] = []
    meta_rows: List[Dict[str, Any]] = []

    total_signals = len(signals_data)
    signals_with_recall = 0

    for idx, (iq, meta) in enumerate(signals_data):
        sig_id = str(meta.get("signal_id", f"sig_{idx:05d}"))
        true_rate = float(meta.get("true_symbol_rate_hz", meta.get("symbol_rate_hz", 0.0)))
        fs = float(meta.get("sample_rate_hz", 192000.0))
        mod_context = meta.get("modulation_context", None)

        # 1. Extract DSP evidence
        evidence = extract_dsp_evidence(iq, fs, config)

        # 2. Generate candidate rates
        candidates = generate_candidates(evidence, fs, config)

        # Measure recall for this signal
        found_recall = False
        for c in candidates:
            rel_err = abs(c.rate_hz - true_rate) / max(1.0, true_rate)
            if rel_err <= (tolerance_percent / 100.0):
                found_recall = True
                break
        if found_recall:
            signals_with_recall += 1

        # 3. Build candidate feature matrix
        feat_matrix, dicts = build_candidate_feature_matrix(
            candidates=candidates,
            evidence=evidence,
            sample_rate_hz=fs,
            modulation_context=mod_context,
        )

        # 4. Create labels and metadata rows
        for c_idx, cand in enumerate(candidates):
            rel_err = abs(cand.rate_hz - true_rate) / max(1.0, true_rate)
            is_correct = 1 if rel_err <= (tolerance_percent / 100.0) else 0

            feature_rows.append([cand.features[col] for col in FEATURE_COLUMNS])
            labels.append(is_correct)
            groups.append(sig_id)

            meta_rows.append({
                "signal_id": sig_id,
                "candidate_rate_hz": cand.rate_hz,
                "true_rate_hz": true_rate,
                "relative_error": rel_err,
                "is_correct": is_correct,
                "mod_type": meta.get("mod_type", "UNKNOWN"),
                "snr_db": meta.get("snr_db", 0.0),
                "cfo_hz": meta.get("cfo_hz", 0.0),
                "sources": cand.sources,
            })

    candidate_recall = (signals_with_recall / max(1, total_signals)) if total_signals > 0 else 0.0

    X = np.asarray(feature_rows, dtype=np.float32) if feature_rows else np.zeros((0, len(FEATURE_COLUMNS)), dtype=np.float32)
    y = np.asarray(labels, dtype=np.int32) if labels else np.zeros((0,), dtype=np.int32)
    groups_arr = np.asarray(groups, dtype=object)

    return {
        "X": X,
        "y": y,
        "groups": groups_arr,
        "feature_names": FEATURE_COLUMNS,
        "meta_rows": meta_rows,
        "total_signals": total_signals,
        "candidate_recall": candidate_recall,
    }


def generate_synthetic_training_pool(
    num_signals: int = 120,
    seed: int = 42,
) -> List[Tuple[np.ndarray, Dict[str, Any]]]:
    """
    Generates a diverse synthetic dataset across various modulations, SNRs, CFOs, and symbol rates.
    """
    np.random.seed(seed)
    mods = ["2-FSK", "4-FSK", "BPSK", "QPSK", "8PSK", "16-QAM", "64-QAM"]
    rates = [1200, 2400, 4800, 9600, 19200, 38400, 57600, 7350, 14200]
    sample_rates = [96000, 192000, 250000, 384000]
    snrs = [-5, 0, 5, 10, 15, 20]
    rolloffs = [0.20, 0.25, 0.35, 0.50]

    signals = []
    for i in range(num_signals):
        mod = mods[i % len(mods)]
        rate = rates[(i // len(mods)) % len(rates)]
        fs = sample_rates[(i // (len(mods) * len(rates))) % len(sample_rates)]
        # Ensure SPS >= 2.0
        while fs / rate < 2.5:
            fs *= 2

        snr = float(np.random.choice(snrs))
        cfo = float(np.random.uniform(-1500, 1500))
        rolloff = float(np.random.choice(rolloffs))

        iq, meta = generate_synthetic_signal(
            mod_type=mod,
            symbol_rate_hz=rate,
            sample_rate_hz=fs,
            num_symbols=1024,
            snr_db=snr,
            cfo_hz=cfo,
            rolloff=rolloff,
            seed=seed + i,
        )
        meta["signal_id"] = f"synth_sig_{i:05d}"
        signals.append((iq, meta))

    return signals
