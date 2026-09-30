"""
ASTRA Random Forest Dataset Builder.
Extracts tabular DSP features and builds signal-level grouped training datasets.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np

from .feature_extractor_adapter import extract_dsp_feature_dict
from .feature_schema import FEATURE_COLUMNS, validate_feature_vector
from .label_mapping import extract_quality_labels, map_modulation_to_family
from .utils import generate_impaired_signal

logger = logging.getLogger("astra_random_forest.dataset_builder")


def build_dataset_from_signals(
    signals_data: Sequence[Tuple[np.ndarray, Dict[str, Any]]],
    quality_thresholds: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Transforms a collection of (iq_array, metadata) tuples into training tabular matrices.
    """
    q_thresh = quality_thresholds or {}
    low_snr_max = float(q_thresh.get("low_snr_db_max", 3.0))
    clip_min = float(q_thresh.get("clipping_ratio_min", 0.05))
    cfo_min = float(q_thresh.get("cfo_affected_hz_min", 500.0))

    X_rows: List[List[float]] = []
    y_families: List[str] = []
    y_qualities: List[List[int]] = []
    groups: List[str] = []
    metadata_rows: List[Dict[str, Any]] = []

    quality_label_names = ["low_snr", "clipped", "multipath", "cfo_affected"]

    for idx, (iq, meta) in enumerate(signals_data):
        sig_id = str(meta.get("signal_id", f"sig_{idx:05d}"))
        fs = float(meta.get("sample_rate_hz", 192000.0))
        mod_type = str(meta.get("mod_type", "UNKNOWN"))

        # 1. Extract DSP features
        feat_dict = extract_dsp_feature_dict(iq, sample_rate_hz=fs)
        row = validate_feature_vector(feat_dict)
        X_rows.append(row)

        # 2. Map Family Label
        family_label = map_modulation_to_family(mod_type)
        y_families.append(family_label)

        # 3. Extract Multi-Label Quality Labels
        q_dict = extract_quality_labels(
            meta,
            low_snr_threshold_db=low_snr_max,
            clipping_threshold=clip_min,
            cfo_affected_threshold_hz=cfo_min,
        )
        y_qualities.append([q_dict[lbl] for lbl in quality_label_names])

        groups.append(sig_id)
        metadata_rows.append({
            "signal_id": sig_id,
            "mod_type": mod_type,
            "family_label": family_label,
            "snr_db": meta.get("snr_db", 0.0),
            "cfo_hz": meta.get("cfo_hz", 0.0),
            "quality_labels": q_dict,
        })

    X = np.asarray(X_rows, dtype=np.float32) if X_rows else np.zeros((0, len(FEATURE_COLUMNS)), dtype=np.float32)
    y_fam = np.asarray(y_families, dtype=object)
    y_qual = np.asarray(y_qualities, dtype=np.int32) if y_qualities else np.zeros((0, len(quality_label_names)), dtype=np.int32)
    groups_arr = np.asarray(groups, dtype=object)

    return {
        "X": X,
        "y_family": y_fam,
        "y_quality": y_qual,
        "quality_label_names": quality_label_names,
        "groups": groups_arr,
        "metadata": metadata_rows,
        "feature_names": FEATURE_COLUMNS,
    }


def generate_synthetic_rf_pool(
    num_signals: int = 140,
    seed: int = 42,
) -> List[Tuple[np.ndarray, Dict[str, Any]]]:
    """
    Generates a rich, balanced training pool of synthetic signals across FSK, PSK, QAM, and Noise.
    """
    np.random.seed(seed)
    mods = ["2-FSK", "4-FSK", "BPSK", "QPSK", "8PSK", "16-QAM", "64-QAM", "NOISE"]
    snrs = [-5, 0, 5, 10, 15, 20, 25]
    rates = [2400, 4800, 9600, 19200, 38400]
    sample_rates = [96000, 192000, 250000]

    signals = []
    for i in range(num_signals):
        mod = mods[i % len(mods)]
        snr = float(np.random.choice(snrs))
        rate = float(np.random.choice(rates))
        fs = float(np.random.choice(sample_rates))
        while fs / rate < 2.5:
            fs *= 2

        cfo = float(np.random.uniform(-1500, 1500))
        clip = float(np.random.choice([0.0, 0.0, 0.0, 0.08, 0.15]))
        multipath = bool(np.random.choice([False, False, True]))

        iq, meta = generate_impaired_signal(
            mod_type=mod,
            sample_rate_hz=fs,
            symbol_rate_hz=rate,
            num_symbols=1024,
            snr_db=snr,
            cfo_hz=cfo,
            clipping_ratio=clip,
            multipath=multipath,
            seed=seed + i,
        )
        meta["signal_id"] = f"rf_sig_{i:05d}"
        signals.append((iq, meta))

    return signals
