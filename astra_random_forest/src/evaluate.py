"""
ASTRA Random Forest Support Model Comprehensive Evaluation Suite.
Evaluates accuracy vs SNR, Modulation, and generates Fusion-vs-RF comparison report.
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from .dataset_builder import build_dataset_from_signals
from .family_classifier import BroadFamilyClassifier
from .inference import RandomForestSupportEngine
from .quality_classifier import SignalQualityClassifier
from .utils import generate_impaired_signal

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("astra_random_forest.evaluate")


def run_benchmark_evaluation(
    engine: RandomForestSupportEngine,
    num_test_signals: int = 120,
    seed: int = 999,
) -> Dict[str, Any]:
    """
    Evaluates Random Forest Family and Quality models across a test grid of signals.
    """
    np.random.seed(seed)
    mods = ["2-FSK", "4-FSK", "BPSK", "QPSK", "8PSK", "16-QAM", "64-QAM", "NOISE"]
    snrs = [-5, 0, 5, 10, 15, 20, 25]
    rates = [2400, 4800, 9600, 19200]
    sample_rates = [96000, 192000]

    signals = []
    for i in range(num_test_signals):
        mod = mods[i % len(mods)]
        snr = float(np.random.choice(snrs))
        rate = float(np.random.choice(rates))
        fs = float(np.random.choice(sample_rates))
        while fs / rate < 2.5:
            fs *= 2

        cfo = float(np.random.uniform(-1500, 1500))
        clip = float(np.random.choice([0.0, 0.0, 0.08, 0.15]))
        multipath = bool(np.random.choice([False, True]))

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
        meta["signal_id"] = f"rf_test_{i:05d}"
        signals.append((iq, meta))

    dataset = build_dataset_from_signals(signals)
    X = dataset["X"]
    y_fam_true = dataset["y_family"]
    y_qual_true = dataset["y_quality"]
    meta_rows = dataset["metadata"]

    # 1. Family Model Evaluation
    pred_res = engine.predict_batch(X)
    fam_preds = [r["family"] for r in pred_res]

    fam_acc = float(accuracy_score(y_fam_true, fam_preds))
    fam_bal_acc = float(balanced_accuracy_score(y_fam_true, fam_preds))
    fam_macro_f1 = float(f1_score(y_fam_true, fam_preds, average="macro", zero_division=0))
    classes = sorted(list(set(y_fam_true).union(set(fam_preds))))
    cm = confusion_matrix(y_fam_true, fam_preds, labels=classes).tolist()

    # Per-class metrics
    per_class = {}
    for cls in classes:
        cls_mask_true = (y_fam_true == cls)
        cls_mask_pred = np.array(fam_preds) == cls
        per_class[cls] = {
            "precision": float(precision_score(cls_mask_true, cls_mask_pred, zero_division=0)),
            "recall": float(recall_score(cls_mask_true, cls_mask_pred, zero_division=0)),
            "f1": float(f1_score(cls_mask_true, cls_mask_pred, zero_division=0)),
            "support": int(np.sum(cls_mask_true)),
        }

    # 2. Quality Model Evaluation
    qual_preds_arr = np.column_stack([
        [1 if r["quality"]["low_snr_probability"] >= 0.50 else 0 for r in pred_res],
        [1 if r["quality"]["clipping_probability"] >= 0.50 else 0 for r in pred_res],
        [1 if r["quality"]["multipath_probability"] >= 0.50 else 0 for r in pred_res],
        [1 if r["quality"]["cfo_affected_probability"] >= 0.50 else 0 for r in pred_res],
    ])

    qual_metrics = {}
    for idx, lbl in enumerate(dataset["quality_label_names"]):
        lbl_acc = float(accuracy_score(y_qual_true[:, idx], qual_preds_arr[:, idx]))
        lbl_f1 = float(f1_score(y_qual_true[:, idx], qual_preds_arr[:, idx], zero_division=0))
        qual_metrics[lbl] = {"accuracy": lbl_acc, "f1": lbl_f1}

    # 3. Accuracy vs SNR Breakdown
    snr_breakdown = {}
    for snr_val in sorted(list(set(m["snr_db"] for m in meta_rows))):
        sub_indices = [idx for idx, m in enumerate(meta_rows) if m["snr_db"] == snr_val]
        sub_true = y_fam_true[sub_indices]
        sub_pred = [fam_preds[idx] for idx in sub_indices]
        snr_breakdown[snr_val] = {
            "count": len(sub_indices),
            "accuracy": float(accuracy_score(sub_true, sub_pred)),
            "macro_f1": float(f1_score(sub_true, sub_pred, average="macro", zero_division=0)),
        }

    # 4. Accuracy vs Underlying Modulation Breakdown
    mod_breakdown = {}
    for mod_val in sorted(list(set(m["mod_type"] for m in meta_rows))):
        sub_indices = [idx for idx, m in enumerate(meta_rows) if m["mod_type"] == mod_val]
        sub_true = y_fam_true[sub_indices]
        sub_pred = [fam_preds[idx] for idx in sub_indices]
        mod_breakdown[mod_val] = {
            "count": len(sub_indices),
            "accuracy": float(accuracy_score(sub_true, sub_pred)),
            "predicted_families": {f: sub_pred.count(f) for f in set(sub_pred)},
        }

    return {
        "num_test_signals": len(signals),
        "family": {
            "accuracy": fam_acc,
            "balanced_accuracy": fam_bal_acc,
            "macro_f1": fam_macro_f1,
            "classes": classes,
            "confusion_matrix": cm,
            "per_class": per_class,
        },
        "quality": qual_metrics,
        "snr_breakdown": snr_breakdown,
        "mod_breakdown": mod_breakdown,
    }
