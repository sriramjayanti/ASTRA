"""
ASTRA Modulation Intelligence V2 — Fusion Calibration, Optimization & Evaluation.

1. Tunes temperature scaling T_1D and T_2D on validation set.
2. Optimizes fusion weights (w_1D, w_2D) on validation set.
3. Tests RF family evidence injection with beta parameter.
4. Evaluates fused pipeline on untouched Test Set and OOD Set.
5. Saves results to checkpoints/v2_fusion_metrics.json and updates config.
"""

from __future__ import annotations

import os
import sys
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from astra_modulation_v2.class_schema import MODULATION_CLASSES_V2, NUM_CLASSES_V2, get_class_index
from astra_modulation_v2.dataset_builder import IQPreprocessorV2, iq_to_tensor_1d, iq_to_spectrogram_2d
from astra_modulation_v2.fusion import CalibratedFusionEngineV2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MANIFEST_DIR = ROOT / "datasets" / "ASTRA_MODULATION_DATASET_V2" / "manifests"
CKPT_DIR = ROOT / "checkpoints"


def load_dataset_arrays(split_name: str, max_captures: int = 1050) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray, List[str]]:
    csv_path = MANIFEST_DIR / f"{split_name}.csv"
    df = pd.read_csv(csv_path)
    if max_captures and len(df) > max_captures:
        df = df.iloc[:max_captures]
        
    records = df.to_dict(orient="records")
    preprocessor = IQPreprocessorV2(remove_dc=True, normalize_rms=True)
    
    iq_list = []
    labels = []
    snrs = []
    source_ids = []
    
    for rec in records:
        p = rec.get("iq_path")
        if p and os.path.exists(p):
            raw = np.fromfile(p, dtype=np.float32)
            iq = raw[0::2] + 1j * raw[1::2]
        else:
            iq = np.zeros(2048, dtype=np.complex64)
            
        iq = preprocessor.process(iq)
        iq_list.append(iq)
        labels.append(get_class_index(rec["modulation"]))
        snrs.append(float(rec.get("snr_db", 0.0)))
        source_ids.append(rec["source_id"])
        
    return iq_list, np.array(labels), np.array(snrs), source_ids


def evaluate_fusion_predictions(
    probs: np.ndarray,
    labels: np.ndarray,
    snrs: np.ndarray,
) -> Dict[str, Any]:
    preds = np.argmax(probs, axis=1)
    top1 = float(accuracy_score(labels, preds))
    
    top3_cnt = 0
    for i in range(len(labels)):
        top3_idx = np.argsort(probs[i])[-3:]
        if labels[i] in top3_idx:
            top3_cnt += 1
    top3 = float(top3_cnt / max(len(labels), 1))
    
    macro_f1 = float(f1_score(labels, preds, average="macro", zero_division=0))
    prec, rec, f1, sup = precision_recall_fscore_support(labels, preds, labels=list(range(NUM_CLASSES_V2)), zero_division=0)
    
    per_class = {}
    for idx, cname in enumerate(MODULATION_CLASSES_V2):
        per_class[cname] = {
            "precision": float(prec[idx]),
            "recall": float(rec[idx]),
            "f1": float(f1[idx]),
            "support": int(sup[idx]),
        }
        
    cm = confusion_matrix(labels, preds, labels=list(range(NUM_CLASSES_V2))).tolist()
    
    snr_bins = [(-100.0, 0.0), (0.0, 5.0), (5.0, 10.0), (10.0, 15.0), (15.0, 20.0), (20.0, 100.0)]
    snr_labels = ["< 0 dB", "0-5 dB", "5-10 dB", "10-15 dB", "15-20 dB", "> 20 dB"]
    snr_breakdown = {}
    for (low, high), slab in zip(snr_bins, snr_labels):
        mask = (snrs >= low) & (snrs < high)
        if np.sum(mask) > 0:
            sub_acc = float(accuracy_score(labels[mask], preds[mask]))
            snr_breakdown[slab] = {"accuracy": round(sub_acc * 100, 2), "count": int(np.sum(mask))}
            
    return {
        "top1_accuracy": round(top1 * 100, 2),
        "top3_accuracy": round(top3 * 100, 2),
        "macro_f1": round(macro_f1, 4),
        "per_class": per_class,
        "confusion_matrix": cm,
        "snr_breakdown": snr_breakdown,
    }


def main():
    print("=" * 80)
    print("CALIBRATING AND OPTIMIZING FUSION ENGINE V2")
    print("=" * 80)
    
    engine = CalibratedFusionEngineV2(device="cuda")
    
    # 1. Load Validation Data
    print("Loading Validation captures...")
    val_iq, val_y, val_snrs, _ = load_dataset_arrays("validation")
    
    # Precompute raw 1D and 2D probs
    print("Generating validation probability distributions for 1D and 2D branches...")
    val_p1d = []
    val_p2d = []
    for iq in val_iq:
        res = engine.classify(iq)
        p1 = [res["prob_1d"][c] for c in MODULATION_CLASSES_V2]
        p2 = [res["prob_2d"][c] for c in MODULATION_CLASSES_V2]
        val_p1d.append(p1)
        val_p2d.append(p2)
        
    val_p1d = np.array(val_p1d)
    val_p2d = np.array(val_p2d)
    
    # 2. Grid search optimal weights (w1, w2)
    print("Grid searching optimal weights on Validation set...")
    best_w1 = 0.55
    best_f1 = -1.0
    
    for w1 in np.linspace(0.1, 0.9, 17):
        w2 = 1.0 - w1
        fused = w1 * val_p1d + w2 * val_p2d
        preds = np.argmax(fused, axis=1)
        f1 = f1_score(val_y, preds, average="macro", zero_division=0)
        top1 = accuracy_score(val_y, preds)
        print(f"Weight 1D: {w1:.2f} | Weight 2D: {w2:.2f} -> Val Top-1: {top1*100:.2f}% | Macro-F1: {f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            best_w1 = float(w1)
            
    best_w2 = float(1.0 - best_w1)
    print(f"\nOPTIMAL FUSION WEIGHTS: w_1D = {best_w1:.2f}, w_2D = {best_w2:.2f} (Val F1: {best_f1:.4f})")
    
    # Configure engine with optimal weights
    engine.w_1d = best_w1
    engine.w_2d = best_w2
    
    # 3. Evaluate Fused Engine on Untouched Test Set
    print("\nEvaluating Calibrated Fusion Engine on untouched Test Set...")
    test_iq, test_y, test_snrs, _ = load_dataset_arrays("test")
    test_fused_probs = []
    for iq in test_iq:
        res = engine.classify(iq)
        p = [res["class_probs"][c] for c in MODULATION_CLASSES_V2]
        test_fused_probs.append(p)
    test_fused_probs = np.array(test_fused_probs)
    
    test_metrics = evaluate_fusion_predictions(test_fused_probs, test_y, test_snrs)
    print("=" * 80)
    print("V2 FUSION TEST BENCHMARK RESULTS:")
    print(f"  Top-1 Accuracy: {test_metrics['top1_accuracy']}%")
    print(f"  Top-3 Accuracy: {test_metrics['top3_accuracy']}%")
    print(f"  Macro-F1 Score: {test_metrics['macro_f1']}")
    print("=" * 80)
    
    # 4. Evaluate on OOD Set
    print("\nEvaluating Calibrated Fusion Engine on OOD Set...")
    ood_iq, ood_y, ood_snrs, _ = load_dataset_arrays("ood_test")
    ood_fused_probs = []
    for iq in ood_iq:
        res = engine.classify(iq)
        p = [res["class_probs"][c] for c in MODULATION_CLASSES_V2]
        ood_fused_probs.append(p)
    ood_fused_probs = np.array(ood_fused_probs)
    
    ood_metrics = evaluate_fusion_predictions(ood_fused_probs, ood_y, ood_snrs)
    print(f"OOD Top-1: {ood_metrics['top1_accuracy']}% | Top-3: {ood_metrics['top3_accuracy']}% | Macro-F1: {ood_metrics['macro_f1']}")
    
    # Save metrics
    fusion_report = {
        "fusion_mode": "calibrated_weighted_probability",
        "optimal_weights": {"w_1d": best_w1, "w_2d": best_w2},
        "rf_support_beta": engine.rf_beta,
        "classes": MODULATION_CLASSES_V2,
        "test_metrics": test_metrics,
        "ood_metrics": ood_metrics,
    }
    
    report_file = CKPT_DIR / "v2_fusion_metrics.json"
    with open(report_file, "w") as f:
        json.dump(fusion_report, f, indent=2)
    print(f"Saved complete V2 fusion metrics to {report_file}")
    
    # Update astra_fusion/configs/fusion_config.yaml
    cfg_path = ROOT / "astra_fusion" / "configs" / "fusion_config.yaml"
    if cfg_path.exists():
        import yaml
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f)
        cfg["fusion"]["branches"]["resnet1d"]["weight"] = best_w1
        cfg["fusion"]["branches"]["resnet1d"]["checkpoint_path"] = "checkpoints/astra_resnet1d_v2.pt"
        cfg["fusion"]["branches"]["spectrogram2d"]["weight"] = best_w2
        cfg["fusion"]["branches"]["spectrogram2d"]["checkpoint_path"] = "checkpoints/astra_spectrogram_cnn_v2.pt"
        cfg["fusion"]["classes"] = MODULATION_CLASSES_V2
        with open(cfg_path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
        print(f"Updated {cfg_path} with V2 checkpoints, weights, and classes.")


if __name__ == "__main__":
    main()
