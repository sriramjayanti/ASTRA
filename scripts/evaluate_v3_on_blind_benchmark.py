"""
evaluate_v3_on_blind_benchmark.py
Evaluates the retrained and hardened Stage 3 Modulation Intelligence V3 on the untouched ASTRA_FINAL_TEST_SET_V2_BLIND.
Reports:
- Top-1 and Top-3 accuracy
- Macro-F1 and per-class F1
- Breakdown by SNR regimes (High, Medium, Low, Negative)
- Breakdown by CFO regimes (Low, Medium, High)
- Breakdown by Multipath conditions (Single-path vs Multi-tap)
- Confusion matrix
"""

from __future__ import annotations

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Any
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from astra_modulation_v2.class_schema import MODULATION_CLASSES_V2, CLASS_TO_IDX_V2, NUM_CLASSES_V2
from astra_modulation_v2.fusion import CalibratedFusionEngineV2

print("=" * 80)
print("EVALUATING STAGE 3 MODULATION INTELLIGENCE V3 ON BLIND BENCHMARK")
print("=" * 80)

blind_manifest = ROOT / "datasets" / "ASTRA_FINAL_TEST_SET_V2_BLIND" / "blind_benchmark_manifest.json"
with open(blind_manifest, "r", encoding="utf-8") as f:
    blind_data = json.load(f)

captures = blind_data["captures"]
print(f"Loaded {len(captures)} total blind benchmark captures from {blind_manifest.name}")

# Initialize Fusion Engine with calibrated weights (favoring ResNet-1D high-fidelity temporal modeling)
engine = CalibratedFusionEngineV2(weight_1d=0.75, weight_2d=0.25, rf_beta=0.20, device="cuda")

y_true = []
y_pred_top1 = []
top3_hits = []

# Breakdown buckets
snr_buckets = {"high": [], "med": [], "low": [], "neg": []}
cfo_buckets = {"low": [], "med": [], "high": []}
multipath_buckets = {"single": [], "multipath": []}

per_class_stats = {mod: {"total": 0, "top1_hits": 0, "top3_hits": 0} for mod in MODULATION_CLASSES_V2}

t0 = time.time()
for idx, cap in enumerate(captures):
    raw = np.fromfile(cap["iq_path"], dtype=np.float32)
    iq = raw[0::2] + 1j * raw[1::2]

    true_mod = cap["true_modulation"]
    sr = float(cap["sample_rate"])
    snr = float(cap.get("snr_db", 15.0))
    cfo = float(cap.get("cfo_hz", 0.0))

    res = engine.classify(iq, sample_rate=sr)
    top1_mod = res["predicted_modulation"]
    top3_mods = [c["modulation"] for c in res["top_k_candidates"][:3]]

    is_top1 = (top1_mod == true_mod)
    is_top3 = (true_mod in top3_mods)

    y_true.append(true_mod)
    y_pred_top1.append(top1_mod)
    top3_hits.append(is_top3)

    per_class_stats[true_mod]["total"] += 1
    if is_top1:
        per_class_stats[true_mod]["top1_hits"] += 1
    if is_top3:
        per_class_stats[true_mod]["top3_hits"] += 1

    # SNR bucket
    if snr >= 15.0:
        snr_buckets["high"].append((is_top1, is_top3))
    elif snr >= 5.0:
        snr_buckets["med"].append((is_top1, is_top3))
    elif snr >= 0.0:
        snr_buckets["low"].append((is_top1, is_top3))
    else:
        snr_buckets["neg"].append((is_top1, is_top3))

    # CFO bucket
    abs_cfo = abs(cfo)
    if abs_cfo <= 500.0:
        cfo_buckets["low"].append((is_top1, is_top3))
    elif abs_cfo <= 2000.0:
        cfo_buckets["med"].append((is_top1, is_top3))
    else:
        cfo_buckets["high"].append((is_top1, is_top3))

    # Multipath bucket
    if cap.get("multipath_taps", 1) > 1 or cap.get("multipath", False):
        multipath_buckets["multipath"].append((is_top1, is_top3))
    else:
        multipath_buckets["single"].append((is_top1, is_top3))

    if (idx + 1) % 200 == 0 or (idx + 1) == len(captures):
        elapsed = time.time() - t0
        curr_top1 = 100.0 * np.mean([p == t for p, t in zip(y_pred_top1, y_true)])
        curr_top3 = 100.0 * np.mean(top3_hits)
        print(f"Evaluated [{idx+1:4d}/{len(captures)}] | Top-1: {curr_top1:.1f}% | Top-3: {curr_top3:.1f}% ({elapsed:.1f}s)", flush=True)

# Compute metrics
overall_top1 = 100.0 * np.mean([p == t for p, t in zip(y_pred_top1, y_true)])
overall_top3 = 100.0 * np.mean(top3_hits)
macro_f1 = 100.0 * f1_score(y_true, y_pred_top1, average="macro", zero_division=0)

cm = confusion_matrix(y_true, y_pred_top1, labels=MODULATION_CLASSES_V2)

print("\n" + "=" * 80)
print("FINAL BENCHMARK EVALUATION RESULTS (MODULATION V3):")
print("=" * 80)
print(f"Total Captures Evaluated: {len(captures)}")
print(f"Overall Top-1 Accuracy:   {overall_top1:.2f}%")
print(f"Overall Top-3 Accuracy:   {overall_top3:.2f}% (Candidate Retention)")
print(f"Macro-F1 Score:           {macro_f1:.2f}%")

print("\n--- PER-CLASS PERFORMANCE ---")
class_metrics = {}
for mod in MODULATION_CLASSES_V2:
    st = per_class_stats[mod]
    tot = st["total"]
    t1_acc = (100.0 * st["top1_hits"] / tot) if tot > 0 else 0.0
    t3_acc = (100.0 * st["top3_hits"] / tot) if tot > 0 else 0.0
    class_metrics[mod] = {
        "total": tot,
        "top1_accuracy": round(t1_acc, 1),
        "top3_accuracy": round(t3_acc, 1),
    }
    print(f"  {mod:10s} | Total: {tot:3d} | Top-1: {t1_acc:5.1f}% | Top-3: {t3_acc:5.1f}%")

print("\n--- PERFORMANCE BY SNR REGIME ---")
snr_metrics = {}
for k, name in [("high", "High (>= 15 dB)"), ("med", "Medium (5 to 15 dB)"), ("low", "Low (0 to 5 dB)"), ("neg", "Negative (< 0 dB)")]:
    items = snr_buckets[k]
    t1 = 100.0 * np.mean([x[0] for x in items]) if items else 0.0
    t3 = 100.0 * np.mean([x[1] for x in items]) if items else 0.0
    snr_metrics[k] = {"count": len(items), "top1": round(t1, 1), "top3": round(t3, 1)}
    print(f"  {name:22s} | Count: {len(items):3d} | Top-1: {t1:5.1f}% | Top-3: {t3:5.1f}%")

print("\n--- PERFORMANCE BY CFO REGIME ---")
cfo_metrics = {}
for k, name in [("low", "Low (<= 500 Hz)"), ("med", "Med (500 to 2000 Hz)"), ("high", "High (> 2000 Hz)")]:
    items = cfo_buckets[k]
    t1 = 100.0 * np.mean([x[0] for x in items]) if items else 0.0
    t3 = 100.0 * np.mean([x[1] for x in items]) if items else 0.0
    cfo_metrics[k] = {"count": len(items), "top1": round(t1, 1), "top3": round(t3, 1)}
    print(f"  {name:22s} | Count: {len(items):3d} | Top-1: {t1:5.1f}% | Top-3: {t3:5.1f}%")

# Save detailed JSON report
report_data = {
    "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "total_captures": len(captures),
    "overall_top1_accuracy": round(overall_top1, 2),
    "overall_top3_accuracy": round(overall_top3, 2),
    "macro_f1": round(macro_f1, 2),
    "per_class": class_metrics,
    "snr_breakdown": snr_metrics,
    "cfo_breakdown": cfo_metrics,
    "confusion_matrix": cm.tolist(),
    "classes": MODULATION_CLASSES_V2,
}

out_report = ROOT / "checkpoints" / "v3_evaluation_report.json"
with open(out_report, "w", encoding="utf-8") as f:
    json.dump(report_data, f, indent=2)
print(f"\n[DONE] Saved evaluation report to: {out_report}")
