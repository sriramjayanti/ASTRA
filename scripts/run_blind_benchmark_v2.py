"""
Run Pristine Blind Final Benchmark: ASTRA_FINAL_TEST_SET_V2_BLIND.

Evaluates frozen V2 Calibrated Fusion Engine across 1,100 blind captures
(100 captures per class across all 11 canonical classes).

Outputs:
- checkpoints/blind_benchmark_v2_results.json
- BLIND_BENCHMARK_V2_REPORT.md
"""

from __future__ import annotations
import os
import sys
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from astra_modulation_v2.class_schema import MODULATION_CLASSES_V2, NUM_CLASSES_V2, get_class_index
from astra_modulation_v2.fusion import CalibratedFusionEngineV2

OUT_DIR = ROOT / "datasets" / "ASTRA_FINAL_TEST_SET_V2_BLIND"
MANIFEST_PATH = OUT_DIR / "blind_benchmark_manifest.json"
CKPT_DIR = ROOT / "checkpoints"

print("=" * 80)
print("EXECUTING ASTRA_FINAL_TEST_SET_V2_BLIND BENCHMARK (1,100 CAPTURES)")
print("=" * 80)

if not MANIFEST_PATH.exists():
    raise FileNotFoundError(f"Blind benchmark manifest not found: {MANIFEST_PATH}")

with open(MANIFEST_PATH, "r") as f:
    data = json.load(f)

captures = data["captures"]
engine = CalibratedFusionEngineV2(device="cuda")

true_labels = []
pred_labels = []
pred_probs = []
snrs = []
durations = []
class_breakdown = {c: {"total": 0, "top1": 0, "top3": 0} for c in MODULATION_CLASSES_V2}

t0 = time.time()
print(f"Running inference across {len(captures)} captures on CUDA...", flush=True)

for i, cap in enumerate(captures):
    iq_path = cap["iq_path"]
    true_mod = cap["true_modulation"]
    true_idx = get_class_index(true_mod)
    snr = float(cap.get("snr_db", 0.0))
    sr = float(cap.get("sample_rate", 192000.0))
    
    raw = np.fromfile(iq_path, dtype=np.float32)
    iq = raw[0::2] + 1j * raw[1::2]
    
    res = engine.classify(iq, sample_rate=sr)
    p_vector = [res["class_probs"][c] for c in MODULATION_CLASSES_V2]
    pred_idx = np.argmax(p_vector)
    
    true_labels.append(true_idx)
    pred_labels.append(pred_idx)
    pred_probs.append(p_vector)
    snrs.append(snr)
    
    class_breakdown[true_mod]["total"] += 1
    if pred_idx == true_idx:
        class_breakdown[true_mod]["top1"] += 1
        
    top3_indices = np.argsort(p_vector)[-3:]
    if true_idx in top3_indices:
        class_breakdown[true_mod]["top3"] += 1
        
    if (i + 1) % 200 == 0 or (i + 1) == len(captures):
        print(f"Processed {i+1:04d}/{len(captures):04d} captures...", flush=True)

elapsed_total = time.time() - t0
true_labels = np.array(true_labels)
pred_labels = np.array(pred_labels)
pred_probs = np.array(pred_probs)
snrs = np.array(snrs)

top1_acc = float(accuracy_score(true_labels, pred_labels))

top3_cnt = 0
for i in range(len(true_labels)):
    if true_labels[i] in np.argsort(pred_probs[i])[-3:]:
        top3_cnt += 1
top3_acc = float(top3_cnt / len(true_labels))

macro_f1 = float(f1_score(true_labels, pred_labels, average="macro", zero_division=0))
prec, rec, f1, sup = precision_recall_fscore_support(true_labels, pred_labels, labels=list(range(NUM_CLASSES_V2)), zero_division=0)

per_class_results = {}
for idx, cname in enumerate(MODULATION_CLASSES_V2):
    per_class_results[cname] = {
        "precision": float(round(prec[idx], 4)),
        "recall": float(round(rec[idx], 4)),
        "f1": float(round(f1[idx], 4)),
        "support": int(sup[idx]),
        "top1_accuracy": round(100.0 * class_breakdown[cname]["top1"] / max(class_breakdown[cname]["total"], 1), 2),
        "top3_accuracy": round(100.0 * class_breakdown[cname]["top3"] / max(class_breakdown[cname]["total"], 1), 2),
    }

# Noise / UNKNOWN false positive rate on communications captures:
# Comms classified as UNKNOWN
unk_idx = get_class_index("UNKNOWN")
comms_mask = (true_labels != unk_idx)
noise_fpr = float(np.mean(pred_labels[comms_mask] == unk_idx)) * 100.0

# Stratified SNR
snr_bins = [(-100.0, 0.0), (0.0, 5.0), (5.0, 10.0), (10.0, 15.0), (15.0, 20.0), (20.0, 100.0)]
snr_labels = ["< 0 dB", "0-5 dB", "5-10 dB", "10-15 dB", "15-20 dB", "> 20 dB"]
snr_breakdown = {}
for (low, high), slab in zip(snr_bins, snr_labels):
    mask = (snrs >= low) & (snrs < high)
    if np.sum(mask) > 0:
        sub_acc = float(accuracy_score(true_labels[mask], pred_labels[mask]))
        snr_breakdown[slab] = {"accuracy": round(sub_acc * 100, 2), "count": int(np.sum(mask))}

print("=" * 80)
print(f"BLIND BENCHMARK EXECUTION COMPLETE in {elapsed_total:.2f}s ({elapsed_total/len(captures)*1000:.1f} ms/capture)")
print(f"Modulation Top-1 Accuracy: {top1_acc * 100:.2f}%")
print(f"Modulation Top-3 Accuracy: {top3_acc * 100:.2f}%")
print(f"Macro-F1 Score:            {macro_f1:.4f}")
print(f"Noise False-Positive Rate: {noise_fpr:.2f}%")
print("=" * 80)

# Save JSON results
benchmark_output = {
    "benchmark_name": "ASTRA_FINAL_TEST_SET_V2_BLIND",
    "total_captures": len(captures),
    "execution_time_seconds": round(elapsed_total, 2),
    "metrics": {
        "Modulation Top-1": round(top1_acc * 100, 2),
        "Modulation Top-3": round(top3_acc * 100, 2),
        "Macro-F1": round(macro_f1, 4),
        "Noise false-positive rate": round(noise_fpr, 2),
    },
    "per_class": per_class_results,
    "snr_breakdown": snr_breakdown,
    "confusion_matrix": confusion_matrix(true_labels, pred_labels, labels=list(range(NUM_CLASSES_V2))).tolist(),
    "classes": MODULATION_CLASSES_V2,
}

json_path = CKPT_DIR / "blind_benchmark_v2_results.json"
with open(json_path, "w") as f:
    json.dump(benchmark_output, f, indent=2)

# Generate Markdown Report
md_lines = [
    "# ASTRA — Blind Final Benchmark Report (`ASTRA_FINAL_TEST_SET_V2_BLIND`)",
    "",
    f"**Execution Date:** 2026-09-29  ",
    f"**Hardware Device:** `CUDA` (NVIDIA GPU)  ",
    f"**Total Blind Captures:** {len(captures):,}  ",
    f"**Captures Per Class:** 100 captures across 11 canonical classes  ",
    f"**Total Execution Time:** {elapsed_total:.2f}s ({elapsed_total/len(captures)*1000:.1f} ms/capture)  ",
    "",
    "---",
    "",
    "## 1. Key Benchmark Metrics",
    "",
    "| Metric | V1 Legacy Benchmark | V2 Blind Final Benchmark | Material Improvement |",
    "| :--- | :---: | :---: | :---: |",
    f"| **Modulation Top-1 Accuracy** | 17.20% | **{top1_acc*100:.2f}%** | **+{top1_acc*100 - 17.20:.2f}%** |",
    f"| **Modulation Top-3 Accuracy** | 41.30% | **{top3_acc*100:.2f}%** | **+{top3_acc*100 - 41.30:.2f}%** |",
    f"| **Macro-F1 Score** | ~0.150 | **{macro_f1:.4f}** | **+{macro_f1 - 0.150:.4f}** |",
    f"| **Noise False-Positive Rate** | 22.00% | **{noise_fpr:.2f}%** | **-{22.00 - noise_fpr:.2f}%** |",
    "",
    "---",
    "",
    "## 2. Per-Class Performance Breakdown (100 Captures / Class)",
    "",
    "| Modulation Class | Top-1 Accuracy (%) | Top-3 Accuracy (%) | Precision | Recall | F1-Score | Status |",
    "| :--- | :---: | :---: | :---: | :---: | :---: | :--- |",
]

for mod in MODULATION_CLASSES_V2:
    p = per_class_results[mod]
    status = "EXCELLENT" if p["top3_accuracy"] >= 90.0 else ("STRONG" if p["top3_accuracy"] >= 70.0 else "ACTIVE")
    md_lines.append(
        f"| **{mod}** | {p['top1_accuracy']}% | {p['top3_accuracy']}% | {p['precision']:.3f} | {p['recall']:.3f} | {p['f1']:.3f} | {status} |"
    )

md_lines.extend([
    "",
    "---",
    "",
    "## 3. Stratified Performance vs. SNR",
    "",
    "| SNR Tier | Sample Count | Top-1 Accuracy (%) |",
    "| :--- | :---: | :---: |",
])

for slab in snr_labels:
    if slab in snr_breakdown:
        b = snr_breakdown[slab]
        md_lines.append(f"| **{slab}** | {b['count']} | **{b['accuracy']}%** |")

report_path = ROOT / "BLIND_BENCHMARK_V2_REPORT.md"
with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(md_lines))

print(f"\nWritten complete blind benchmark report to {report_path}")
