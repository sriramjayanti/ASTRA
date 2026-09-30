import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath("."))

from astra_symbol_rate.src.inference import SymbolRateEstimator
from astra_symbol_rate.src.dataset_builder import extract_dsp_evidence
from astra_symbol_rate.src.candidate_generator import generate_candidates

manifest_path = "datasets/ASTRA_FINAL_TEST_SET_V2_BLIND/blind_benchmark_manifest.json"
with open(manifest_path, "r") as f:
    manifest = json.load(f)

captures = manifest["captures"]
# Test on first 50 captures across various classes
sample_caps = captures[::22][:50]

print(f"Running Diagnostic on {len(sample_caps)} captures...")
estimator = SymbolRateEstimator()

in_candidates_count = 0
rank_1_count = 0
rank_3_count = 0
rank_5_count = 0
total = 0

details = []

for cap in sample_caps:
    iq_path = cap["iq_path"]
    if not os.path.exists(iq_path):
        continue
    with open(iq_path, "rb") as f:
        iq_raw = np.fromfile(f, dtype=np.complex64)
    
    true_baud = float(cap["symbol_rate"])
    sr = float(cap["sample_rate"])
    mod = cap["true_modulation"]
    cfo = float(cap.get("cfo_hz", 0.0))
    snr = float(cap.get("snr_db", 20.0))

    if mod == "UNKNOWN":
        continue

    total += 1
    # 1. Inspect raw candidates
    evidence = extract_dsp_evidence(iq_raw, sr)
    candidates = generate_candidates(evidence, sr)
    cand_rates = [c.rate_hz for c in candidates]

    # Check if true_baud is in candidates within 3%
    matches = [i for i, r in enumerate(cand_rates) if abs(r - true_baud) / true_baud <= 0.03]
    in_cands = len(matches) > 0
    if in_cands:
        in_candidates_count += 1

    # 2. Run estimator
    try:
        pred = estimator.estimate(iq_raw, sample_rate_hz=sr)
        top_k = pred.top_k
        top_rates = [c["rate_hz"] for c in top_k]
        top1_err = abs(pred.best_symbol_rate_hz - true_baud) / true_baud
        
        r1_match = abs(pred.best_symbol_rate_hz - true_baud) / true_baud <= 0.03
        r3_match = any(abs(r - true_baud) / true_baud <= 0.03 for r in top_rates[:3])
        r5_match = any(abs(r - true_baud) / true_baud <= 0.03 for r in top_rates[:5])

        if r1_match:
            rank_1_count += 1
        if r3_match:
            rank_3_count += 1
        if r5_match:
            rank_5_count += 1

        details.append({
            "mod": mod,
            "true_baud": round(true_baud, 1),
            "est_baud": round(pred.best_symbol_rate_hz, 1),
            "err_pct": round(top1_err * 100, 1),
            "in_cands": in_cands,
            "cand_count": len(candidates),
            "cand_rates": [round(r, 1) for r in cand_rates[:6]],
            "cfo": round(cfo, 1),
            "snr": round(snr, 1),
            "status": pred.status
        })
    except Exception as e:
        details.append({
            "mod": mod,
            "true_baud": round(true_baud, 1),
            "error": str(e),
            "in_cands": in_cands,
            "cfo": round(cfo, 1),
        })

print("\n" + "="*80)
print(f"DIAGNOSTIC SUMMARY ({total} signals):")
print(f"Candidate Recall (within 3%): {in_candidates_count}/{total} ({in_candidates_count/total*100:.1f}%)")
print(f"Top-1 Accuracy (within 3%):   {rank_1_count}/{total} ({rank_1_count/total*100:.1f}%)")
print(f"Top-3 Accuracy (within 3%):   {rank_3_count}/{total} ({rank_3_count/total*100:.1f}%)")
print(f"Top-5 Accuracy (within 3%):   {rank_5_count}/{total} ({rank_5_count/total*100:.1f}%)")
print("="*80)

print("\nSAMPLE DETAILS (first 20):")
for d in details[:20]:
    print(d)
