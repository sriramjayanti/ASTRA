"""
Stage 4 Baud Estimation Harmonic & Candidate Ranking Audit.
Analyzes 100 representative blind captures across all 10 active modulation classes.
"""

from __future__ import annotations
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

import numpy as np
from astra_symbol_rate.src.inference import SymbolRateEstimator
from astra_symbol_rate.src.dataset_builder import extract_dsp_evidence
from astra_symbol_rate.src.candidate_generator import generate_candidates


def main():
    manifest_path = root / "datasets" / "ASTRA_FINAL_TEST_SET_V2_BLIND" / "blind_benchmark_manifest.json"
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    captures = manifest["captures"]
    
    # Select 10 captures per non-UNKNOWN modulation class = 100 captures total
    class_caps: Dict[str, List[Dict[str, Any]]] = {}
    for cap in captures:
        mod = cap["true_modulation"]
        if mod == "UNKNOWN":
            continue
        if mod not in class_caps:
            class_caps[mod] = []
        if len(class_caps[mod]) < 10:
            class_caps[mod].append(cap)

    selected = []
    for mod in ["BPSK", "QPSK", "8PSK", "DQPSK", "16QAM", "64QAM", "256QAM", "2-FSK", "4-FSK", "MSK"]:
        selected.extend(class_caps.get(mod, []))

    print(f"Auditing Stage 4 on {len(selected)} captures across 10 modulation classes...\n")

    estimator = SymbolRateEstimator(top_k=5)
    
    results = []
    class_stats = {m: {"total": 0, "recall@1": 0, "recall@3": 0, "recall@5": 0, "in_all_cands": 0, "harmonic_confusions": {}} for m in class_caps}
    
    for cap in selected:
        raw = np.fromfile(cap["iq_path"], dtype=np.complex64)
        true_baud = float(cap["symbol_rate"])
        sr = float(cap["sample_rate"])
        mod = cap["true_modulation"]
        snr = float(cap.get("snr_db", 20.0))
        cfo = float(cap.get("cfo_hz", 0.0))
        
        class_stats[mod]["total"] += 1
        
        # 1. DSP Evidence & Candidate Generation
        evidence = extract_dsp_evidence(raw, sr)
        candidates = generate_candidates(evidence, sr, modulation_hint=mod)
        cand_rates = [c.rate_hz for c in candidates]
        
        # Check presence in raw candidate pool
        in_pool = any(abs(r - true_baud) / true_baud <= 0.03 for r in cand_rates)
        if in_pool:
            class_stats[mod]["in_all_cands"] += 1
            
        # 2. Run Estimator with modulation hint
        pred = estimator.estimate(raw, sample_rate_hz=sr, modulation=mod)
        top_k_rates = [c["rate_hz"] for c in pred.top_k]
        
        r1_match = abs(pred.best_symbol_rate_hz - true_baud) / true_baud <= 0.03
        r3_match = any(abs(r - true_baud) / true_baud <= 0.03 for r in top_k_rates[:3])
        r5_match = any(abs(r - true_baud) / true_baud <= 0.03 for r in top_k_rates[:5])
        
        if r1_match: class_stats[mod]["recall@1"] += 1
        if r3_match: class_stats[mod]["recall@3"] += 1
        if r5_match: class_stats[mod]["recall@5"] += 1
        
        # Check what ratio the Top-1 prediction is relative to true baud
        ratio = pred.best_symbol_rate_hz / true_baud
        ratio_label = "Exact (1.0x)"
        if abs(ratio - 1.0) <= 0.03:
            ratio_label = "Exact (1.0x)"
        elif abs(ratio - 0.5) <= 0.04:
            ratio_label = "Subharmonic (0.5x / Rs/2)"
        elif abs(ratio - 0.75) <= 0.04:
            ratio_label = "Subharmonic (0.75x / 3/4Rs)"
        elif abs(ratio - 2.0) <= 0.06:
            ratio_label = "Harmonic (2.0x / 2Rs)"
        elif abs(ratio - 3.0) <= 0.08:
            ratio_label = "Harmonic (3.0x / 3Rs)"
        elif abs(ratio - 4.0) <= 0.10:
            ratio_label = "Harmonic (4.0x / 4Rs)"
        else:
            ratio_label = f"Other ({ratio:.2f}x)"
            
        class_stats[mod]["harmonic_confusions"][ratio_label] = class_stats[mod]["harmonic_confusions"].get(ratio_label, 0) + 1
        
        results.append({
            "capture_id": Path(cap["iq_path"]).stem,
            "mod": mod,
            "true_baud": true_baud,
            "pred_top1": pred.best_symbol_rate_hz,
            "top_k": top_k_rates,
            "ratio": ratio,
            "ratio_label": ratio_label,
            "obw_hz": evidence.occupied_bandwidth_hz,
            "obw_ratio": evidence.occupied_bandwidth_hz / true_baud if true_baud > 0 else 0,
            "cand_sources": [(c.rate_hz, c.sources, c.score) for c in candidates[:6]],
        })

    print("=" * 80)
    print(f"{'Modulation':<10} | {'Recall@1 (<=3%)':<16} | {'Recall@3 (<=3%)':<16} | {'Recall@5 (<=3%)':<16} | {'In DSP Pool':<12}")
    print("-" * 80)
    tot_r1, tot_r3, tot_r5, tot_pool, tot_cnt = 0, 0, 0, 0, 0
    for mod, s in class_stats.items():
        cnt = s["total"]
        tot_cnt += cnt
        tot_r1 += s["recall@1"]
        tot_r3 += s["recall@3"]
        tot_r5 += s["recall@5"]
        tot_pool += s["in_all_cands"]
        print(f"{mod:<10} | {s['recall@1']}/{cnt} ({s['recall@1']/cnt*100:5.1f}%)    | {s['recall@3']}/{cnt} ({s['recall@3']/cnt*100:5.1f}%)    | {s['recall@5']}/{cnt} ({s['recall@5']/cnt*100:5.1f}%)    | {s['in_all_cands']}/{cnt} ({s['in_all_cands']/cnt*100:5.1f}%)")
    print("-" * 80)
    print(f"{'TOTAL':<10} | {tot_r1}/{tot_cnt} ({tot_r1/tot_cnt*100:5.1f}%)    | {tot_r3}/{tot_cnt} ({tot_r3/tot_cnt*100:5.1f}%)    | {tot_r5}/{tot_cnt} ({tot_r5/tot_cnt*100:5.1f}%)    | {tot_pool}/{tot_cnt} ({tot_pool/tot_cnt*100:5.1f}%)")
    print("=" * 80)

    print("\nHARMONIC CONFUSION PATTERNS BY MODULATION CLASS:")
    for mod, s in class_stats.items():
        print(f"\n--- {mod} (Top-1 Ratios) ---")
        for lbl, count in s["harmonic_confusions"].items():
            print(f"  {lbl:<30}: {count} captures ({count/s['total']*100:.1f}%)")

    # Save detailed audit json
    audit_file = root / "checkpoints" / "stage4_harmonic_audit.json"
    with open(audit_file, "w") as f:
        json.dump({"summary": class_stats, "details": results}, f, indent=2)
    print(f"\nSaved detailed audit to: {audit_file}")


if __name__ == "__main__":
    main()
