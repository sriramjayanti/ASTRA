"""
ASTRA Stage 3 — Candidate Ranking, Branch Transition & Fusion Optimization Harness.

Evaluates:
1. Exact rank transitions (1D -> 2D -> Fused -> RF-Supported Fused -> Optimized Fused).
2. Grid search calibration on isolated Validation Set (330 captures).
3. Blind evaluation on untouched ASTRA_FINAL_TEST_SET_V2_BLIND (1100 captures).
4. Class-specific ranking patterns (PSK siblings, QAM siblings, FSK/MSK).
5. Downstream Stage 5/6 candidate beam size and runtime impact analysis.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

import numpy as np
import torch
import torch.nn.functional as F

from astra_modulation_v2.class_schema import (
    MODULATION_CLASSES_V2,
    MODULATION_FAMILIES_V2,
    NUM_CLASSES_V2,
    get_class_index,
    get_class_name,
)
from astra_modulation_v2.dataset_builder import IQPreprocessorV2, iq_to_tensor_1d, iq_to_spectrogram_2d
from astra_modulation_v2.models.resnet1d import ResNet1DV2
from astra_modulation_v2.models.spectrogram_cnn import SpectrogramCNN2DV2
from astra_modulation_v2.rf_support import RFFamilySupportAdapter


def extract_capture_features_and_logits(
    captures: List[Dict[str, Any]],
    model_1d: ResNet1DV2,
    model_2d: SpectrogramCNN2DV2,
    rf_adapter: RFFamilySupportAdapter,
    preprocessor: IQPreprocessorV2,
    device: torch.device,
    window_size: int = 2048,
) -> List[Dict[str, Any]]:
    """Extracts raw logits and evidence for a list of captures."""
    results = []
    
    for cap in captures:
        raw = np.fromfile(cap["iq_path"], dtype=np.float32)
        iq = raw[0::2] + 1j * raw[1::2]
        true_mod = cap.get("true_modulation", cap.get("modulation", "UNKNOWN"))
        sr = float(cap.get("sample_rate", 192000.0))
        
        iq_clean = preprocessor.process(iq)
        if len(iq_clean) < window_size:
            iq_window = np.pad(iq_clean, (0, window_size - len(iq_clean)), mode="constant")
        else:
            iq_window = iq_clean[:window_size]
            
        # 1D logits
        t_1d = iq_to_tensor_1d(iq_window).unsqueeze(0).to(device)
        with torch.no_grad():
            logits_1d, _ = model_1d(t_1d)
            logits_1d_np = logits_1d.cpu().numpy()[0]
            
        # 2D logits
        t_2d = iq_to_spectrogram_2d(iq_window).unsqueeze(0).to(device)
        with torch.no_grad():
            logits_2d, _ = model_2d(t_2d)
            logits_2d_np = logits_2d.cpu().numpy()[0]
            
        # RF family evidence
        rf_fam_probs = rf_adapter.get_family_probabilities(iq, sample_rate=sr)
        rf_vec = rf_adapter.get_class_support_vector(rf_fam_probs)
        
        results.append({
            "capture_id": cap.get("capture_id", Path(cap["iq_path"]).stem),
            "true_modulation": true_mod,
            "true_idx": get_class_index(true_mod),
            "sample_rate": sr,
            "snr_db": cap.get("snr_db", 20.0),
            "cfo_hz": cap.get("cfo_hz", 0.0),
            "channel_type": cap.get("channel_type", "AWGN"),
            "logits_1d": logits_1d_np,
            "logits_2d": logits_2d_np,
            "rf_fam_probs": rf_fam_probs,
            "rf_vec": rf_vec,
        })
        
    return results


def compute_ranks_and_metrics(
    extracted_data: List[Dict[str, Any]],
    w_1d: float = 0.75,
    w_2d: float = 0.25,
    t_1d: float = 1.0,
    t_2d: float = 1.0,
    rf_beta: float = 0.20,
    family_boost: bool = False,
    family_boost_alpha: float = 0.15,
) -> Dict[str, Any]:
    """Computes ranks, accuracy, Top-K retention, and transition statistics."""
    tot = len(extracted_data)
    hits = {k: 0 for k in range(1, 12)}
    hits_1d = {k: 0 for k in range(1, 12)}
    hits_2d = {k: 0 for k in range(1, 12)}
    hits_base_fused = {k: 0 for k in range(1, 12)}
    
    per_class_hits = {c: {1: 0, 3: 0, 5: 0, 6: 0, "total": 0} for c in MODULATION_CLASSES_V2}
    rank_transitions = []
    
    w_sum = w_1d + w_2d
    w1 = w_1d / max(w_sum, 1e-6)
    w2 = w_2d / max(w_sum, 1e-6)
    
    for item in extracted_data:
        true_idx = item["true_idx"]
        true_mod = item["true_modulation"]
        per_class_hits[true_mod]["total"] += 1
        
        # Softmax probabilities
        exp_1d = np.exp((item["logits_1d"] - np.max(item["logits_1d"])) / max(t_1d, 0.05))
        p_1d = exp_1d / np.sum(exp_1d)
        
        exp_2d = np.exp((item["logits_2d"] - np.max(item["logits_2d"])) / max(t_2d, 0.05))
        p_2d = exp_2d / np.sum(exp_2d)
        
        # Base fused
        p_base = w1 * p_1d + w2 * p_2d
        
        # RF injection
        p_rf_fused = p_base * (1.0 + rf_beta * item["rf_vec"])
        p_rf_fused = p_rf_fused / np.sum(p_rf_fused)
        
        # Optional Adaptive Family Sibling Expansion / Boost
        if family_boost:
            # If a family has high confidence from RF and top neural candidates, boost all members of that family
            fam_support = {}
            for fam in ["FSK", "PSK", "QAM", "UNKNOWN"]:
                fam_support[fam] = item["rf_fam_probs"].get(fam, 0.0)
            
            p_boosted = p_rf_fused.copy()
            for idx, cname in enumerate(MODULATION_CLASSES_V2):
                fam = MODULATION_FAMILIES_V2.get(cname, "UNKNOWN")
                if fam_support.get(fam, 0.0) > 0.40:
                    p_boosted[idx] += family_boost_alpha * fam_support[fam]
            p_final = p_boosted / np.sum(p_boosted)
        else:
            p_final = p_rf_fused
            
        # Compute Ranks (1-indexed)
        rank_1d = int(np.where(np.argsort(p_1d)[::-1] == true_idx)[0][0]) + 1
        rank_2d = int(np.where(np.argsort(p_2d)[::-1] == true_idx)[0][0]) + 1
        rank_base = int(np.where(np.argsort(p_base)[::-1] == true_idx)[0][0]) + 1
        rank_final = int(np.where(np.argsort(p_final)[::-1] == true_idx)[0][0]) + 1
        
        for k in range(1, 12):
            if rank_1d <= k: hits_1d[k] += 1
            if rank_2d <= k: hits_2d[k] += 1
            if rank_base <= k: hits_base_fused[k] += 1
            if rank_final <= k: hits[k] += 1
            
        if rank_final <= 1: per_class_hits[true_mod][1] += 1
        if rank_final <= 3: per_class_hits[true_mod][3] += 1
        if rank_final <= 5: per_class_hits[true_mod][5] += 1
        if rank_final <= 6: per_class_hits[true_mod][6] += 1
        
        # Cosine agreement
        norm1 = np.linalg.norm(p_1d)
        norm2 = np.linalg.norm(p_2d)
        agreement = float(np.dot(p_1d, p_2d) / (norm1 * norm2)) if (norm1 > 0 and norm2 > 0) else 1.0
        disagreement = 1.0 - agreement
        
        sorted_p = np.sort(p_final)[::-1]
        margin = float(sorted_p[0] - sorted_p[1]) if len(sorted_p) > 1 else 1.0
        
        rank_transitions.append({
            "capture_id": item["capture_id"],
            "true_modulation": true_mod,
            "rank_1d": rank_1d,
            "rank_2d": rank_2d,
            "rank_base_fused": rank_base,
            "rank_final": rank_final,
            "margin": margin,
            "agreement": agreement,
            "disagreement": disagreement,
            "rf_fam_probs": item["rf_fam_probs"],
        })
        
    return {
        "hits_final": {k: float(round(hits[k] / tot * 100, 2)) for k in range(1, 12)},
        "hits_1d": {k: float(round(hits_1d[k] / tot * 100, 2)) for k in range(1, 12)},
        "hits_2d": {k: float(round(hits_2d[k] / tot * 100, 2)) for k in range(1, 12)},
        "hits_base_fused": {k: float(round(hits_base_fused[k] / tot * 100, 2)) for k in range(1, 12)},
        "per_class": per_class_hits,
        "rank_transitions": rank_transitions,
    }


def main():
    root = Path(__file__).resolve().parent.parent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing Ranking & Fusion Audit on device: {device}")
    
    # 1. Load Models
    model_1d = ResNet1DV2(in_channels=2, num_classes=NUM_CLASSES_V2).to(device)
    ckpt_1d = torch.load(root / "checkpoints" / "astra_resnet1d_v2.pt", map_location=device)
    model_1d.load_state_dict(ckpt_1d.get("state_dict", ckpt_1d))
    model_1d.eval()
    
    model_2d = SpectrogramCNN2DV2(in_channels=1, num_classes=NUM_CLASSES_V2).to(device)
    ckpt_2d = torch.load(root / "checkpoints" / "astra_spectrogram_cnn_v2.pt", map_location=device)
    model_2d.load_state_dict(ckpt_2d.get("state_dict", ckpt_2d))
    model_2d.eval()
    
    rf_adapter = RFFamilySupportAdapter()
    preprocessor = IQPreprocessorV2(remove_dc=True, normalize_rms=True)
    
    # 2. Load Validation Captures (Isolated Calibration Set)
    with open(root / "datasets" / "ASTRA_MODULATION_V3_AUGMENTED" / "manifest.json") as f:
        v3_manifest = json.load(f)
    val_caps = [c for c in v3_manifest["captures"] if c.get("split") == "val"]
    print(f"Loaded {len(val_caps)} Validation captures for tuning...")
    val_extracted = extract_capture_features_and_logits(val_caps, model_1d, model_2d, rf_adapter, preprocessor, device)
    
    # 3. Load Blind Benchmark Captures (1100 captures)
    with open(root / "datasets" / "ASTRA_FINAL_TEST_SET_V2_BLIND" / "blind_benchmark_manifest.json") as f:
        blind_manifest = json.load(f)
    blind_caps = blind_manifest["captures"]
    print(f"Loaded {len(blind_caps)} Blind Benchmark captures for evaluation...")
    blind_extracted = extract_capture_features_and_logits(blind_caps, model_1d, model_2d, rf_adapter, preprocessor, device)
    
    # 4. Grid Search Optimization on Validation Set ONLY
    print("\n--- Running Grid Search on Validation Set (330 captures) ---")
    best_val_score = 0.0
    best_config = {}
    
    weight_candidates = [(0.50, 0.50), (0.60, 0.40), (0.70, 0.30), (0.75, 0.25), (0.80, 0.20), (0.85, 0.15), (0.90, 0.10)]
    t1_candidates = [0.8, 1.0, 1.2, 1.5]
    t2_candidates = [0.8, 1.0, 1.2, 1.5]
    rf_beta_candidates = [0.0, 0.10, 0.20, 0.30, 0.40]
    fam_boost_candidates = [(False, 0.0), (True, 0.08), (True, 0.12), (True, 0.18)]
    
    for (w1, w2) in weight_candidates:
        for t1 in t1_candidates:
            for t2 in t2_candidates:
                for beta in rf_beta_candidates:
                    for (fam_b, fam_a) in fam_boost_candidates:
                        res = compute_ranks_and_metrics(
                            val_extracted,
                            w_1d=w1,
                            w_2d=w2,
                            t_1d=t1,
                            t_2d=t2,
                            rf_beta=beta,
                            family_boost=fam_b,
                            family_boost_alpha=fam_a,
                        )
                        # Score combines Top-1, Top-3, and Top-5 retention
                        score = 0.2 * res["hits_final"][1] + 0.5 * res["hits_final"][3] + 0.3 * res["hits_final"][5]
                        if score > best_val_score:
                            best_val_score = score
                            best_config = {
                                "w_1d": w1,
                                "w_2d": w2,
                                "t_1d": t1,
                                "t_2d": t2,
                                "rf_beta": beta,
                                "family_boost": fam_b,
                                "family_boost_alpha": fam_a,
                                "val_top1": res["hits_final"][1],
                                "val_top3": res["hits_final"][3],
                                "val_top5": res["hits_final"][5],
                            }
                            
    print(f"Optimal Config from Validation Search: {best_config}")
    
    # 5. Evaluate Baseline vs Optimal on Blind Benchmark
    baseline_blind = compute_ranks_and_metrics(blind_extracted, w_1d=0.75, w_2d=0.25, t_1d=1.0, t_2d=1.0, rf_beta=0.20)
    opt_blind = compute_ranks_and_metrics(
        blind_extracted,
        w_1d=best_config["w_1d"],
        w_2d=best_config["w_2d"],
        t_1d=best_config["t_1d"],
        t_2d=best_config["t_2d"],
        rf_beta=best_config["rf_beta"],
        family_boost=best_config["family_boost"],
        family_boost_alpha=best_config["family_boost_alpha"],
    )
    
    # 6. Save Detailed Audit Report JSON
    audit_report = {
        "best_config": best_config,
        "validation_metrics": {
            "top1": best_config["val_top1"],
            "top3": best_config["val_top3"],
            "top5": best_config["val_top5"],
        },
        "blind_baseline_ladder": baseline_blind["hits_final"],
        "blind_optimized_ladder": opt_blind["hits_final"],
        "blind_1d_ladder": opt_blind["hits_1d"],
        "blind_2d_ladder": opt_blind["hits_2d"],
        "blind_base_fused_ladder": opt_blind["hits_base_fused"],
        "per_class_baseline": baseline_blind["per_class"],
        "per_class_optimized": opt_blind["per_class"],
        "rank_transitions": opt_blind["rank_transitions"],
    }
    
    out_json = root / "checkpoints" / "stage3_ranking_audit_data.json"
    with open(out_json, "w") as f:
        json.dump(audit_report, f, indent=2)
    print(f"\nSaved ranking audit report data to: {out_json}")
    
    # Print summary
    print("\n================ BLIND BENCHMARK RANKING COMPARISON ================")
    print(f"{'Beam Width':<12} | {'1D Alone':<10} | {'2D Alone':<10} | {'Base Fused':<12} | {'V3 Baseline':<12} | {'Optimized Fused':<16}")
    print("-" * 80)
    for k in [1, 2, 3, 4, 5, 6]:
        print(f"Top-{k:<7} | {opt_blind['hits_1d'][k]:<9.2f}% | {opt_blind['hits_2d'][k]:<9.2f}% | {opt_blind['hits_base_fused'][k]:<11.2f}% | {baseline_blind['hits_final'][k]:<11.2f}% | {opt_blind['hits_final'][k]:<15.2f}%")


if __name__ == "__main__":
    main()
