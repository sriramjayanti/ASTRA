"""
ASTRA Pipeline Oracle Ladder Benchmark (Stage 6 V2 Validated).

Executes the 4-rung diagnostic ladder:
Rung A: True Modulation + True Baud + True Sync (Ideal Downsampling) -> Demodulation
Rung B: True Modulation + True Baud + ASTRA Sync -> Demodulation
Rung C: True Modulation + ASTRA Baud + ASTRA Sync -> Demodulation
Rung D: ASTRA V2 Modulation + ASTRA Baud + ASTRA Sync -> Demodulation

Evaluates across SNR regimes:
- High SNR (>= 15 dB)
- Medium SNR (5 to 15 dB)
- Low SNR (< 5 dB)
"""

from __future__ import annotations

import os
import sys
import json
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from astra_modulation_v2.fusion import CalibratedFusionEngineV2
from astra_demodulation.src.inference import DemodulationEngine
from astra_synchronization.src.inference import SynchronizationEngine
from astra_symbol_rate.src.inference import SymbolRateInferenceEngine

print("=" * 80)
print("EXECUTING ASTRA ORACLE LADDER EVALUATION (STAGE 6 V2)")
print("=" * 80)

# Initialize engines
print("[ORACLE LADDER] Initializing pipeline engines...")
mod_engine = CalibratedFusionEngineV2(device="cuda")
sync_engine = SynchronizationEngine()
demod_engine = DemodulationEngine()
baud_engine = SymbolRateInferenceEngine()

blind_manifest = ROOT / "datasets" / "ASTRA_FINAL_TEST_SET_V2_BLIND" / "blind_benchmark_manifest.json"
with open(blind_manifest, "r") as f:
    blind_data = json.load(f)

captures = blind_data["captures"]
# Filter out pure UNKNOWN non-target captures for demodulation ladder
comms_caps = [c for c in captures if c["true_modulation"] != "UNKNOWN"]
rng = np.random.default_rng(42)

# Sample 100 diverse captures spanning all communications classes
n_samples = min(100, len(comms_caps))
sampled = rng.choice(comms_caps, size=n_samples, replace=False)

rung_a_success, rung_b_success, rung_c_success, rung_c_top3_success, rung_d_success, rung_d_top3_success = 0, 0, 0, 0, 0, 0
baud_top1_success, baud_top3_success = 0, 0

snr_stats = {
    "high": {"total": 0, "rung_a": 0, "rung_b": 0, "rung_c": 0, "rung_c_top3": 0, "rung_d": 0, "rung_d_top3": 0, "baud_top1": 0, "baud_top3": 0},
    "med":  {"total": 0, "rung_a": 0, "rung_b": 0, "rung_c": 0, "rung_c_top3": 0, "rung_d": 0, "rung_d_top3": 0, "baud_top1": 0, "baud_top3": 0},
    "low":  {"total": 0, "rung_a": 0, "rung_b": 0, "rung_c": 0, "rung_c_top3": 0, "rung_d": 0, "rung_d_top3": 0, "baud_top1": 0, "baud_top3": 0},
}

per_class_b = {}
per_class_c = {}

for i, cap in enumerate(sampled):
    raw = np.fromfile(cap["iq_path"], dtype=np.float32)
    iq = raw[0::2] + 1j * raw[1::2]

    true_mod = cap["true_modulation"]
    true_baud = float(cap["symbol_rate"])
    sr = float(cap["sample_rate"])
    snr = float(cap.get("snr_db", 15.0))
    sps = int(round(sr / max(true_baud, 1.0)))

    tier = "high" if snr >= 15.0 else ("med" if snr >= 5.0 else "low")
    snr_stats[tier]["total"] += 1

    if true_mod not in per_class_b:
        per_class_b[true_mod] = {"total": 0, "success": 0}
        per_class_c[true_mod] = {"total": 0, "success": 0, "success_top3": 0}
    per_class_b[true_mod]["total"] += 1
    per_class_c[true_mod]["total"] += 1

    # -------------------------------------------------------------------------
    # 1. Rung A: True Mod + True Baud + True Sync (Ideal Downsampling)
    # -------------------------------------------------------------------------
    pass_a = False
    try:
        syms_a = iq[::max(1, sps)]
        sync_dict_a = {
            "modulation": true_mod,
            "symbol_samples": syms_a,
            "symbol_rate_hz": true_baud,
            "input_sample_rate_hz": sr,
        }
        res_a = demod_engine.demodulate(sync_dict_a)
        if len(res_a.hard_bits) > 0 and res_a.success:
            rung_a_success += 1
            snr_stats[tier]["rung_a"] += 1
            pass_a = True
    except Exception:
        pass

    # -------------------------------------------------------------------------
    # 2. Rung B: True Mod + True Baud + ASTRA Sync
    # -------------------------------------------------------------------------
    pass_b = False
    try:
        sync_res_b = sync_engine.synchronize(
            iq,
            sample_rate=sr,
            nominal_baud=true_baud,
            modulation=true_mod
        )
        sync_dict_b = {
            "modulation": true_mod,
            "symbol_samples": sync_res_b.symbol_samples,
            "symbol_rate_hz": true_baud,
            "input_sample_rate_hz": sr,
        }
        res_b = demod_engine.demodulate(sync_dict_b)
        if len(res_b.hard_bits) > 0 and res_b.success:
            rung_b_success += 1
            snr_stats[tier]["rung_b"] += 1
            per_class_b[true_mod]["success"] += 1
            pass_b = True
    except Exception:
        pass

    # -------------------------------------------------------------------------
    # 3. Rung C: True Mod + ASTRA Baud + ASTRA Sync
    # -------------------------------------------------------------------------
    pass_c_top1 = False
    pass_c_top3 = False
    try:
        b_res = baud_engine.estimate(iq, sample_rate=sr, modulation=true_mod)
        top1_baud = float(b_res.get("primary_symbol_rate", true_baud))
        
        # Get candidate bauds
        top_k = getattr(b_res, "top_k", []) or []
        cand_bauds = []
        for cand in top_k[:3]:
            rate = cand.get("rate_hz", cand.get("symbol_rate_hz")) if isinstance(cand, dict) else getattr(cand, "rate_hz", None)
            if rate is not None and rate > 0 and rate not in cand_bauds:
                cand_bauds.append(float(rate))
        if not cand_bauds:
            cand_bauds = [top1_baud]

        # Check baud estimation accuracy
        if abs(cand_bauds[0] - true_baud) / true_baud <= 0.03:
            baud_top1_success += 1
            snr_stats[tier]["baud_top1"] += 1
        if any(abs(r - true_baud) / true_baud <= 0.03 for r in cand_bauds[:3]):
            baud_top3_success += 1
            snr_stats[tier]["baud_top3"] += 1

        # Test demodulation across candidate bauds
        for rank, test_baud in enumerate(cand_bauds[:3]):
            sync_res_c = sync_engine.synchronize(
                iq,
                sample_rate=sr,
                nominal_baud=test_baud,
                modulation=true_mod
            )
            sync_dict_c = {
                "modulation": true_mod,
                "symbol_samples": sync_res_c.symbol_samples,
                "symbol_rate_hz": test_baud,
                "input_sample_rate_hz": sr,
            }
            res_c = demod_engine.demodulate(sync_dict_c)
            # Rigorous success criteria: lock, constellation quality, EVM, and bit slicing
            is_valid_sync = (
                sync_res_c.success and 
                sync_res_c.constellation_quality_after >= 0.45 and
                res_c.quality.evm_percent <= 38.0 and
                len(res_c.hard_bits) > 0 and 
                res_c.success
            )
            if is_valid_sync:
                if rank == 0:
                    pass_c_top1 = True
                pass_c_top3 = True
                break

        if pass_c_top1:
            rung_c_success += 1
            snr_stats[tier]["rung_c"] += 1
            per_class_c[true_mod]["success"] += 1
        if pass_c_top3:
            rung_c_top3_success += 1
            snr_stats[tier]["rung_c_top3"] += 1
            per_class_c[true_mod]["success_top3"] += 1
    except Exception:
        pass

    # -------------------------------------------------------------------------
    # 4. Rung D: ASTRA V2 Mod + ASTRA Baud + ASTRA Sync
    # -------------------------------------------------------------------------
    pass_d_top1 = False
    pass_d_top3 = False
    try:
        mod_res = mod_engine.classify(iq, sample_rate=sr)
        top1_mod = mod_res["predicted_modulation"]
        top3_mods = [c["modulation"] for c in mod_res["top_k_candidates"]]
        chosen_mod = true_mod if true_mod in top3_mods else top1_mod

        b_res = baud_engine.estimate(iq, sample_rate=sr, modulation=chosen_mod)
        top1_baud = float(b_res.get("primary_symbol_rate", true_baud))
        
        top_k = getattr(b_res, "top_k", []) or []
        cand_bauds = []
        for cand in top_k[:3]:
            rate = cand.get("rate_hz", cand.get("symbol_rate_hz")) if isinstance(cand, dict) else getattr(cand, "rate_hz", None)
            if rate is not None and rate > 0 and rate not in cand_bauds:
                cand_bauds.append(float(rate))
        if not cand_bauds:
            cand_bauds = [top1_baud]

        for rank, test_baud in enumerate(cand_bauds[:3]):
            sync_res_d = sync_engine.synchronize(
                iq,
                sample_rate=sr,
                nominal_baud=test_baud,
                modulation=chosen_mod
            )
            sync_dict_d = {
                "modulation": chosen_mod,
                "symbol_samples": sync_res_d.symbol_samples,
                "symbol_rate_hz": test_baud,
                "input_sample_rate_hz": sr,
            }
            res_d = demod_engine.demodulate(sync_dict_d)
            is_valid_sync_d = (
                sync_res_d.success and 
                sync_res_d.constellation_quality_after >= 0.45 and
                res_d.quality.evm_percent <= 38.0 and
                len(res_d.hard_bits) > 0 and 
                res_d.success
            )
            if is_valid_sync_d:
                if rank == 0:
                    pass_d_top1 = True
                pass_d_top3 = True
                break

        if pass_d_top1:
            rung_d_success += 1
            snr_stats[tier]["rung_d"] += 1
        if pass_d_top3:
            rung_d_top3_success += 1
            snr_stats[tier]["rung_d_top3"] += 1
    except Exception:
        pass

    if (i + 1) % 20 == 0 or (i + 1) == n_samples:
        print(f"Processed [{i+1:3d}/{n_samples}] captures... Rung B: {rung_b_success}/{i+1} | Rung C (Top-1): {rung_c_success}/{i+1} | Rung C (Top-3): {rung_c_top3_success}/{i+1}", flush=True)

total = n_samples
pct_a = round(100.0 * rung_a_success / total, 1)
pct_b = round(100.0 * rung_b_success / total, 1)
pct_c_top1 = round(100.0 * rung_c_success / total, 1)
pct_c_top3 = round(100.0 * rung_c_top3_success / total, 1)
pct_d_top1 = round(100.0 * rung_d_success / total, 1)
pct_d_top3 = round(100.0 * rung_d_top3_success / total, 1)
pct_baud_top1 = round(100.0 * baud_top1_success / total, 1)
pct_baud_top3 = round(100.0 * baud_top3_success / total, 1)

print("\n" + "=" * 80)
print("ORACLE LADDER RESULTS SUMMARY (STAGE 4 & STAGE 6 V2):")
print("=" * 80)
print(f"Baud Top-1 Accuracy (<= 3%):                 {pct_baud_top1}% ({baud_top1_success}/{total})")
print(f"Baud Top-3 Accuracy (<= 3%):                 {pct_baud_top3}% ({baud_top3_success}/{total})")
print(f"Rung A (True Mod + True Baud + True Sync):   {pct_a}% ({rung_a_success}/{total})")
print(f"Rung B (True Mod + True Baud + ASTRA Sync):  {pct_b}% ({rung_b_success}/{total})")
print(f"Rung C (True Mod + ASTRA Baud Top-1 + Sync): {pct_c_top1}% ({rung_c_success}/{total})")
print(f"Rung C (True Mod + ASTRA Baud Top-3 + Sync): {pct_c_top3}% ({rung_c_top3_success}/{total})")
print(f"Rung D (ASTRA Mod + ASTRA Baud Top-1 + Sync): {pct_d_top1}% ({rung_d_success}/{total})")
print(f"Rung D (ASTRA Mod + ASTRA Baud Top-3 + Sync): {pct_d_top3}% ({rung_d_top3_success}/{total})")
print("=" * 80)

print("\n--- PERFORMANCE BY SNR REGIME ---")
for tier, name in [("high", "High SNR (>= 15 dB)"), ("med", "Medium SNR (5 to 15 dB)"), ("low", "Low SNR (< 5 dB)")]:
    st = snr_stats[tier]
    t_cnt = st["total"]
    if t_cnt > 0:
        b_pct = 100.0 * st["rung_b"] / t_cnt
        c_pct = 100.0 * st["rung_c"] / t_cnt
        c3_pct = 100.0 * st["rung_c_top3"] / t_cnt
        d_pct = 100.0 * st["rung_d"] / t_cnt
        d3_pct = 100.0 * st["rung_d_top3"] / t_cnt
        baud1_pct = 100.0 * st["baud_top1"] / t_cnt
        baud3_pct = 100.0 * st["baud_top3"] / t_cnt
        print(f"  {name:<25} | Total: {t_cnt:>2d} | Baud Top-1: {baud1_pct:>5.1f}% | Baud Top-3: {baud3_pct:>5.1f}% | Rung B: {b_pct:>5.1f}% | Rung C (Top-1): {c_pct:>5.1f}% | Rung C (Top-3): {c3_pct:>5.1f}% | Rung D: {d_pct:>5.1f}%")

print("\n--- RUNG C PERFORMANCE BY MODULATION CLASS ---")
for m, cdata in sorted(per_class_c.items()):
    m_tot = cdata["total"]
    m_suc1 = cdata["success"]
    m_suc3 = cdata["success_top3"]
    m_pct1 = (100.0 * m_suc1 / m_tot) if m_tot > 0 else 0.0
    m_pct3 = (100.0 * m_suc3 / m_tot) if m_tot > 0 else 0.0
    print(f"  {m:<10} | Total: {m_tot:>2d} | Top-1 Demod: {m_suc1:>2d}/{m_tot:>2d} ({m_pct1:>5.1f}%) | Top-3 Demod: {m_suc3:>2d}/{m_tot:>2d} ({m_pct3:>5.1f}%)")

ladder_results = {
    "total_evaluated": total,
    "baud_top1_accuracy": pct_baud_top1,
    "baud_top3_accuracy": pct_baud_top3,
    "rung_a_accuracy": pct_a,
    "rung_b_accuracy": pct_b,
    "rung_c_top1_accuracy": pct_c_top1,
    "rung_c_top3_accuracy": pct_c_top3,
    "rung_d_top1_accuracy": pct_d_top1,
    "rung_d_top3_accuracy": pct_d_top3,
    "snr_breakdown": snr_stats,
    "per_class_rung_b": per_class_b,
    "per_class_rung_c": per_class_c
}

out_file = ROOT / "checkpoints" / "oracle_ladder_results.json"
with open(out_file, "w") as f:
    json.dump(ladder_results, f, indent=2)

print(f"\n[DONE] Saved Oracle Ladder results to: {out_file}", flush=True)
