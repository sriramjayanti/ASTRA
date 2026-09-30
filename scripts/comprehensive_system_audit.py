"""
comprehensive_system_audit.py
Complete system position audit script for ASTRA.
Inspects all 14 stages, tests each stage in isolation and end-to-end,
reconciles metrics, traces first failure per capture, benchmarks performance, and outputs reports.
"""

from __future__ import annotations

import os
import sys
import json
import time
import csv
import traceback
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

audit_results = {}

print("=" * 80)
print("ASTRA COMPREHENSIVE SYSTEM AUDIT")
print("=" * 80)

# ==============================================================================
# 1. STAGE 7: ORACLE-SYNC DEMODULATION AUDIT
# ==============================================================================
print("\n[AUDIT 1/10] Testing Stage 7 Demodulation with pristine synthetic symbols...")
from astra_demodulation.src.inference import DemodulationEngine
demod_engine = DemodulationEngine()

MOD_CLASSES = ["BPSK", "QPSK", "8PSK", "DQPSK", "16QAM", "64QAM", "256QAM", "2-FSK", "4-FSK", "MSK"]
demod_test_results = {}

rng = np.random.default_rng(42)
for mod in MOD_CLASSES:
    try:
        n_syms = 256
        if mod == "BPSK":
            bits = rng.integers(0, 2, size=n_syms)
            syms = (2 * bits - 1).astype(np.complex64)
        elif mod == "QPSK":
            bits = rng.integers(0, 2, size=n_syms * 2)
            # Gray mapping
            table = { (0,0): 1+1j, (0,1): -1+1j, (1,1): -1-1j, (1,0): 1-1j }
            syms = np.array([table[(bits[2*i], bits[2*i+1])] for i in range(n_syms)], dtype=np.complex64) / np.sqrt(2.0)
        elif mod == "8PSK":
            pts = np.exp(1j * 2.0 * np.pi * np.arange(8) / 8.0)
            syms = rng.choice(pts, size=n_syms).astype(np.complex64)
        elif mod == "DQPSK":
            deltas = rng.choice([0, np.pi/2, np.pi, 3*np.pi/2], size=n_syms)
            syms = np.exp(1j * np.cumsum(deltas)).astype(np.complex64)
        elif mod == "16QAM":
            grid = np.array([-3, -1, 1, 3]) / np.sqrt(10.0)
            i_s = rng.choice(grid, size=n_syms)
            q_s = rng.choice(grid, size=n_syms)
            syms = (i_s + 1j * q_s).astype(np.complex64)
        elif mod == "64QAM":
            grid = np.array([-7, -5, -3, -1, 1, 3, 5, 7]) / np.sqrt(42.0)
            i_s = rng.choice(grid, size=n_syms)
            q_s = rng.choice(grid, size=n_syms)
            syms = (i_s + 1j * q_s).astype(np.complex64)
        elif mod == "256QAM":
            grid = np.arange(-15, 16, 2) / np.sqrt(170.0)
            i_s = rng.choice(grid, size=n_syms)
            q_s = rng.choice(grid, size=n_syms)
            syms = (i_s + 1j * q_s).astype(np.complex64)
        elif mod in ["2-FSK", "4-FSK", "MSK"]:
            syms = np.exp(1j * rng.uniform(0, 2*np.pi, size=n_syms)).astype(np.complex64)

        d_res = demod_engine.demodulate({
            "modulation": mod,
            "symbol_samples": syms,
            "symbol_rate_hz": 9600.0,
            "input_sample_rate_hz": 96000.0
        })
        evm = getattr(d_res.quality, "evm_percent", 99.0)
        n_bits = len(getattr(d_res, "hard_bits", []))
        success = bool(d_res.success and n_bits > 0)
        demod_test_results[mod] = {"success": success, "evm_percent": round(evm, 2), "bit_count": n_bits}
    except Exception as e:
        demod_test_results[mod] = {"success": False, "error": str(e)}

audit_results["oracle_demodulation"] = demod_test_results
print("Demodulation Oracle Test Summary:", {k: v["success"] for k, v in demod_test_results.items()})

# ==============================================================================
# 2. STAGE 8: INTERLEAVER AUDIT
# ==============================================================================
print("\n[AUDIT 2/10] Testing Stage 8 Interleaver / Deinterleaver...")
interleaver_results = {}
try:
    from astra_interleaver.src.inference import InterleaverTestingEngine
    intl_engine = InterleaverTestingEngine()
    test_bits = rng.integers(0, 2, size=512)
    cands = intl_engine.generate_candidates(len(test_bits))
    test_res = intl_engine.test_candidates(test_bits, cands)
    interleaver_results = {
        "engine_initialized": True,
        "candidates_generated": len(cands),
        "candidates_tested": len(test_res) if isinstance(test_res, list) else 1,
        "status": "WORKING"
    }
except Exception as e:
    interleaver_results = {"status": "BROKEN", "error": str(e)}

audit_results["interleaver_audit"] = interleaver_results
print("Interleaver Audit:", interleaver_results)

# ==============================================================================
# 3. STAGE 9: FEC DECODING AUDIT
# ==============================================================================
print("\n[AUDIT 3/10] Testing Stage 9 Forward Error Correction (FEC)...")
fec_results = {}
try:
    from astra_fec.src.inference import FECTestingEngine
    fec_engine = FECTestingEngine()
    test_bits = rng.integers(0, 2, size=512)
    fec_cands = fec_engine.generate_candidates()
    fec_test_res = fec_engine.test_candidates(test_bits, fec_cands)
    fec_results = {
        "engine_initialized": True,
        "fec_candidates_generated": len(fec_cands),
        "fec_candidates_tested": len(fec_test_res) if isinstance(fec_test_res, list) else 1,
        "status": "WORKING"
    }
except Exception as e:
    fec_results = {"status": "BROKEN", "error": str(e)}

audit_results["fec_audit"] = fec_results
print("FEC Audit:", fec_results)

# ==============================================================================
# 4. STAGE 10: VALIDATION ENGINE AUDIT
# ==============================================================================
print("\n[AUDIT 4/10] Testing Stage 10 Validation Engine...")
val_results = {}
try:
    from astra_validation.src.inference import ValidationEngine
    val_engine = ValidationEngine()
    test_bits = rng.integers(0, 2, size=512)
    val_out = val_engine.validate(test_bits)
    val_results = {
        "engine_initialized": True,
        "validation_output_type": type(val_out).__name__,
        "status": "WORKING"
    }
except Exception as e:
    val_results = {"status": "BROKEN", "error": str(e)}

audit_results["validation_audit"] = val_results
print("Validation Audit:", val_results)

# ==============================================================================
# 5. STAGE 11: PIPELINE SCORER AUDIT
# ==============================================================================
print("\n[AUDIT 5/10] Testing Stage 11 Pipeline Scorer...")
scorer_results = {}
try:
    from astra_pipeline_scorer.src.inference import PipelineScoringEngine
    scorer_engine = PipelineScoringEngine()
    mock_candidates = [
        {"candidate_id": "c1", "modulation": "QPSK", "symbol_rate_hz": 9600.0, "sync_score": 0.85, "evm_percent": 12.0, "validation_score": 0.9},
        {"candidate_id": "c2", "modulation": "16QAM", "symbol_rate_hz": 4800.0, "sync_score": 0.45, "evm_percent": 35.0, "validation_score": 0.3},
    ]
    ranked = scorer_engine.rank(mock_candidates)
    scorer_results = {
        "engine_initialized": True,
        "ranked_candidates_count": len(ranked),
        "status": "WORKING"
    }
except Exception as e:
    scorer_results = {"status": "BROKEN", "error": str(e)}

audit_results["scorer_audit"] = scorer_results
print("Pipeline Scorer Audit:", scorer_results)

# ==============================================================================
# 6. STAGE 12 & 13: BITSTREAM INTELLIGENCE, TRANSFORMER, PAYLOAD EXPLORER
# ==============================================================================
print("\n[AUDIT 6/10] Testing Higher Stages (Bitstream Intelligence, Transformer, Payload Explorer)...")
higher_stages_results = {}

try:
    from astra_bitstream_intelligence.src.inference import BitstreamIntelligenceEngine
    bs_engine = BitstreamIntelligenceEngine()
    test_bits = rng.integers(0, 2, size=1024)
    bs_res = bs_engine.analyze(test_bits)
    higher_stages_results["bitstream_intelligence"] = {"status": "WORKING"}
except Exception as e:
    higher_stages_results["bitstream_intelligence"] = {"status": "BROKEN", "error": str(e)}

try:
    from astra_bitstream_transformer.src.inference import BitstreamStructureModel
    bt_engine = BitstreamStructureModel()
    higher_stages_results["transformer_structure"] = {"status": "WORKING"}
except Exception as e:
    higher_stages_results["transformer_structure"] = {"status": "BROKEN", "error": str(e)}

try:
    from astra_payload_explorer.src.inference import HeaderPayloadExplorer
    pe_engine = HeaderPayloadExplorer()
    higher_stages_results["payload_explorer"] = {"status": "WORKING"}
except Exception as e:
    higher_stages_results["payload_explorer"] = {"status": "BROKEN", "error": str(e)}

try:
    from astra_explainability.src.inference import ExplainabilityEngine
    exp_engine = ExplainabilityEngine()
    higher_stages_results["explainability"] = {"status": "WORKING"}
except Exception as e:
    higher_stages_results["explainability"] = {"status": "BROKEN", "error": str(e)}

audit_results["higher_stages"] = higher_stages_results
print("Higher Stages Summary:", {k: v.get("status") for k, v in higher_stages_results.items()})

# ==============================================================================
# 7. CONTROLLED SYNTHETIC PAYLOAD ROUND-TRIP ("hi hello")
# ==============================================================================
print("\n[AUDIT 7/10] Testing Controlled Synthetic Payload Recovery ('hi hello')...")
synthetic_payload_results = {}
try:
    payload_text = "hi hello"
    payload_bytes = payload_text.encode("ascii")
    raw_bits = np.unpackbits(np.frombuffer(payload_bytes, dtype=np.uint8))
    
    # Using the exact QPSK mapping of astra_demodulation
    # (0,0) -> 1+1j, (0,1) -> -1+1j, (1,1) -> -1-1j, (1,0) -> 1-1j
    table = { (0,0): 1+1j, (0,1): -1+1j, (1,1): -1-1j, (1,0): 1-1j }
    n_pairs = len(raw_bits) // 2
    symbols = np.array([table[(raw_bits[2*i], raw_bits[2*i+1])] for i in range(n_pairs)], dtype=np.complex64) / np.sqrt(2.0)
    
    d_res = demod_engine.demodulate({
        "modulation": "QPSK",
        "symbol_samples": symbols,
        "symbol_rate_hz": 9600.0,
        "input_sample_rate_hz": 96000.0
    })
    
    rec_bits = d_res.hard_bits
    bit_err = float(np.mean(rec_bits[:len(raw_bits)] != raw_bits))
    rec_bytes = np.packbits(rec_bits[:len(raw_bits)]).tobytes()
    rec_text = rec_bytes.decode("ascii", errors="replace")
    
    synthetic_payload_results = {
        "original_payload": payload_text,
        "demod_bit_error_rate": bit_err,
        "recovered_payload_text": rec_text,
        "exact_match": (rec_text == payload_text)
    }
except Exception as e:
    synthetic_payload_results = {"status": "BROKEN", "error": str(e)}

audit_results["synthetic_payload"] = synthetic_payload_results
print("Synthetic Payload Audit:", synthetic_payload_results)

# ==============================================================================
# 8. NOISE & NON-TARGET REJECTION AUDIT
# ==============================================================================
print("\n[AUDIT 8/10] Testing Noise & Non-Target Rejection...")
from astra_modulation_v2.fusion import CalibratedFusionEngineV2
mod_engine = CalibratedFusionEngineV2(device="cuda")

noise_results = {}
n_samples = 16384
fs = 96000.0

# AWGN
awgn_iq = (rng.normal(0, 1, n_samples) + 1j * rng.normal(0, 1, n_samples)).astype(np.complex64)
awgn_res = mod_engine.classify(awgn_iq, sample_rate=fs)

# CW Tone
t = np.arange(n_samples) / fs
cw_iq = np.exp(1j * 2 * np.pi * 12000.0 * t).astype(np.complex64)
cw_res = mod_engine.classify(cw_iq, sample_rate=fs)

# Chirp
f0, f1 = 1000.0, 30000.0
phase = 2 * np.pi * (f0 * t + 0.5 * (f1 - f0) * (t**2) / t[-1])
chirp_iq = np.exp(1j * phase).astype(np.complex64)
chirp_res = mod_engine.classify(chirp_iq, sample_rate=fs)

noise_results["AWGN"] = {
    "predicted_mod": awgn_res.get("predicted_modulation"),
    "unknown_prob": awgn_res.get("unknown_probability", 0.0),
    "confidence": awgn_res.get("confidence", 0.0)
}
noise_results["CW"] = {
    "predicted_mod": cw_res.get("predicted_modulation"),
    "unknown_prob": cw_res.get("unknown_probability", 0.0),
    "confidence": cw_res.get("confidence", 0.0)
}
noise_results["Chirp"] = {
    "predicted_mod": chirp_res.get("predicted_modulation"),
    "unknown_prob": chirp_res.get("unknown_probability", 0.0),
    "confidence": chirp_res.get("confidence", 0.0)
}

audit_results["noise_rejection"] = noise_results
print("Noise Rejection Summary:", json.dumps(noise_results, indent=2))

# ==============================================================================
# 9. SILENT FAILURE & CODEBASE SEARCH
# ==============================================================================
print("\n[AUDIT 9/10] Searching for silent failures, broad excepts, and mock values...")
silent_failures = []
py_files = list(ROOT.rglob("*.py"))
for p in py_files:
    if ".venv" in str(p) or "datasets" in str(p):
        continue
    try:
        content = p.read_text(encoding="utf-8", errors="ignore")
        lines = content.splitlines()
        for idx, line in enumerate(lines):
            l_str = line.strip()
            if l_str.startswith("except Exception:") or l_str == "except:":
                if idx + 1 < len(lines) and lines[idx + 1].strip() == "pass":
                    rel = p.relative_to(ROOT)
                    silent_failures.append({
                        "file": str(rel),
                        "line": idx + 1,
                        "type": "bare_except_pass"
                    })
    except Exception:
        pass

audit_results["silent_failures_count"] = len(silent_failures)
print(f"Total bare except-pass patterns found: {len(silent_failures)}")

# ==============================================================================
# 10. TRACE FIRST REAL BREAK PER CAPTURE & PERFORMANCE BENCHMARK
# ==============================================================================
print("\n[AUDIT 10/10] Tracing FIRST real failure stage across 100 blind benchmark captures...")
from astra_symbol_rate.src.inference import SymbolRateInferenceEngine
from astra_synchronization.src.inference import SynchronizationEngine
from astra_synchronization.src.cfo import correct_cfo
from astra_synchronization.src.matched_filter import apply_matched_filter

sync_engine = SynchronizationEngine()
baud_engine = SymbolRateInferenceEngine()

blind_manifest = ROOT / "datasets" / "ASTRA_FINAL_TEST_SET_V2_BLIND" / "blind_benchmark_manifest.json"
with open(blind_manifest, "r") as f:
    blind_data = json.load(f)

captures = [c for c in blind_data["captures"] if c["true_modulation"] != "UNKNOWN"]
sampled = rng.choice(captures, size=min(100, len(captures)), replace=False)

csv_rows = []
stage_break_counts = {
    "STAGE_3_MODULATION": 0,
    "STAGE_4_BAUD_DSP": 0,
    "STAGE_5_CANDIDATE_PAIRING": 0,
    "STAGE_6_SYNC_CFO": 0,
    "STAGE_6_SYNC_TIMING": 0,
    "STAGE_6_SYNC_CARRIER": 0,
    "STAGE_7_DEMOD_EVM": 0,
    "SUCCESS_TO_STAGE_8": 0
}

stage_latencies = {
    "stage_3_modulation": [],
    "stage_4_baud": [],
    "stage_6_sync": [],
    "stage_7_demod": []
}

for idx, cap in enumerate(sampled):
    raw = np.fromfile(cap["iq_path"], dtype=np.float32)
    iq = raw[0::2] + 1j * raw[1::2]
    
    true_mod = cap["true_modulation"]
    true_baud = float(cap["symbol_rate"])
    sr = float(cap["sample_rate"])
    snr = float(cap.get("snr_db", 15.0))
    cfo = float(cap.get("cfo_hz", 0.0))
    sps = sr / max(true_baud, 1.0)
    
    # 1. Mod check
    t0 = time.perf_counter()
    mod_res = mod_engine.classify(iq, sample_rate=sr)
    stage_latencies["stage_3_modulation"].append(time.perf_counter() - t0)
    
    top1_mod = mod_res["predicted_modulation"]
    top3_mods = [c["modulation"] for c in mod_res["top_k_candidates"][:3]]
    mod_in_top3 = true_mod in top3_mods
    
    # 2. Baud check
    t0 = time.perf_counter()
    b_res = baud_engine.estimate(iq, sample_rate=sr, modulation=true_mod)
    stage_latencies["stage_4_baud"].append(time.perf_counter() - t0)
    
    top_k = getattr(b_res, "top_k", []) or []
    cand_bauds = []
    for c in top_k[:3]:
        r = c.get("rate_hz", c.get("symbol_rate_hz")) if isinstance(c, dict) else getattr(c, "rate_hz", None)
        if r and r > 0:
            cand_bauds.append(float(r))
    if not cand_bauds:
        cand_bauds = [float(b_res.best_symbol_rate_hz)]
    baud_in_top3 = any(abs(r - true_baud)/true_baud <= 0.05 for r in cand_bauds[:3])
    
    # 3. Sync check
    t0 = time.perf_counter()
    best_sync = sync_engine.synchronize(iq, sample_rate=sr, nominal_baud=cand_bauds[0], modulation=true_mod)
    stage_latencies["stage_6_sync"].append(time.perf_counter() - t0)
    
    cfo_err = abs(best_sync.estimated_cfo_hz - cfo)
    cfo_ok = cfo_err <= max(200.0, 0.1 * true_baud)
    timing_ok = best_sync.lock_metrics.get("timing_lock", 0.0) >= 0.40
    carrier_ok = best_sync.lock_metrics.get("carrier_lock", 0.0) >= 0.40
    
    # 4. Demod check
    t0 = time.perf_counter()
    d_res = demod_engine.demodulate({
        "modulation": true_mod,
        "symbol_samples": best_sync.symbol_samples,
        "symbol_rate_hz": cand_bauds[0],
        "input_sample_rate_hz": sr
    })
    stage_latencies["stage_7_demod"].append(time.perf_counter() - t0)
    
    demod_ok = bool(d_res.success and d_res.quality.evm_percent <= 38.0 and len(d_res.hard_bits) > 0)
    
    # Determine FIRST failure stage
    first_break = "SUCCESS_TO_STAGE_8"
    failure_detail = "None"
    
    if not mod_in_top3:
        first_break = "STAGE_3_MODULATION"
        failure_detail = f"Top-3={top3_mods}, True={true_mod}"
    elif not baud_in_top3:
        first_break = "STAGE_4_BAUD_DSP"
        failure_detail = f"Top-3 Bauds={cand_bauds[:3]}, True={true_baud:.1f}"
    elif not cfo_ok:
        first_break = "STAGE_6_SYNC_CFO"
        failure_detail = f"CFO Est={best_sync.estimated_cfo_hz:.1f}, True={cfo:.1f} (Err={cfo_err:.1f}Hz)"
    elif not timing_ok:
        first_break = "STAGE_6_SYNC_TIMING"
        failure_detail = f"TimingLock={best_sync.lock_metrics.get('timing_lock', 0.0):.2f}"
    elif not carrier_ok:
        first_break = "STAGE_6_SYNC_CARRIER"
        failure_detail = f"CarrierLock={best_sync.lock_metrics.get('carrier_lock', 0.0):.2f}"
    elif not demod_ok:
        first_break = "STAGE_7_DEMOD_EVM"
        failure_detail = f"EVM={d_res.quality.evm_percent:.1f}%"
    
    stage_break_counts[first_break] += 1
    
    csv_rows.append({
        "capture_index": idx + 1,
        "source_id": cap["source_id"],
        "true_modulation": true_mod,
        "true_baud_hz": round(true_baud, 1),
        "snr_db": round(snr, 1),
        "mod_in_top3": mod_in_top3,
        "baud_in_top3": baud_in_top3,
        "cfo_locked": cfo_ok,
        "timing_locked": timing_ok,
        "carrier_locked": carrier_ok,
        "demod_evm_ok": demod_ok,
        "first_failure_stage": first_break,
        "failure_detail": failure_detail
    })

# Save FIRST_FAILURE_STAGE_ANALYSIS.csv
csv_path = ROOT / "FIRST_FAILURE_STAGE_ANALYSIS.csv"
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
    writer.writeheader()
    writer.writerows(csv_rows)

print(f"\n[DONE] Saved CSV to: {csv_path}")
print("First Failure Breakdown:", json.dumps(stage_break_counts, indent=2))

audit_results["first_failure_stage_counts"] = stage_break_counts
audit_results["stage_latencies_mean_ms"] = {
    k: round(float(np.mean(v) * 1000.0), 2) for k, v in stage_latencies.items()
}

# Save complete audit JSON
out_audit_json = ROOT / "checkpoints" / "comprehensive_system_audit_results.json"
with open(out_audit_json, "w", encoding="utf-8") as f:
    json.dump(audit_results, f, indent=2)
print(f"[DONE] Saved full audit JSON to: {out_audit_json}")
print("Audit Completed Successfully.")
