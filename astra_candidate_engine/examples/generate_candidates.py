"""
generate_candidates.py
Demonstration script for ASTRA Stage 5 — Candidate / Hypothesis Engine.
Generates and ranks a 3x3 Cartesian grid of receiver hypotheses with explainability.
"""

import sys
import os
import json

# Ensure workspace root is in sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
engine_root = os.path.dirname(script_dir)
workspace_root = os.path.dirname(engine_root)

for p in [workspace_root, engine_root]:
    if p not in sys.path:
        sys.path.insert(0, p)

from astra_candidate_engine.src.inference import CandidateHypothesisEngine
from astra_candidate_engine.src.utils import (
    create_mock_fusion_prediction,
    create_mock_symbol_rate_prediction,
    create_mock_rf_support,
    create_mock_constellation_evidence,
)


def run_demo():
    print("=" * 80)
    print("ASTRA STAGE 5 — CANDIDATE / HYPOTHESIS ENGINE DEMONSTRATION")
    print("=" * 80)

    engine = CandidateHypothesisEngine()

    # 1. Structured Inputs (Milestone 1 Acceptance Criteria)
    fusion_input = create_mock_fusion_prediction([
        {"class": "QPSK", "probability": 0.80},
        {"class": "8PSK", "probability": 0.12},
        {"class": "16-QAM", "probability": 0.05}
    ])
    
    symbol_rate_input = create_mock_symbol_rate_prediction([
        {"symbol_rate_hz": 9600.0, "score": 0.85, "samples_per_symbol": 20.0},
        {"symbol_rate_hz": 4800.0, "score": 0.10, "samples_per_symbol": 40.0},
        {"symbol_rate_hz": 19200.0, "score": 0.05, "samples_per_symbol": 10.0}
    ])

    sample_rate_hz = 192000.0
    signal_id = "SIG_DEMO_001"

    print("\n[1] INPUT EVIDENCE:")
    print("   Modulation Candidates (Fusion):")
    for m in fusion_input["top_k"]:
        print(f"     - {m['class']:<10} P = {m['probability']:.2f}")
    
    print("   Symbol-Rate Candidates (Estimator):")
    for r in symbol_rate_input["top_k"]:
        print(f"     - {r['symbol_rate_hz']:<10g} Bd, Score = {r['score']:.2f}")
    print(f"   Sampling Rate: {sample_rate_hz:g} Hz")

    # 2. Baseline Candidate Generation (Modulation x Baud Grid)
    print("\n[2] GENERATING BASELINE CANDIDATE SET (No Auxiliary Models):")
    baseline_cand_set = engine.generate(
        modulation_prediction=fusion_input,
        symbol_rate_prediction=symbol_rate_input,
        sample_rate_hz=sample_rate_hz,
        signal_id=signal_id
    )

    print(f"\n{engine.print_summary(baseline_cand_set)}")
    print(f"\nTotal Generated: {baseline_cand_set.all_generated_count}, Valid: {baseline_cand_set.valid_candidate_count}, Beam Width: {len(baseline_cand_set.beam_candidates)}")

    # 3. Multi-Evidence Candidate Generation (with RF & Constellation)
    print("\n" + "-" * 80)
    print("[3] GENERATING CANDIDATE SET WITH AUXILIARY SUPPORT (RF + Constellation):")
    rf_support = create_mock_rf_support({"PSK": 0.94, "QAM": 0.05, "FSK": 0.01})
    constellation_support = create_mock_constellation_evidence({
        "QPSK": 0.88,
        "8PSK": 0.35,
        "16-QAM": 0.10
    })

    print("   RF Family Support: PSK=0.94, QAM=0.05, FSK=0.01")
    print("   Constellation Support: QPSK=0.88, 8PSK=0.35, 16-QAM=0.10")

    supported_cand_set = engine.generate(
        modulation_prediction=fusion_input,
        symbol_rate_prediction=symbol_rate_input,
        sample_rate_hz=sample_rate_hz,
        signal_id=signal_id,
        rf_support=rf_support,
        constellation_evidence=constellation_support,
        constellation_stage="pre_sync"
    )

    print(f"\n{engine.print_summary(supported_cand_set)}")

    # 4. Explainability Breakdown for Top Hypothesis
    top_cand = supported_cand_set.beam_candidates[0]
    print("\n" + "-" * 80)
    print(f"[4] AUDITABLE EXPLAINABILITY BREAKDOWN FOR TOP-1 CANDIDATE ({top_cand.candidate_id}):")
    explanation = engine.explain(top_cand)
    print(json.dumps(explanation, indent=2))

    # 5. Synchronization Profile Hints for Next Stage (Stage 6)
    print("\n" + "-" * 80)
    print(f"[5] PREPARED SYNC ROUTE HINTS FOR STAGE 6 SYNCHRONIZATION ENGINE:")
    print(f"   Candidate: {top_cand.modulation} @ {top_cand.symbol_rate_hz:g} Bd")
    print(f"   Carrier Recovery : {top_cand.sync_hints.get('carrier_recovery')}")
    print(f"   Matched Filter   : {top_cand.sync_hints.get('matched_filter_family')}")
    print(f"   Timing Recovery  : {top_cand.sync_hints.get('timing_recovery_methods')}")
    print(f"   Costas Loop Order: {top_cand.sync_hints.get('costas_order')}")
    print(f"   Demodulator Type : {top_cand.demod_hints.get('demodulator_type')}")
    print("=" * 80)


if __name__ == "__main__":
    run_demo()
