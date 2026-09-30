"""
test_interleaver_candidates.py
Example demonstration of ASTRA Stage 8 — Interleaver Candidate Testing Engine.
Simulates Stage 7 demodulated phase variants, applies candidate hypotheses, and outputs ranked streams for Stage 9 FEC.
"""

import sys
import os
import json
import numpy as np

# Ensure workspace is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from astra_interleaver.src.inference import InterleaverTestingEngine
from astra_interleaver.src.utils import (
    generate_synthetic_interleaved_stream,
    evaluate_candidate_recall,
    format_interleaver_summary
)


def run_demo():
    print("==================================================")
    print("ASTRA STAGE 8 — INTERLEAVER CANDIDATE TESTING DEMO")
    print("==================================================")
    
    # 1. Initialize Engine
    engine = InterleaverTestingEngine()
    print(f"Loaded Interleaver Engine (Beam Width: {engine.beam_width})")
    
    # 2. Simulate Demodulated Bitstream Variant from Stage 7
    # Ground truth: 2048 bits with 16x32 Block Interleaver + periodic frame structure
    print("\n[Stage 7 Demodulation Output Simulation]")
    true_rows, true_cols = 16, 32
    raw_b, raw_l, int_b, int_l, meta = generate_synthetic_interleaved_stream(
        bit_count=2048,
        pattern_type="structured_frame",
        interleaver_family="block",
        interleaver_params={"rows": true_rows, "cols": true_cols, "orientation": "row_to_column"},
        seed=100
    )
    print(f"  Ground Truth Family: {meta['family']} ({true_rows}x{true_cols}, row_to_column)")
    print(f"  Received Bit Count:  {len(int_b)} bits")
    print(f"  Received LLR Count:  {len(int_l)} LLRs")
    
    # 3. Form Demodulation Variant Input
    variant = {
        "candidate_id": "cand_qpsk_9600",
        "variant_id": "rot90",
        "hard_bits": int_b,
        "soft_llrs": int_l
    }
    
    # 4. Execute Interleaver Candidate Testing
    print("\n[Executing Interleaver Hypothesis Testing...]")
    test_result = engine.test_candidates(variant)
    
    # 5. Output Summary Report
    print("\n" + format_interleaver_summary(test_result))
    
    # 6. Evaluate Candidate Recall
    recall = evaluate_candidate_recall(
        test_result,
        ground_truth_family="block",
        ground_truth_params={"rows": true_rows, "cols": true_cols, "orientation": "row_to_column"}
    )
    print("\n[Evaluation & Ground Truth Verification]")
    print(f"  Ground Truth Present in Top-{engine.beam_width}: {recall['found_in_top_k']} (Rank: #{recall['rank']})")
    
    # 7. Check Hard/Soft Output for Top Candidate
    top_cand = test_result.top_candidate
    if top_cand is not None:
        print(f"\n[Top Candidate Hand-off for Stage 9 FEC]")
        print(f"  Candidate ID:       {top_cand.interleaver_candidate_id}")
        print(f"  Family:             {top_cand.interleaver_family}")
        print(f"  Deinterleaved Bits: {len(top_cand.deinterleaved_hard_bits)} bits (Head: {top_cand.deinterleaved_hard_bits[:16]}...)")
        print(f"  Deinterleaved LLRs: {len(top_cand.deinterleaved_soft_llrs)} LLRs (Head: {np.round(top_cand.deinterleaved_soft_llrs[:4], 2)}...)")
        print(f"  Structural Score:   {top_cand.overall_interleaver_score:.4f}")
        print(f"  Periodicity Score:  {top_cand.periodicity_score:.4f}")
        print(f"  Autocorr Peak:      {top_cand.autocorrelation_score:.4f}")
        
    # 8. Sample JSON Serialization
    sample_json = test_result.to_dict(include_arrays=False)
    print("\n[Serialized Metadata JSON (Excerpt)]")
    print(json.dumps(sample_json, indent=2)[:600] + "\n  ...\n}")


if __name__ == "__main__":
    run_demo()
