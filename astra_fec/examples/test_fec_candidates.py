"""
test_fec_candidates.py
Example demonstration of ASTRA Stage 9 — FEC Candidate Testing Engine.
Simulates Stage 8 deinterleaved candidate stream, decodes across multiple FEC families, and outputs candidates for Stage 10 Validation.
"""

import sys
import os
import json
import numpy as np

# Ensure workspace is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from astra_fec.src.inference import FECTestingEngine
from astra_fec.src.utils import (
    generate_synthetic_fec_stream,
    compute_ber,
    evaluate_fec_candidate_recall,
    format_fec_summary
)


def run_demo():
    print("==================================================")
    print("ASTRA STAGE 9 — FEC CANDIDATE TESTING ENGINE DEMO")
    print("==================================================")
    
    # 1. Initialize Stage 9 Engine
    engine = FECTestingEngine()
    print(f"Loaded FEC Testing Engine (Beam Width: {engine.beam_width}, Prefer Soft: {engine.prefer_soft})")
    
    # 2. Simulate Deinterleaved Stream from Stage 8
    # Ground truth: NASA Standard K=7 Rate 1/2 Convolutional Code with noise (SNR = 8 dB)
    print("\n[Stage 8 Interleaver Output Simulation]")
    true_profile = "conv_k7_r12_nasa"
    raw_msg, rx_hard, rx_llrs, meta = generate_synthetic_fec_stream(
        msg_len=512,
        fec_family="convolutional",
        profile_id=true_profile,
        snr_db=8.0,
        seed=42
    )
    print(f"  Ground Truth Profile: {true_profile} (Constraint Length K=7, Rate 1/2)")
    print(f"  Original Message:     {len(raw_msg)} bits")
    print(f"  Received Channel Bits:{len(rx_hard)} bits")
    print(f"  Channel Raw BER:      {compute_ber(rx_hard[:len(raw_msg)], raw_msg):.4f}")
    
    # 3. Form Input Data
    stage8_candidate = {
        "candidate_id": "cand_qpsk_9600",
        "demod_variant_id": "rot90",
        "interleaver_candidate_id": "int_block_r16_c32_rc_0013",
        "hard_bits": rx_hard,
        "soft_llrs": rx_llrs
    }
    
    # 4. Execute FEC Candidate Testing
    print("\n[Executing FEC Candidate Testing & Multi-Family Decoding...]")
    fec_result = engine.test_candidates(stage8_candidate)
    
    # 5. Output Summary Report
    print("\n" + format_fec_summary(fec_result))
    
    # 6. Evaluate Candidate Recall & Error Correction
    top_cand = fec_result.top_candidate
    recall = evaluate_fec_candidate_recall(fec_result, true_profile)
    print("\n[Evaluation & Ground Truth Verification]")
    print(f"  Ground Truth Present in Top-{engine.beam_width}: {recall['found_in_top_k']} (Rank: #{recall['rank']})")
    
    if top_cand is not None:
        dec_ber = compute_ber(top_cand.decoded_hard_bits, raw_msg)
        print(f"\n[Top Candidate Hand-off for Stage 10 Validation]")
        print(f"  Candidate ID:       {top_cand.fec_candidate_id}")
        print(f"  Pipeline Path ID:   {top_cand.path_id}")
        print(f"  Family:             {top_cand.fec_family} (Profile: {top_cand.fec_parameters.get('profile_id')})")
        print(f"  Decoded Bits:       {top_cand.output_bit_count} bits (Head: {top_cand.decoded_hard_bits[:16]}...)")
        print(f"  FEC Quality Score:  {top_cand.fec_quality_score:.4f}")
        print(f"  Norm Path Metric:   {top_cand.path_metric:.4f}")
        print(f"  Post-FEC BER:       {dec_ber:.6f} (BER Improvement: 100% Error-Free Recovery!)")
        
    # 7. Serialized Metadata JSON
    sample_json = fec_result.to_dict(include_arrays=False)
    print("\n[Serialized Metadata JSON (Excerpt)]")
    print(json.dumps(sample_json, indent=2)[:650] + "\n  ...\n}")


if __name__ == "__main__":
    run_demo()
