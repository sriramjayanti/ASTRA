import os
import sys
import json
import numpy as np

# Add workspace root to sys.path for standalone execution
WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from astra_validation.src.inference import ValidationEngine
from astra_validation.src.utils import generate_synthetic_stream


def main():
    print("=" * 70)
    print("ASTRA STAGE 10 — VALIDATION ENGINE DEMONSTRATION")
    print("=" * 70)

    engine = ValidationEngine()

    # 1. Generate synthetic decoded bitstream (8 frames x 512 bits)
    print("\n[1] Generating synthetic telemetry frames (512 bits/frame, CRC-16 CCITT)...")
    stream = generate_synthetic_stream(
        num_frames=8,
        sync_pattern_hex="EB90",
        version=1,
        start_seq_id=100,
        payload_len=56,
        crc_profile_name="crc16_ccitt_false",
        seed=42,
    )
    print(f"Generated bitstream length: {len(stream)} bits")

    # 2. Construct Stage 9 FECCandidateResult mock
    mock_candidate = {
        "candidate_id": "cand_qpsk_9600_01",
        "demod_variant_id": "rot0",
        "interleaver_candidate_id": "block_16x16",
        "fec_candidate_id": "conv_k7_r12",
        "path_id": "QPSK_9600_rot0_block16x16_convK7R12",
        "fec_family": "convolutional",
        "fec_parameters": {"constraint_length": 7, "generators_octal": [0o171, 0o133]},
        "decoded_hard_bits": stream,
        "decoder_success": True,
        "path_metric": 0.02,
        "decoder_metrics": {"normalized_path_metric": 0.02, "termination_valid": True},
    }

    # 3. Validate
    print("\n[2] Executing Multi-Mechanism Validation Engine...")
    result = engine.validate(mock_candidate, context={"candidate_frame_lengths": [512]})

    # 4. Print Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    print(f"Pipeline Path ID         : {result.pipeline_path_id}")
    print(f"Validation Status        : {result.validation_status.value}")
    print(f"Overall Validation Score : {result.overall_validation_score:.4f}")
    print(f"Independent Evidence Grps: {result.evidence_count}")
    print(f"Strong Evidence Groups   : {result.strong_evidence_count}")
    print(f"Contradictions           : {result.contradiction_count}")
    print(f"Decoded Bitstream Hash   : {result.decoded_bits_hash}")

    print("\nDETAILED CHECKS:")
    for crc in result.crc_results:
        if crc.checked_frames > 0 and crc.pass_rate > 0:
            print(f"  [CRC]      {crc.profile_name:18s} : {crc.passed_frames}/{crc.checked_frames} frames PASS ({crc.pass_rate*100:.1f}%)")
    print(f"  [FEC]      Syndrome/Metric    : Valid={result.syndrome_result.syndrome_valid} (score={result.syndrome_result.normalized_syndrome_score:.2f})")
    print(f"  [Repetition] Frame Period     : {result.frame_repetition_result.candidate_period_bits} bits ({result.frame_repetition_result.repetition_count} frames, score={result.frame_repetition_result.periodicity_score:.2f})")
    for s in result.sync_word_results:
        if s.match_count > 0:
            print(f"  [Sync]     {s.pattern_id:18s} : {s.match_count} hits, periodicity={s.periodicity_score:.2f}")
    for h in result.header_results:
        print(f"  [Header]   {h.field_name:18s} : {h.rule_type} score={h.consistency_score:.2f} ({h.check_state.value})")
    print(f"  [Length]   Declared vs Actual : Declared={result.length_result.declared_length}B, Actual={result.length_result.observed_length}B ({result.length_result.check_state.value})")

    # 5. Export Stage 11 Features
    print("\n" + "=" * 70)
    print("STAGE 11 FEATURE EXPORT (validation_features_v1)")
    print("=" * 70)
    print(json.dumps(result.validation_features, indent=2))


if __name__ == "__main__":
    main()
