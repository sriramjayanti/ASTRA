"""
analyze_bitstream.py
Example demonstration of ASTRA Stage 12 Bitstream Intelligence Engine.
Analyzes recovered bitstream candidate, discovers framing structure, and exports Stage 13 features.
"""

import sys
import os
import json
import numpy as np

# Ensure parent directory is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from astra_bitstream_intelligence.src.inference import BitstreamIntelligenceEngine
from astra_bitstream_intelligence.src.utils import generate_synthetic_framed_bitstream


def main():
    print("=" * 70)
    print("ASTRA STAGE 12: BITSTREAM INTELLIGENCE ENGINE DEMONSTRATION")
    print("=" * 70)

    # 1. Initialize engine
    engine = BitstreamIntelligenceEngine()

    # 2. Generate a realistic synthetic post-FEC recovered bitstream
    # 64 frames of 512 bits = 32,768 bits
    # Structure: [EB90 Sync (16 bits)][Header (32 bits)][Random Payload (464 bits)]
    print("\n1. Generating Synthetic Recovered Bitstream (from Stage 11)...")
    bits, truth = generate_synthetic_framed_bitstream(
        frame_length=512,
        num_frames=64,
        sync_word_hex="EB90",
        header_length=32,
        seed=42
    )

    print(f"   Total Bits Recovered: {len(bits):,}")
    print(f"   Ground Truth Frame Length: {truth['frame_length']} bits")
    print(f"   Ground Truth Sync Word: 0x{truth['sync_word_hex']}")

    # 3. Analyze bitstream structure
    print("\n2. Executing Stage 12 Structural Analysis...")
    result = engine.analyze(
        bits,
        context={
            "pipeline_path_id": "stage11_top_candidate_pipeline_01",
            "known_sync_words": ["EB90", "1ACFFC1D"]
        }
    )

    # 4. Display findings
    print("\n3. Analysis Findings:")
    print(f"   Status:                   {result.status.value}")
    print(f"   Global Binary Entropy:    {result.binary_entropy_global:.4f}")
    print(f"   Bit Balance:              Zeros={result.bit_balance.zero_fraction:.3f}, Ones={result.bit_balance.one_fraction:.3f}")
    print(f"   Byte Alignment (Offset):  {result.byte_alignment.best_offset}")

    print("\n   Top Autocorrelation Peaks:")
    for i, p in enumerate(result.autocorrelation_peaks[:3]):
        print(f"     [{i+1}] Lag: {p.lag} bits | Correlation: {p.correlation:.4f} | Prominence: {p.prominence:.4f}")

    print("\n   Candidate Frame Lengths (Top-K):")
    for i, f in enumerate(result.frame_length_candidates[:3]):
        print(f"     [{i+1}] Length: {f.period_bits} bits | Score: {f.score:.4f} | Supported by: {f.support_sources}")

    print("\n   Sync Word Matches:")
    for s in result.sync_results:
        print(f"     Pattern: {s.pattern_id} | Matches: {s.match_count} | Mean Spacing: {s.mean_spacing:.1f} bits | Confidence: {s.confidence:.4f}")

    print("\n   Candidate Structural Regions within 512-bit Frame:")
    for r in result.structural_regions:
        print(f"     Bits [{r.start_bit:3d}..{r.end_bit:3d}] -> Type: {r.region_type:<10s} | Stability: {r.mean_stability:.3f} | Entropy: {r.mean_entropy:.3f}")

    print("\n4. Stage 13 Handoff Feature Schema:")
    print(f"   Feature Schema Version:   bitstream_features_v1")
    print(f"   Global Scalar Features:   {len(result.global_features)} features")
    if result.sequence_feature_map is not None:
        print(f"   Sequence Map Tensor:      Shape {result.sequence_feature_map.shape} (N_bits x 4 channels)")
        print("   Channels: [0: Bipolar Bits, 1: Soft Confidence, 2: Window Entropy, 3: Positional Stability/Boundary]")

    print("\n" + "=" * 70)
    print("STAGE 12 BITSTREAM INTELLIGENCE ENGINE ANALYSIS COMPLETED SUCCESSFULLY.")
    print("=" * 70)


if __name__ == "__main__":
    main()
