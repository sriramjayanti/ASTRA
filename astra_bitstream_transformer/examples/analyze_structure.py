"""
analyze_structure.py
Example demonstration of ASTRA Stage 13 1D CNN + Transformer Bitstream Structure Model.
Receives recovered bitstream and Stage 12 features, predicting and parsing higher-level bitstream structure.
"""

import sys
import os
import json
import numpy as np

# Ensure parent directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from astra_bitstream_intelligence.src.inference import BitstreamIntelligenceEngine
from astra_bitstream_transformer.src.inference import BitstreamStructureModel
from astra_bitstream_transformer.src.utils import generate_annotated_synthetic_stream


def main():
    print("=" * 75)
    print("ASTRA STAGE 13: 1D CNN + TRANSFORMER BITSTREAM STRUCTURE MODEL")
    print("=" * 75)

    # 1. Initialize Stage 12 and Stage 13 engines
    stage12_engine = BitstreamIntelligenceEngine()
    stage13_model = BitstreamStructureModel()

    # 2. Generate a structured post-FEC candidate bitstream
    # 8 frames of 512 bits = 4096 bits
    # Structure: [Sync (16 bits)][Header (32 bits)][Payload (448 bits)][CRC (16 bits)]
    print("\n1. Generating Synthetic Recovered Candidate Bitstream (Post-FEC)...")
    raw_bits, channels, labels, truth = generate_annotated_synthetic_stream(
        frame_length=512,
        num_frames=8,
        sync_word_hex="EB90",
        header_length=32,
        crc_length=16,
        prefix_noise_bits=16,
        ber=0.001,
        seed=123
    )

    print(f"   Total Bits:                {len(raw_bits):,}")
    print(f"   Ground Truth Frame Length: {truth['frame_length']} bits")
    print(f"   Ground Truth Sync Word:    0x{truth['sync_word_hex']} ({truth['sync_length']} bits)")
    print(f"   Ground Truth Header:       {truth['header_length']} bits")
    print(f"   Ground Truth CRC:          {truth['crc_length']} bits")

    # 3. Stage 12 Statistical Feature Extraction
    print("\n2. Running Stage 12 Bitstream Intelligence Engine...")
    s12_res = stage12_engine.analyze(
        raw_bits,
        context={"pipeline_path_id": "candidate_pipeline_best"}
    )
    print(f"   Stage 12 Status:           {s12_res.status.value}")
    print(f"   Global Binary Entropy:     {s12_res.binary_entropy_global:.4f}")
    if s12_res.frame_length_candidates:
        print(f"   Top Frame-Length Estimate: {s12_res.frame_length_candidates[0].period_bits} bits")

    # 4. Stage 13 Neural Sequence Modeling
    print("\n3. Running Stage 13 1D CNN + Transformer Structure Predictor...")
    prediction = stage13_model.predict(
        raw_bits,
        stage12_result=s12_res,
        context={"pipeline_path_id": "candidate_pipeline_best"}
    )

    print(f"   Model Version:             {prediction.model_version}")
    print(f"   Input Schema:              {prediction.feature_schema_version}")
    print(f"   Overall Model Confidence:  {prediction.model_confidence:.4f}")
    print(f"   Mean Bit Uncertainty:      {prediction.mean_uncertainty:.4f}")

    print("\n4. Parsed Contiguous Structural Regions:")
    for i, r in enumerate(prediction.regions[:12]):
        print(f"   [{i+1:2d}] Bits [{r.start_bit:4d}..{r.end_bit:4d}] ({r.bit_length:3d} bits) -> {r.label:<8s} | Conf: {r.mean_probability:.3f}")

    if len(prediction.regions) > 12:
        print(f"   ... ({len(prediction.regions) - 12} more regions)")

    if prediction.frame_level_predictions:
        print("\n5. Frame-Level Multi-Label Telemetry:")
        for k, v in prediction.frame_level_predictions.items():
            print(f"   - {k:<20s}: {v:.4f}")

    print("\n" + "=" * 75)
    print("STAGE 13 BITSTREAM STRUCTURE MODEL DEMONSTRATION COMPLETED SUCCESSFULLY.")
    print("=" * 75)


if __name__ == "__main__":
    main()
