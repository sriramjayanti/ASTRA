"""
ASTRA Synthetic Engine 3: FEC Encoding & Coding Ground-Truth Demonstration Script.
Generates FEC-encoded records from Engine 2 frames across all 5 FEC families (None,
Convolutional, Reed-Solomon, Concatenated, LDPC), validates reference decoders,
saves artifacts, and prints the summary comparison report.
"""

from __future__ import annotations

import logging
from pathlib import Path
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("astra_fec_demo")

# Setup project path
import sys
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from astra_synthetic.payload import PayloadGenerator
from astra_synthetic.framing import FrameGenerator
from astra_synthetic.fec import (
    FECGenerator,
    save_fec_record,
    reference_decode,
)


def run_demonstration(output_dir: Path | str = "output/demo_fec") -> list:
    """Generate and display 5 FEC-encoded records from a single FrameRecord."""
    payload_cfg = current_dir.parent / "configs" / "payload_config.yaml"
    frame_cfg = current_dir.parent / "configs" / "frame_config.yaml"
    fec_cfg = current_dir.parent / "configs" / "fec_config.yaml"

    payload_gen = PayloadGenerator(payload_cfg if payload_cfg.exists() else None)
    frame_gen = FrameGenerator(frame_cfg if frame_cfg.exists() else None)
    fec_gen = FECGenerator(fec_cfg if fec_cfg.exists() else None)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 85)
    print("ASTRA SYNTHETIC ENGINE 3 - DEMONSTRATION FEC ENCODING GENERATION")
    print("=" * 85 + "\n")

    # Step 1: Generate a single ground-truth FrameRecord using Engines 1 & 2
    # 1024-bit payload -> 1128-bit frame (32 sync + 56 header + 1024 payload + 16 crc)
    payload = payload_gen.generate(payload_type="random_bits", bit_length=1024, seed=42, payload_id="payload_000001")
    frame = frame_gen.generate(payload, sequence_number=1, frame_id="frame_000001")

    print(f"Source Frame: {frame.frame_id} (Bit Length: {frame.frame_bit_length} bits, SHA-256: {frame.sha256[:16]}...)\n")

    # Step 2: Generate 5 FEC modes
    modes = [
        ("A. None (Uncoded)", "none", "none"),
        ("B. Convolutional K=7 Rate 1/2", "convolutional", "conv_k7_r12"),
        ("C. Reed-Solomon (255, 223)", "reed_solomon", "rs_255_223"),
        ("D. Concatenated (RS + Conv)", "concatenated", "concat_rs255223_conv_k7"),
        ("E. LDPC (128, 64) Rate 1/2", "ldpc", "ldpc_n128_k64_r12"),
    ]

    records = []
    comparison_rows = []

    for title, ftype, prof_name in modes:
        rec = fec_gen.encode(frame, fec_type=ftype, profile_name=prof_name)
        records.append(rec)
        saved_dir = save_fec_record(rec, output_dir=out_path)

        # Reference decode check
        recovered_bits = reference_decode(rec)
        decode_ok = np.array_equal(recovered_bits, frame.frame_bits)

        overhead_pct = ((rec.encoded_bit_length - rec.input_bit_length) / rec.input_bit_length) * 100.0

        comparison_rows.append({
            "type": rec.fec_type.upper(),
            "profile": rec.fec_profile or "N/A",
            "input_bits": rec.input_bit_length,
            "encoded_bits": rec.encoded_bit_length,
            "nominal_rate": f"{rec.nominal_code_rate:.3f}",
            "effective_rate": f"{rec.effective_code_rate:.3f}",
            "overhead": f"+{overhead_pct:.1f}%" if overhead_pct > 0 else "0.0%",
            "decode_verified": "YES" if decode_ok else "NO",
        })

        first_64_enc = "".join(str(b) for b in rec.encoded_bits[:64])

        print(f"{title}")
        print(f"  FEC Record ID:       {rec.fec_record_id}")
        print(f"  Profile:             {rec.fec_profile}")
        print(f"  Input Bits:          {rec.input_bit_length} bits")
        print(f"  Encoded Bits:        {rec.encoded_bit_length} bits")
        print(f"  Nominal Code Rate:   {rec.nominal_code_rate:.4f}")
        print(f"  Effective Code Rate: {rec.effective_code_rate:.4f} (Overhead: {overhead_pct:.1f}%)")
        print(f"  Padding Length:      {rec.padding_length} bits")
        print(f"  Block Count:         {rec.block_count}")
        print(f"  Encoded SHA-256:     {rec.encoded_sha256}")
        print(f"  First 64 Encoded:    {first_64_enc}")
        print(f"  Reference Decode:    {'PASSED (Bit-Exact Recovery)' if decode_ok else 'FAILED'}")
        print(f"  Saved Directory:     {saved_dir}")
        print("-" * 85)

    # Print Comparison Table
    print("\n" + "=" * 85)
    print("FEC FAMILY COMPARISON REPORT")
    print("=" * 85)
    header_fmt = "{:<16} {:<24} {:<10} {:<12} {:<12} {:<10} {:<10}"
    print(header_fmt.format("FEC FAMILY", "PROFILE", "IN BITS", "OUT BITS", "EFF RATE", "OVERHEAD", "DECODED"))
    print("-" * 85)
    for row in comparison_rows:
        print(header_fmt.format(
            row["type"],
            row["profile"],
            row["input_bits"],
            row["encoded_bits"],
            row["effective_rate"],
            row["overhead"],
            row["decode_verified"],
        ))
    print("=" * 85 + "\n")

    return records


if __name__ == "__main__":
    run_demonstration()
