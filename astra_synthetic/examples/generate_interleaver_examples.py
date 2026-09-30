"""
ASTRA Synthetic Engine 4: Interleaver / Bit-Reordering Demonstration Script.
Consumes FECRecord from Engine 3, generates interleaved records across all 5 families
(None, Block, Convolutional, Diagonal, Pseudo-Random), validates reference deinterleaving,
saves artifacts to disk, and prints the summary comparison report.
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
logger = logging.getLogger("astra_interleaver_demo")

# Setup project path
import sys
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from astra_synthetic.payload import PayloadGenerator
from astra_synthetic.framing import FrameGenerator
from astra_synthetic.fec import FECGenerator
from astra_synthetic.interleaving import (
    InterleaverGenerator,
    save_interleaver_record,
    reference_deinterleave,
)


def run_demonstration(output_dir: Path | str = "output/demo_interleaver") -> list:
    """Generate and display 5 Interleaver records from a single source FECRecord."""
    payload_cfg = current_dir.parent / "configs" / "payload_config.yaml"
    frame_cfg = current_dir.parent / "configs" / "frame_config.yaml"
    fec_cfg = current_dir.parent / "configs" / "fec_config.yaml"
    int_cfg = current_dir.parent / "configs" / "interleaver_config.yaml"

    payload_gen = PayloadGenerator(payload_cfg if payload_cfg.exists() else None)
    frame_gen = FrameGenerator(frame_cfg if frame_cfg.exists() else None)
    fec_gen = FECGenerator(fec_cfg if fec_cfg.exists() else None)
    int_gen = InterleaverGenerator(int_cfg if int_cfg.exists() else None)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 90)
    print("ASTRA SYNTHETIC ENGINE 4 - DEMONSTRATION INTERLEAVER GENERATION")
    print("=" * 90 + "\n")

    # Step 1: Upstream pipeline: Payload -> Frame -> FEC (Convolutional K=7 Rate 1/2)
    payload = payload_gen.generate(payload_type="random_bits", bit_length=1024, seed=42, payload_id="payload_000001")
    frame = frame_gen.generate(payload, sequence_number=1, frame_id="frame_000001")
    fec_rec = fec_gen.encode(frame, profile_name="conv_k7_r12", fec_record_id="fec_000001")

    print(f"Source FEC: {fec_rec.fec_record_id} ({fec_rec.fec_type} / {fec_rec.fec_profile})")
    print(f"Input Encoded Bits: {fec_rec.encoded_bit_length} bits, SHA-256: {fec_rec.encoded_sha256[:16]}...\n")

    # Step 2: Generate 5 Interleaver modes
    modes = [
        ("A. None (No Interleaving)", "none", "none"),
        ("B. Block Interleaver (16x32)", "block", "block_16x32"),
        ("C. Convolutional Interleaver (B=4, M=2)", "convolutional", "conv_b4_d2"),
        ("D. Diagonal Interleaver (8x8)", "diagonal", "diag_8x8"),
        ("E. Pseudo-Random Interleaver (256-bit block)", "pseudo_random", "pr_256"),
    ]

    records = []
    comparison_rows = []

    for title, itype, prof_name in modes:
        rec = int_gen.interleave(fec_rec, interleaver_type=itype, profile_name=prof_name)
        records.append(rec)
        saved_dir = save_interleaver_record(rec, output_dir=out_path)

        # Reference deinterleave check
        recovered_bits = reference_deinterleave(rec)
        deinterleave_ok = np.array_equal(recovered_bits, fec_rec.encoded_bits)

        comparison_rows.append({
            "type": rec.interleaver_type.upper(),
            "profile": rec.profile_name or "N/A",
            "input_bits": rec.input_bit_length,
            "output_bits": rec.output_bit_length,
            "padding": rec.padding_length,
            "reversible": "YES" if deinterleave_ok else "NO",
            "test_passed": "YES" if deinterleave_ok else "NO",
        })

        first_64_in = "".join(str(b) for b in rec.input_bits[:64])
        first_64_out = "".join(str(b) for b in rec.interleaved_bits[:64])

        print(f"{title}")
        print(f"  Record ID:           {rec.interleaver_record_id}")
        print(f"  FEC Source:          {rec.fec_record_id} (Frame: {rec.frame_id})")
        print(f"  Profile:             {rec.profile_name}")
        print(f"  Input Bit Length:    {rec.input_bit_length} bits")
        print(f"  Output Bit Length:   {rec.output_bit_length} bits")
        print(f"  Padding / Flush:     {rec.padding_length} bits")
        print(f"  Parameters:          {rec.parameters}")
        print(f"  First 64 Input Bits:  {first_64_in}")
        print(f"  First 64 Output Bits: {first_64_out}")
        print(f"  Output SHA-256:      {rec.output_sha256}")
        print(f"  Ref Deinterleave:    {'PASSED (Bit-Exact Recovery)' if deinterleave_ok else 'FAILED'}")
        print(f"  Saved Directory:     {saved_dir}")
        print("-" * 90)

    # Print Comparison Table
    print("\n" + "=" * 90)
    print("INTERLEAVER FAMILY COMPARISON REPORT")
    print("=" * 90)
    header_fmt = "{:<16} {:<18} {:<12} {:<14} {:<10} {:<12} {:<12}"
    print(header_fmt.format("INTERLEAVER", "PROFILE", "INPUT BITS", "OUTPUT BITS", "PADDING", "REVERSIBLE", "TEST PASSED"))
    print("-" * 90)
    for row in comparison_rows:
        print(header_fmt.format(
            row["type"],
            row["profile"],
            row["input_bits"],
            row["output_bits"],
            row["padding"],
            row["reversible"],
            row["test_passed"],
        ))
    print("=" * 90 + "\n")

    return records


if __name__ == "__main__":
    run_demonstration()
