"""
ASTRA Synthetic Engine 2: Frame / Sync / Header / CRC Generator Demonstration Script.
Generates framed bitstreams from Engine 1 payloads, displays structural telemetry,
saves individual frame artifacts, and creates a concatenated 10-frame stream.
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
logger = logging.getLogger("astra_frame_demo")

# Setup project path
import sys
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from astra_synthetic.payload import PayloadGenerator
from astra_synthetic.framing import (
    FrameGenerator,
    save_frame,
    save_frame_batch,
    save_frame_stream,
)


def run_demonstration(output_dir: Path | str = "output/demo_frames") -> list:
    """Generate and display 5 framed payloads with structural region inspection."""
    payload_cfg = current_dir.parent / "configs" / "payload_config.yaml"
    frame_cfg = current_dir.parent / "configs" / "frame_config.yaml"

    payload_gen = PayloadGenerator(payload_cfg if payload_cfg.exists() else None)
    frame_gen = FrameGenerator(frame_cfg if frame_cfg.exists() else None)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("ASTRA SYNTHETIC ENGINE 2 - DEMONSTRATION FRAME GENERATION")
    print("=" * 80 + "\n")

    # Generate 5 diverse source payloads using Engine 1
    payloads = [
        payload_gen.generate(payload_type="random_bits", bit_length=1024, seed=101, payload_id="payload_000001"),
        payload_gen.generate(payload_type="text", text="ASTRA TEST FRAME DATA", payload_id="payload_000002"),
        payload_gen.generate(payload_type="repeated_pattern", pattern="10110011", bit_length=512, payload_id="payload_000003"),
        payload_gen.generate(payload_type="counter", byte_length=64, start_value=10, payload_id="payload_000004"),
        payload_gen.generate(payload_type="biased_random", bit_length=768, probability_one=0.08, seed=105, payload_id="payload_000005"),
    ]

    frames = []
    for i, prec in enumerate(payloads):
        frame = frame_gen.generate(prec)
        frames.append(frame)
        save_path = save_frame(frame, output_dir=out_path)

        first_128_bits = "".join(str(b) for b in frame.frame_bits[:128])
        sync_end = frame.sync_start + frame.sync_length - 1
        header_end = frame.header_start + frame.header_length - 1
        payload_end = frame.payload_start + frame.payload_length - 1
        crc_end = frame.crc_start + frame.crc_length - 1

        print(f"FRAME {frame.frame_id} (Payload: {frame.payload_id})")
        print(f"  Sequence Number:     {frame.frame_sequence_number}")
        print(f"  Sync Word:           {frame.sync_value} ({frame.sync_length} bits)")
        print(f"  Header Schema:       {frame.header_schema} ({frame.header_length} bits)")
        print(f"  Payload Length:      {frame.payload_length} bits")
        print(f"  CRC Profile:         {frame.crc_type} (Val: {frame.crc_value}, {frame.crc_length} bits)")
        print(f"  Total Frame Length:  {frame.frame_bit_length} bits ({len(frame.frame_bytes)} bytes)")
        print(f"  SHA-256 Hash:        {frame.sha256}")
        print(f"  Structural Layout:")
        print(f"    [{frame.sync_start:4d}:{sync_end:4d}] SYNC    ({frame.sync_length} bits)")
        print(f"    [{frame.header_start:4d}:{header_end:4d}] HEADER  ({frame.header_length} bits, fields: {frame.header_fields})")
        print(f"    [{frame.payload_start:4d}:{payload_end:4d}] PAYLOAD ({frame.payload_length} bits)")
        print(f"    [{frame.crc_start:4d}:{crc_end:4d}] CRC     ({frame.crc_length} bits, scope: {frame.crc_scope})")
        print(f"  First 128 Bits:      {first_128_bits}")
        print(f"  Saved to:            {save_path}")
        print("-" * 80)

    print("\nDemonstration frames saved to:", out_path.resolve())
    return frames


def run_stream_demonstration(output_dir: Path | str = "output/demo_stream") -> Path:
    """Generate 10 frames and concatenate into a single continuous correlation stream."""
    payload_cfg = current_dir.parent / "configs" / "payload_config.yaml"
    frame_cfg = current_dir.parent / "configs" / "frame_config.yaml"

    payload_gen = PayloadGenerator(payload_cfg if payload_cfg.exists() else None)
    frame_gen = FrameGenerator(frame_cfg if frame_cfg.exists() else None)
    frame_gen.reset(seed=500)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("ASTRA SYNTHETIC ENGINE 2 - CONTINUOUS FRAME STREAM GENERATION")
    print("=" * 80 + "\n")

    # Generate 10 diverse payloads
    payloads = [
        payload_gen.generate(payload_type="random_bits", bit_length=256, seed=1000 + i, payload_id=f"payload_{i+1:06d}")
        for i in range(10)
    ]

    # Generate stream with optional 16-bit zero gap between frames
    stream = frame_gen.generate_stream(
        payload_records=payloads,
        inter_frame_gap={"enabled": True, "length_bits": 16, "mode": "zeros"},
    )

    npy_path, meta_path = save_frame_stream(stream, output_dir=out_path, base_name="frame_stream")

    print(f"Generated continuous stream with {len(stream.frame_ids)} frames.")
    print(f"Total stream bit length: {stream.total_bit_length} bits ({len(stream.stream_bytes)} bytes)")
    print(f"SHA-256:                 {stream.sha256}")
    print("\nFrame Indexing Telemetry:")
    for idx, (fid, start, length) in enumerate(zip(stream.frame_ids, stream.frame_start_positions, stream.frame_lengths)):
        sync_pos = stream.sync_positions[idx]
        print(f"  [{idx+1:02d}] {fid}: start={start:5d} bits | length={length:4d} bits | sync_start={sync_pos['start']:5d}")

    print(f"\nStream files saved successfully:")
    print(f"  NPY Array: {npy_path.resolve()}")
    print(f"  Metadata:  {meta_path.resolve()}")

    return npy_path


if __name__ == "__main__":
    run_demonstration()
    run_stream_demonstration()
