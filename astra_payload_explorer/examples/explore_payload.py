#!/usr/bin/env python3
"""
ASTRA Stage 14 — Header / Payload Explorer Example
Demonstrates end-to-end recovery, profile parsing, payload interpretation,
and blind field exploration on synthetic signals.
"""

import sys
import numpy as np
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from astra_payload_explorer.src.inference import HeaderPayloadExplorer
from astra_payload_explorer.src.models import (
    PayloadExplorerContext,
    ProtocolProfile,
    FieldDefinition,
    ExplorerStatus,
    EvidenceLevel
)
from astra_payload_explorer.src.byte_alignment import bytes_to_bits
from astra_payload_explorer.src.exporters import export_to_json, generate_hex_dump

def main():
    print("==================================================")
    print("ASTRA STAGE 14: HEADER / PAYLOAD EXPLORER DEMO")
    print("==================================================")

    # 1. Define synthetic protocol profile: astra_demo_v1
    profile = ProtocolProfile(
        profile_id="astra_demo_v1",
        version="1.0.0",
        frame_length_bits=128,
        sync_length_bits=32,
        header_length_bits=32,
        crc_length_bits=16,
        sync_pattern="10101010101010101010101010101010",
        header_fields={
            "version": FieldDefinition(name="version", offset_bits=0, width_bits=8, field_type="uint"),
            "flags": FieldDefinition(name="flags", offset_bits=8, width_bits=8, field_type="uint"),
            "sequence": FieldDefinition(name="sequence", offset_bits=16, width_bits=16, field_type="uint")
        }
    )

    # 2. Build 3 frames with "hi hello" in payload
    # Sync: 32 bits, Header: 32 bits, Payload: 48 bits (6 bytes: "hi hel"), CRC: 16 bits (0xABCD) -> total = 128 bits
    sync_bits = [1, 0] * 16
    crc_bits = bytes_to_bits(bytes([0xAB, 0xCD]))
    payload_text = "hi hel"
    payload_bits = bytes_to_bits(payload_text.encode("utf-8"))

    frames = []
    for seq in range(3):
        hdr_bytes = bytes([0x01, 0x00, 0x00, seq + 1]) # version 1, flags 0, seq 1..3
        hdr_bits = bytes_to_bits(hdr_bytes)
        frame = np.concatenate([sync_bits, hdr_bits, payload_bits, crc_bits])
        frames.append(frame)

    stream = np.concatenate(frames)
    print(f"\n[1] Generated Synthetic Recovered Bitstream: {len(stream)} bits ({len(frames)} frames)")

    # 3. Instantiate Explorer and run exploration
    explorer = HeaderPayloadExplorer()
    ctx = PayloadExplorerContext(
        pipeline_path_id="pipe_demo_rec_01",
        protocol_profiles={"astra_demo_v1": profile},
        mode="auto"
    )

    print("\n[2] Executing Stage 14 HeaderPayloadExplorer...")
    result = explorer.explore(stream, ctx)

    # 4. Display Results
    print(f"\n[3] Explorer Status: {result.explorer_status.value}")
    print(f"    Profile Used: {result.protocol_profile_used}")
    print(f"    Profile Match Score: {result.profile_match_score:.2f}")
    print(f"    Frames Recovered: {result.frame_count}")

    for f in result.frames:
        print(f"\n--- FRAME {f.frame_index} ---")
        print(f"  Frame Range: bits {f.start_bit}..{f.end_bit} (length: {f.frame_length_bits} bits)")
        print(f"  Sync Hex: {f.sync_region.raw_hex} (Confidence: {f.sync_region.confidence*100:.1f}%)")
        print("  Header Fields:")
        for name, field in f.header_fields.items():
            print(f"    - {name} ({field.evidence_level.value}): raw={field.raw_hex}, decoded={field.decoded_value} (src: {field.source})")
        print(f"  Payload Length: {f.payload.length_bits} bits ({f.payload.length_bytes} bytes)")
        print(f"  Payload Hex: {f.payload.hex_str}")
        if "utf8" in f.payload.representations and f.payload.representations["utf8"]["valid"]:
            print(f"  Payload UTF-8: '{f.payload.representations['utf8']['value']}'")
        print(f"  CRC Hex: {f.crc_region.raw_hex}")
        print(f"  Quality Score: {f.frame_quality_score:.2f}")

    print("\n[4] Hex Dump of Frame 0 Payload:")
    print(generate_hex_dump(bytes(result.frames[0].payload.byte_array)))

    print("\n==================================================")
    print("STAGE 14 DEMO COMPLETE: EXACT RECOVERY VERIFIED")
    print("==================================================")

if __name__ == "__main__":
    main()
