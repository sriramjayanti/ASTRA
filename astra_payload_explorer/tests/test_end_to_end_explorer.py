import pytest
import numpy as np
from astra_payload_explorer.src.inference import HeaderPayloadExplorer
from astra_payload_explorer.src.models import (
    PayloadExplorerContext,
    ProtocolProfile,
    FieldDefinition,
    ExplorerStatus,
    EvidenceLevel
)
from astra_payload_explorer.src.utils import generate_synthetic_stream, build_test_frame
from astra_payload_explorer.src.byte_alignment import bytes_to_bits, bits_to_bytes

def test_perfect_frame_hi_hello_profile_mode():
    # Construct a synthetic frame stream with "hi hello" payload
    # Profile: frame_length=128, sync=32 ("10101010101010101010101010101010"), header=32 (version: 8-bit=1, flags: 8-bit=0, seq: 16-bit=100), payload=64 ("hi hello"), crc=0
    fields_def = {
        "version": FieldDefinition(name="version", offset_bits=0, width_bits=8, field_type="uint"),
        "flags": FieldDefinition(name="flags", offset_bits=8, width_bits=8, field_type="uint"),
        "sequence": FieldDefinition(name="sequence", offset_bits=16, width_bits=16, field_type="uint")
    }
    profile = ProtocolProfile(
        profile_id="astra_demo_hi_hello",
        frame_length_bits=128,
        sync_length_bits=32,
        header_length_bits=32,
        crc_length_bits=0,
        sync_pattern="10101010101010101010101010101010",
        header_fields=fields_def
    )

    sync_bits = [1, 0] * 16
    # header: version=1 (0x01), flags=0 (0x00), seq=100 (0x0064)
    hdr_bytes = bytes([0x01, 0x00, 0x00, 0x64])
    hdr_bits = bytes_to_bits(hdr_bytes)
    payload_bytes = b"hi hello"
    payload_bits = bytes_to_bits(payload_bytes)

    frame_bits = np.concatenate([sync_bits, hdr_bits, payload_bits])
    assert len(frame_bits) == 128

    explorer = HeaderPayloadExplorer()
    ctx = PayloadExplorerContext(
        protocol_profiles={"astra_demo_hi_hello": profile},
        known_profile_id="astra_demo_hi_hello",
        mode="profile_aware"
    )

    result = explorer.explore(frame_bits, ctx)

    assert result.explorer_status in [ExplorerStatus.PROFILE_PARSED, ExplorerStatus.STRUCTURE_PARSED]
    assert result.frame_count == 1
    frame0 = result.frames[0]

    # Verify header fields
    assert frame0.header_fields["version"].decoded_value == 1
    assert frame0.header_fields["sequence"].decoded_value == 100
    assert frame0.header_fields["sequence"].evidence_level == EvidenceLevel.KNOWN

    # Verify payload representations
    assert frame0.payload.hex_str == "68692068656c6c6f"
    assert frame0.payload.representations["utf8"]["value"] == "hi hello"
    assert frame0.payload.representations["utf8"]["valid"] is True
    assert frame0.payload.representations["ascii"]["value"] == "hi hello"

def test_blind_exploration_unknown_profile():
    # Sequence of 4 frames with unknown structure:
    # 16-bit sync "1100110011001100", 16-bit header (8-bit constant 0xFF, 8-bit counter 0, 1, 2, 3), 32-bit binary payload
    frames_bits = []
    sync_bits = [1, 1, 0, 0] * 4
    for i in range(4):
        hdr_bytes = bytes([0xFF, i])
        hdr_bits = bytes_to_bits(hdr_bytes)
        p_bytes = bytes([0x10 + i, 0x20 + i, 0x30 + i, 0x40 + i])
        p_bits = bytes_to_bits(p_bytes)
        frame = np.concatenate([sync_bits, hdr_bits, p_bits])
        frames_bits.append(frame)

    stream = np.concatenate(frames_bits) # 4 * 64 = 256 bits

    explorer = HeaderPayloadExplorer()
    # No known profile, mode=blind
    ctx = PayloadExplorerContext(
        protocol_profiles={},
        mode="blind_exploration"
    )

    result = explorer.explore(stream, ctx)
    assert result.explorer_status in [ExplorerStatus.BLIND_EXPLORED, ExplorerStatus.STRUCTURE_PARSED]
    assert result.frame_count >= 3
    # Check that candidate interpretations/blind discovered fields were produced without pretending to understand semantics
    assert len(result.blind_discovered_fields) > 0 or len(result.candidate_interpretations) > 0
    if result.blind_discovered_fields:
        patterns = [c.pattern for c in result.blind_discovered_fields]
        assert any(p in ["incrementing", "constant", "unknown"] for p in patterns)

def test_wrong_profile_fallback():
    # Provide a profile that does NOT match the stream sync / length
    mismatched_profile = ProtocolProfile(
        profile_id="wrong_prof",
        frame_length_bits=500,
        sync_length_bits=32,
        sync_pattern="00000000000000000000000000000000"
    )
    # Stream is 128-bit frames with "101010..." sync
    stream, _ = generate_synthetic_stream(num_frames=2, frame_length_bits=128, sync_pattern="10101010"*4)

    explorer = HeaderPayloadExplorer()
    ctx = PayloadExplorerContext(
        protocol_profiles={"wrong_prof": mismatched_profile},
        mode="auto"
    )

    result = explorer.explore(stream, ctx)
    # Auto mode should reject the weak profile match and fallback gracefully
    assert result.protocol_profile_used is None or result.profile_match_score < 0.5
    assert result.explorer_status in [ExplorerStatus.UNKNOWN_PROTOCOL, ExplorerStatus.BLIND_EXPLORED, ExplorerStatus.STRUCTURE_PARSED]
