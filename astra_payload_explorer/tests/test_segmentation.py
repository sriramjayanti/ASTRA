import pytest
import numpy as np
from astra_payload_explorer.src.models import ProtocolProfile, FrameCandidate
from astra_payload_explorer.src.segmentation import slice_frames, compute_bit_hash
from astra_payload_explorer.src.utils import generate_synthetic_stream

def test_slice_exact_frames():
    # 3 frames of 64 bits each
    stream, info = generate_synthetic_stream(num_frames=3, frame_length_bits=64, sync_pattern="10101010")
    candidate = FrameCandidate(length_bits=64, alignment_offset_bits=0)
    profile = ProtocolProfile(
        profile_id="test",
        frame_length_bits=64,
        sync_length_bits=8,
        header_length_bits=16,
        crc_length_bits=8,
        sync_pattern="10101010"
    )

    frames = slice_frames(stream, candidate, profile)
    assert len(frames) == 3
    assert all(not f.is_partial for f in frames)
    assert frames[0].frame_index == 0
    assert frames[0].frame_length_bits == 64
    assert frames[0].sync_region.length_bits == 8
    assert frames[0].header_region.length_bits == 16
    assert frames[0].crc_region.length_bits == 8
    assert frames[0].payload_region.length_bits == 32 # 64 - 8 - 16 - 8

def test_partial_frames_handling():
    # Prepend 20 bits of noise and append 15 bits at the end
    lead_noise = np.random.randint(0, 2, size=20, dtype=np.uint8)
    tail_noise = np.random.randint(0, 2, size=15, dtype=np.uint8)
    core_stream, info = generate_synthetic_stream(num_frames=2, frame_length_bits=64, sync_pattern="11001100")
    stream = np.concatenate([lead_noise, core_stream, tail_noise])

    candidate = FrameCandidate(length_bits=64, alignment_offset_bits=20)
    frames = slice_frames(stream, candidate, None)

    # We expect: 1 partial lead frame + 2 full frames + 1 partial tail frame
    assert len(frames) == 4
    assert frames[0].is_partial is True
    assert frames[0].frame_length_bits == 20
    assert frames[1].is_partial is False
    assert frames[1].frame_length_bits == 64
    assert frames[2].is_partial is False
    assert frames[2].frame_length_bits == 64
    assert frames[3].is_partial is True
    assert frames[3].frame_length_bits == 15

def test_compute_bit_hash():
    bits1 = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.uint8)
    bits2 = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.uint8)
    bits3 = np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=np.uint8)

    h1 = compute_bit_hash(bits1)
    h2 = compute_bit_hash(bits2)
    h3 = compute_bit_hash(bits3)

    assert h1 == h2
    assert h1 != h3
    assert len(h1) == 16
