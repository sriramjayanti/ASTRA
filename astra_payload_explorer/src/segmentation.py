"""
segmentation.py
Physical bitstream segmentation into structured FrameRecord instances.
"""

from typing import List, Dict, Optional, Tuple, Any
import hashlib
import numpy as np

from .models import (
    FrameRecord,
    RegionInfo,
    SyncRegionInfo,
    CRCRegionInfo,
    PayloadViews,
    FrameCandidate,
    ProtocolProfile
)
from .byte_alignment import bits_to_hex_str, bits_to_bytes
from .payload_decoders import decode_payload_views


def compute_bit_hash(bits: np.ndarray) -> str:
    """Compute 16-character SHA-256 hash of bit sequence."""
    if len(bits) == 0:
        return ""
    return hashlib.sha256(bits.astype(np.uint8).tobytes()).hexdigest()[:16]


compute_bits_hash = compute_bit_hash  # Alias


def slice_frames(
    stream_bits: np.ndarray,
    candidate: FrameCandidate,
    profile: Optional[ProtocolProfile] = None,
    allow_partial: bool = True
) -> List[FrameRecord]:
    """
    Slice 1D bitstream into structured FrameRecord instances using candidate length, offset, and profile.
    """
    n_bits = len(stream_bits)
    frame_len = candidate.length_bits
    offset = candidate.alignment_offset_bits
    frames: List[FrameRecord] = []
    frame_idx = 0

    sync_len = getattr(profile, "sync_length_bits", 32) if profile else 32
    header_len = getattr(profile, "header_length_bits", 32) if profile else 32
    crc_len = getattr(profile, "crc_length_bits", 16) if profile else 0

    # 1. Partial lead frame
    if offset > 0 and allow_partial:
        p_bits = stream_bits[:offset]
        p_views = decode_payload_views(p_bits)
        frames.append(FrameRecord(
            frame_index=frame_idx,
            start_bit=0,
            end_bit=offset,
            frame_length_bits=len(p_bits),
            is_partial=True,
            frame_quality_score=0.5,
            frame_hash=compute_bit_hash(p_bits),
            payload=p_views,
            payload_hash=compute_bit_hash(p_bits)
        ))
        frame_idx += 1

    # 2. Main frames
    curr = offset
    while curr < n_bits:
        end = min(curr + frame_len, n_bits)
        f_bits = stream_bits[curr:end]
        is_partial = len(f_bits) < frame_len

        if is_partial and not allow_partial:
            break

        f_len = len(f_bits)
        # Determine internal regions
        s_len = min(sync_len, f_len)
        h_len = min(header_len, max(0, f_len - s_len))
        c_len = min(crc_len, max(0, f_len - s_len - h_len)) if crc_len > 0 else 0
        p_len = max(0, f_len - s_len - h_len - c_len)

        sync_bits = f_bits[:s_len] if s_len > 0 else np.array([], dtype=np.uint8)
        header_bits = f_bits[s_len:s_len + h_len] if h_len > 0 else np.array([], dtype=np.uint8)
        payload_bits = f_bits[s_len + h_len:s_len + h_len + p_len] if p_len > 0 else np.array([], dtype=np.uint8)
        crc_bits = f_bits[s_len + h_len + p_len:s_len + h_len + p_len + c_len] if c_len > 0 else np.array([], dtype=np.uint8)

        sync_info = SyncRegionInfo(
            raw_bits="".join(str(b) for b in sync_bits),
            raw_hex=bits_to_hex_str(sync_bits),
            start_bit=curr,
            end_bit=curr + s_len,
            length_bits=s_len,
            matched_pattern=profile.sync_pattern if profile else None,
            confidence=0.95 if profile and profile.sync_pattern else 0.8
        )

        header_region = RegionInfo(
            start_bit=curr + s_len,
            end_bit=curr + s_len + h_len,
            length_bits=h_len,
            raw_bits=header_bits,
            raw_hex=bits_to_hex_str(header_bits),
            label="HEADER",
            confidence=0.9
        )

        payload_region = RegionInfo(
            start_bit=curr + s_len + h_len,
            end_bit=curr + s_len + h_len + p_len,
            length_bits=p_len,
            raw_bits=payload_bits,
            raw_hex=bits_to_hex_str(payload_bits),
            label="PAYLOAD",
            confidence=0.9
        )

        crc_info = CRCRegionInfo(
            raw_bits="".join(str(b) for b in crc_bits),
            raw_hex=bits_to_hex_str(crc_bits),
            start_bit=curr + s_len + h_len + p_len,
            end_bit=curr + f_len,
            length_bits=c_len,
            validation_status="UNCHECKED" if c_len > 0 else "NONE"
        )

        p_views = decode_payload_views(payload_bits)

        f_rec = FrameRecord(
            frame_index=frame_idx,
            start_bit=curr,
            end_bit=end,
            frame_length_bits=f_len,
            sync_region=sync_info,
            header_region=header_region,
            payload_region=payload_region,
            crc_region=crc_info,
            crc_metadata=crc_info,
            payload=p_views,
            is_partial=is_partial,
            frame_quality_score=1.0 if not is_partial else 0.6,
            frame_hash=compute_bit_hash(f_bits),
            header_hash=compute_bit_hash(header_bits),
            payload_hash=compute_bit_hash(payload_bits)
        )
        frames.append(f_rec)
        frame_idx += 1
        curr += frame_len

    return frames
