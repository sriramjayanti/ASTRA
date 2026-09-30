"""
cross_frame.py
Cross-frame comparisons, duplicate frame/payload detection, and variation metrics.
"""

from typing import List, Dict, Tuple, Any
from collections import defaultdict
import numpy as np

from .models import FrameRecord


def detect_duplicate_frames(frames: List[FrameRecord]) -> Dict[str, List[int]]:
    """Return map of frame_hash -> list of frame indices that share the hash (only for duplicates)."""
    hash_map: Dict[str, List[int]] = defaultdict(list)
    for f in frames:
        if f.frame_hash:
            hash_map[f.frame_hash].append(f.frame_index)

    duplicates = {h: indices for h, indices in hash_map.items() if len(indices) > 1}
    return duplicates


def analyze_payload_variations(frames: List[FrameRecord]) -> Dict[str, Any]:
    """Analyze variation in payload across frames."""
    if len(frames) < 2:
        return {"mean_byte_diff": 0.0, "identical_payload_count": 0}

    p_hashes = [f.payload_hash for f in frames if f.payload_hash]
    unique_p = len(set(p_hashes))
    identical_count = len(p_hashes) - unique_p

    return {
        "unique_payloads": unique_p,
        "identical_payload_count": identical_count,
        "is_payload_constant": (unique_p == 1)
    }


def check_sequence_continuity(frames: List[FrameRecord]) -> float:
    """Check continuity of sequence / counter fields across frames."""
    seq_vals = []
    for f in frames:
        fields = f.header_fields if isinstance(f.header_fields, dict) else {fld.field_name: fld for fld in f.header_fields}
        for name, fld in fields.items():
            if "seq" in name.lower() or "counter" in name.lower():
                if isinstance(fld.decoded_value, int):
                    seq_vals.append(fld.decoded_value)
                    break

    if len(seq_vals) < 2:
        return 1.0

    diffs = np.diff(seq_vals)
    return float(np.mean(diffs == 1))


def analyze_cross_frame_statistics(frames: List[FrameRecord]) -> Dict[str, Any]:
    """Perform multi-frame comparative analysis."""
    if not frames:
        return {
            "total_frames": 0,
            "unique_frames": 0,
            "unique_payloads": 0,
            "duplicate_frames_count": 0,
            "duplicate_payloads_count": 0,
        }

    dups = detect_duplicate_frames(frames)
    p_var = analyze_payload_variations(frames)
    seq_cont = check_sequence_continuity(frames)

    return {
        "total_frames": len(frames),
        "duplicate_frames": dups,
        "duplicate_frames_count": sum(len(v) for v in dups.values()),
        "payload_variations": p_var,
        "sequence_continuity_score": round(seq_cont, 4)
    }
