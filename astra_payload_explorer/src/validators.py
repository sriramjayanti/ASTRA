"""
validators.py
Protocol profile matching, schema validation, and profile compatibility scoring.
"""

from typing import List, Dict, Optional, Tuple, Any, Union
import numpy as np

from .models import ProtocolProfile
from .byte_alignment import hex_to_bits


def score_profile_match(
    bits: np.ndarray,
    profile: Union[ProtocolProfile, Dict[str, Any]],
    sample_frames: int = 4
) -> Tuple[float, List[str], List[str]]:
    """
    Evaluate how strongly a recovered bitstream matches a candidate protocol profile schema.

    Returns:
        Tuple: (match_score in [0.0, 1.0], list of supporting evidence, list of contradictions)
    """
    evidence = []
    contradictions = []
    score_components = []

    if isinstance(profile, ProtocolProfile):
        p_len = profile.frame_length_bits
        sync_pat = profile.sync_pattern
        sync_off = profile.sync_offset_bits
        fields = list(profile.header_fields.values()) if isinstance(profile.header_fields, dict) else profile.header_fields
    else:
        p_frame = profile.get("frame", {})
        p_len = p_frame.get("length_bits", 512)
        p_sync = profile.get("sync", {})
        sync_pat = p_sync.get("pattern_bits") or p_sync.get("pattern_hex")
        sync_off = p_sync.get("offset_bits", 0)
        p_hdr = profile.get("header", {})
        fields = p_hdr.get("fields", [])

    sync_bits = np.array([], dtype=np.uint8)
    if sync_pat:
        if all(c in "01" for c in sync_pat):
            sync_bits = np.array([int(c) for c in sync_pat], dtype=np.uint8)
        else:
            sync_bits = hex_to_bits(sync_pat)

    n_bits = len(bits)
    if n_bits < p_len:
        contradictions.append(f"Bitstream length ({n_bits}) shorter than frame length ({p_len})")
        return 0.1, evidence, contradictions

    # 1. Sync Word Verification
    if len(sync_bits) > 0:
        sync_len = len(sync_bits)
        num_checked = min(sample_frames, n_bits // p_len)
        sync_matches = 0

        for f_idx in range(num_checked):
            pos = f_idx * p_len + sync_off
            if pos + sync_len <= n_bits:
                seg = bits[pos:pos + sync_len]
                if np.array_equal(seg, sync_bits):
                    sync_matches += 1

        sync_ratio = sync_matches / max(1, num_checked)
        score_components.append(sync_ratio * 0.5)

        if sync_ratio >= 0.75:
            evidence.append(f"Sync word matched in {sync_matches}/{num_checked} frames")
        elif sync_ratio == 0:
            contradictions.append(f"Sync word not found at expected offset {sync_off}")
    else:
        score_components.append(0.2)

    # 2. Frame Length Fit
    remainder = n_bits % p_len
    if remainder == 0:
        evidence.append(f"Bitstream length {n_bits} is exact multiple of {p_len} bits")
        score_components.append(0.3)
    else:
        score_components.append(0.15)

    # 3. Header Field Constraints
    const_matches = 0
    const_total = 0

    for fld in fields:
        exp_val = getattr(fld, "expected_value", None) if not isinstance(fld, dict) else fld.get("expected_value")
        if exp_val is not None:
            const_total += 1
            f_off = getattr(fld, "offset_bits", 0) if not isinstance(fld, dict) else fld.get("offset_bits", 0)
            f_w = getattr(fld, "width_bits", 8) if not isinstance(fld, dict) else fld.get("width_bits", 8)
            if f_off + f_w <= n_bits:
                val = int(np.dot(bits[f_off:f_off + f_w], 2 ** np.arange(f_w - 1, -1, -1)))
                if val == exp_val:
                    const_matches += 1

    if const_total > 0:
        c_ratio = const_matches / const_total
        score_components.append(c_ratio * 0.2)
        if c_ratio == 1.0:
            evidence.append(f"All {const_total} constant header fields matched expected values")
    else:
        score_components.append(0.2)

    total_score = float(np.clip(sum(score_components), 0.0, 1.0))
    return total_score, evidence, contradictions
