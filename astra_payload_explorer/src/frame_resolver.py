"""
frame_resolver.py
Resolves frame length, starting alignment offset, and candidate segmentation hypotheses.
"""

from typing import List, Dict, Optional, Tuple, Any
import numpy as np

from .models import FrameCandidate, ProtocolProfile


def resolve_frame_hypotheses(
    decoded_bits: np.ndarray,
    stage12_result: Optional[Any] = None,
    stage13_result: Optional[Any] = None,
    validation_result: Optional[Any] = None,
    profile: Optional[ProtocolProfile] = None,
    default_frame_length: int = 512
) -> List[FrameCandidate]:
    """
    Synthesize multi-stage evidence to resolve candidate frame length and alignment hypotheses.
    """
    candidates: List[FrameCandidate] = []
    seen = set()

    # 1. Profile evidence
    if profile is not None:
        p_len = getattr(profile, "frame_length_bits", default_frame_length)
        sync_pat = getattr(profile, "sync_pattern", None)
        sync_off = 0
        score = 0.95

        if sync_pat:
            pat_arr = np.array([int(c) for c in sync_pat if c in "01"], dtype=np.uint8)
            pat_len = len(pat_arr)
            if len(decoded_bits) >= pat_len:
                for off in range(min(len(decoded_bits) - pat_len + 1, p_len)):
                    if np.array_equal(decoded_bits[off:off+pat_len], pat_arr):
                        sync_off = off
                        break

        candidates.append(FrameCandidate(
            length_bits=p_len,
            alignment_offset_bits=sync_off,
            support_score=score,
            support_sources=["protocol_profile"],
            protocol_profile_id=profile.profile_id
        ))
        seen.add((p_len, sync_off))

    # 2. Stage 12 periodicity & sync candidates
    if stage12_result is not None:
        periodicity_cands = getattr(stage12_result, "periodicity_candidates", [])
        if not periodicity_cands:
            periodicity_cands = getattr(stage12_result, "frame_length_candidates", [])
        sync_cands = getattr(stage12_result, "sync_results", [])
        if not sync_cands:
            sync_cands = getattr(stage12_result, "sync_candidates", [])

        sync_off = 0
        if sync_cands:
            c0 = sync_cands[0]
            if hasattr(c0, "positions") and c0.positions:
                sync_off = c0.positions[0]
            elif hasattr(c0, "offset_in_stream"):
                sync_off = c0.offset_in_stream

        for i, pc in enumerate(periodicity_cands):
            lag = getattr(pc, "period_bits", getattr(pc, "lag", 0))
            strength = getattr(pc, "score", getattr(pc, "peak_strength", 0.8))
            if lag >= 16 and (lag, sync_off) not in seen:
                seen.add((lag, sync_off))
                candidates.append(FrameCandidate(
                    length_bits=lag,
                    alignment_offset_bits=sync_off,
                    support_score=float(strength),
                    stage12_score=float(strength),
                    support_sources=["stage12_autocorrelation_periodicity"]
                ))

    # 3. Stage 13 predicted regions evidence
    if stage13_result is not None:
        region_spans = getattr(stage13_result, "regions", getattr(stage13_result, "region_spans", []))
        sync_regions = [
            r for r in region_spans
            if (getattr(r, "label", "") in ["SYNC", "SYNC_PREAMBLE"] or (isinstance(r, dict) and r.get("label") in ["SYNC", "SYNC_PREAMBLE"]))
        ]
        if len(sync_regions) >= 2:
            s0_start = getattr(sync_regions[0], "start_bit", sync_regions[0].get("start", 0) if isinstance(sync_regions[0], dict) else 0)
            spacings = []
            for j in range(len(sync_regions)-1):
                s_next = getattr(sync_regions[j+1], "start_bit", sync_regions[j+1].get("start", 0) if isinstance(sync_regions[j+1], dict) else 0)
                s_curr = getattr(sync_regions[j], "start_bit", sync_regions[j].get("start", 0) if isinstance(sync_regions[j], dict) else 0)
                spacings.append(s_next - s_curr)
            median_sp = int(np.median(spacings))
            if (median_sp, s0_start) not in seen and median_sp >= 16:
                seen.add((median_sp, s0_start))
                candidates.append(FrameCandidate(
                    length_bits=median_sp,
                    alignment_offset_bits=s0_start,
                    support_score=0.88,
                    stage13_consistency=0.90,
                    support_sources=["stage13_transformer_regions"]
                ))

    # 4. Blind Autocorrelation / Repeating Sync Check on decoded_bits if no prior candidates
    if not candidates and len(decoded_bits) >= 64:
        # Check candidate frame lengths: 64, 128, 256, 512
        for cand_len in [64, 128, 256, 512, 1024]:
            if len(decoded_bits) >= cand_len * 2:
                # Check bit repetition
                f0 = decoded_bits[:cand_len]
                f1 = decoded_bits[cand_len:cand_len*2]
                match_ratio = np.mean(f0[:16] == f1[:16])
                if match_ratio >= 0.8:
                    candidates.append(FrameCandidate(
                        length_bits=cand_len,
                        alignment_offset_bits=0,
                        support_score=0.75,
                        support_sources=["blind_frame_repetition"]
                    ))
                    break

    # Fallback
    if not candidates:
        candidates.append(FrameCandidate(
            length_bits=default_frame_length,
            alignment_offset_bits=0,
            support_score=0.50,
            support_sources=["default_fallback"]
        ))

    candidates.sort(key=lambda c: c.support_score, reverse=True)
    return candidates


def select_frame_segmentation(candidates: List[FrameCandidate]) -> Optional[FrameCandidate]:
    """Select the highest scoring frame candidate."""
    if not candidates:
        return None
    return candidates[0]
