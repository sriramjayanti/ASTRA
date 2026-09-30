"""
blind_fields.py
Blind exploratory field discovery across repeated frames without prior protocol knowledge.
"""

from typing import List, Dict, Optional, Tuple, Any, Union
import numpy as np

from .models import FrameRecord, BlindDiscoveredField, EvidenceLevel
from .endian import decode_integer_with_endian
from .byte_alignment import bits_to_hex_str


def _extract_header_matrix(frames_or_matrix: Union[List[FrameRecord], np.ndarray]) -> Tuple[np.ndarray, List[int]]:
    """Helper to convert frames list or matrix into 2D numpy array and payload lengths list."""
    if isinstance(frames_or_matrix, np.ndarray):
        return frames_or_matrix, []
    
    if not frames_or_matrix:
        return np.empty((0, 0), dtype=np.uint8), []

    h_bits_list = []
    p_lens = []
    for f in frames_or_matrix:
        if f.header_region is not None and f.header_region.raw_bits is not None:
            h_bits_list.append(f.header_region.raw_bits)
        if f.payload_region is not None:
            p_lens.append(f.payload_region.length_bits)
        elif f.payload is not None:
            p_lens.append(f.payload.length_bits)
        else:
            p_lens.append(0)

    if h_bits_list:
        min_len = min(len(hb) for hb in h_bits_list)
        matrix = np.array([hb[:min_len] for hb in h_bits_list], dtype=np.uint8)
        return matrix, p_lens

    return np.empty((0, 0), dtype=np.uint8), p_lens


def discover_constant_fields(
    frames_or_matrix: Union[List[FrameRecord], np.ndarray],
    header_offset: int = 0,
    header_length: int = 64,
    min_width: int = 4
) -> List[BlindDiscoveredField]:
    """
    Identify bit spans within the header region that remain completely constant across all frames.
    """
    matrix, _ = _extract_header_matrix(frames_or_matrix)
    if matrix.shape[0] < 2:
        return []

    num_frames, frame_len = matrix.shape
    h_end = min(header_offset + header_length, frame_len)
    if header_offset >= h_end:
        return []

    hdr_slice = matrix[:, header_offset:h_end]
    is_const = np.all(hdr_slice == hdr_slice[0:1, :], axis=0)

    const_fields: List[BlindDiscoveredField] = []
    curr_start = None

    for i in range(len(is_const)):
        if is_const[i]:
            if curr_start is None:
                curr_start = i
        else:
            if curr_start is not None:
                width = i - curr_start
                if width >= min_width:
                    const_fields.append(BlindDiscoveredField(
                        field_name=f"constant_field_{header_offset + curr_start}",
                        offset_bits=header_offset + curr_start,
                        width_bits=width,
                        values_across_frames=[bits_to_hex_str(hdr_slice[0, curr_start:i])],
                        pattern="constant",
                        possible_role="constant_field_candidate",
                        confidence=0.85,
                        evidence_level=EvidenceLevel.INFERRED
                    ))
                curr_start = None

    if curr_start is not None:
        width = len(is_const) - curr_start
        if width >= min_width:
            const_fields.append(BlindDiscoveredField(
                field_name=f"constant_field_{header_offset + curr_start}",
                offset_bits=header_offset + curr_start,
                width_bits=width,
                values_across_frames=[bits_to_hex_str(hdr_slice[0, curr_start:])],
                pattern="constant",
                possible_role="constant_field_candidate",
                confidence=0.85,
                evidence_level=EvidenceLevel.INFERRED
            ))

    return const_fields


def discover_counter_candidates(
    frames_or_matrix: Union[List[FrameRecord], np.ndarray],
    header_offset: int = 0,
    header_length: int = 64,
    candidate_widths: Optional[List[int]] = None
) -> List[BlindDiscoveredField]:
    """
    Search for subfields within candidate header that exhibit monotonically incrementing sequence behavior.
    """
    matrix, _ = _extract_header_matrix(frames_or_matrix)
    if matrix.shape[0] < 2:
        return []

    if candidate_widths is None:
        candidate_widths = [4, 8, 12, 16, 24, 32]

    num_frames, frame_len = matrix.shape
    h_end = min(header_offset + header_length, frame_len)
    counter_candidates: List[BlindDiscoveredField] = []

    for width in sorted(candidate_widths):
        for off in range(header_offset, h_end - width + 1, 4 if width % 4 == 0 else 1):
            for endian in ["big", "little"]:
                values = []
                for f_idx in range(num_frames):
                    f_bits = matrix[f_idx, off:off + width]
                    val = decode_integer_with_endian(f_bits, endian=endian)
                    values.append(val)

                diffs = np.diff(values)
                increment_ones = np.sum(diffs == 1)
                ratio_ones = increment_ones / max(1, len(diffs))

                # Also verify not all zeros or all constant
                if ratio_ones >= 0.60 and len(set(values)) > 1:
                    conf = float(ratio_ones)
                    counter_candidates.append(BlindDiscoveredField(
                        field_name=f"counter_candidate_{off}_w{width}_{endian}",
                        offset_bits=off,
                        width_bits=width,
                        endianness=endian,
                        values_across_frames=values,
                        pattern="incrementing",
                        possible_role="counter_candidate",
                        confidence=round(conf, 4),
                        evidence_level=EvidenceLevel.INFERRED,
                        wrap_detected=bool(any(d < 0 for d in diffs))
                    ))

    # Sort prioritizing higher confidence, then smaller width (parsimony), then earlier offset
    counter_candidates.sort(key=lambda c: (c.confidence, -c.width_bits if c.confidence < 1.0 else (c.width_bits == 8, -c.width_bits), -c.offset_bits), reverse=True)
    return counter_candidates[:5]


def discover_length_candidates(
    frames_or_matrix: Union[List[FrameRecord], np.ndarray],
    payload_lengths: Optional[List[int]] = None,
    header_offset: int = 0,
    header_length: int = 64,
    candidate_widths: Optional[List[int]] = None
) -> List[BlindDiscoveredField]:
    """
    Identify header subfields that correlate with observed payload lengths across frames.
    """
    matrix, extracted_lens = _extract_header_matrix(frames_or_matrix)
    if payload_lengths is None:
        payload_lengths = extracted_lens

    if matrix.shape[0] < 2 or len(payload_lengths) < matrix.shape[0] or len(set(payload_lengths)) <= 1:
        return []

    if candidate_widths is None:
        candidate_widths = [8, 16]

    num_frames, frame_len = matrix.shape
    h_end = min(header_offset + header_length, frame_len)
    length_candidates: List[BlindDiscoveredField] = []

    for width in candidate_widths:
        for off in range(header_offset, h_end - width + 1, 8):
            for endian in ["big", "little"]:
                values = []
                for f_idx in range(num_frames):
                    f_bits = matrix[f_idx, off:off + width]
                    val = decode_integer_with_endian(f_bits, endian=endian)
                    values.append(val)

                p_arr = np.array(payload_lengths[:num_frames], dtype=np.float64)
                v_arr = np.array(values, dtype=np.float64)

                p_bytes = p_arr / 8.0
                exact_byte_matches = float(np.mean(v_arr == p_bytes))
                exact_bit_matches = float(np.mean(v_arr == p_arr))

                best_match = max(exact_byte_matches, exact_bit_matches)

                if best_match >= 0.70:
                    corr = 1.0 if np.std(v_arr) > 0 and np.std(p_arr) > 0 and np.corrcoef(v_arr, p_arr)[0, 1] > 0.9 else best_match
                    length_candidates.append(BlindDiscoveredField(
                        field_name=f"length_candidate_{off}_w{width}_{endian}",
                        offset_bits=off,
                        width_bits=width,
                        endianness=endian,
                        values_across_frames=values,
                        pattern="length_correlating",
                        possible_role="length_candidate",
                        confidence=0.90,
                        evidence_level=EvidenceLevel.INFERRED,
                        correlation=float(corr),
                        exact_match_fraction=best_match
                    ))

    return length_candidates


def analyze_field_variability(
    frames_or_matrix: Union[List[FrameRecord], np.ndarray]
) -> Dict[str, Any]:
    """Compute per-bit entropy across all frames in the header region."""
    matrix, _ = _extract_header_matrix(frames_or_matrix)
    if matrix.shape[0] == 0:
        return {"mean_entropy": 0.0, "constant_bit_count": 0, "variable_bit_count": 0}

    mean_bits = np.mean(matrix, axis=0)
    p = np.clip(mean_bits, 1e-12, 1.0 - 1e-12)
    bit_entropy = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))

    return {
        "mean_entropy": float(np.mean(bit_entropy)),
        "constant_bit_count": int(np.sum((mean_bits == 0.0) | (mean_bits == 1.0))),
        "variable_bit_count": int(np.sum((mean_bits > 0.0) & (mean_bits < 1.0))),
        "bit_entropy": [round(float(e), 4) for e in bit_entropy]
    }
