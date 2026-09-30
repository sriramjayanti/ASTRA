"""
segmentation.py
Frame segmentation, positional stability maps, position entropy, XOR difference maps,
and structural region (fixed header / variable payload) detection.
"""

from typing import List, Tuple, Dict
import numpy as np

from .models import StructuralRegion
from .entropy import binary_entropy


def build_frame_matrix(
    bits: np.ndarray,
    frame_length: int,
    offset: int = 0,
    max_frames: int = 64
) -> np.ndarray:
    """
    Reshape segmented bitstream into a 2D matrix of frames [num_frames, frame_length].
    """
    n = len(bits)
    if offset >= n or frame_length <= 0:
        return np.zeros((0, frame_length), dtype=np.uint8)

    trimmed = bits[offset:]
    total_frames = len(trimmed) // frame_length
    num_frames = min(total_frames, max_frames)

    if num_frames == 0:
        return np.zeros((0, frame_length), dtype=np.uint8)

    matrix = trimmed[:num_frames * frame_length].reshape(num_frames, frame_length)
    return matrix


def compute_position_stability(frame_matrix: np.ndarray) -> np.ndarray:
    """
    Compute bit stability for each position across frames.
    Stability = 2 * |P(bit = 1) - 0.5| -> Range [0.0, 1.0].
    1.0 means constant bit across all frames.
    0.0 means fully variable bit across frames.
    """
    if frame_matrix.shape[0] == 0:
        return np.zeros(frame_matrix.shape[1], dtype=np.float32)

    p1 = np.mean(frame_matrix, axis=0)
    stability = 2.0 * np.abs(p1 - 0.5)
    return stability.astype(np.float32)


def compute_position_entropy(frame_matrix: np.ndarray) -> np.ndarray:
    """
    Compute binary entropy for each bit position across frames.
    0.0 means completely invariant bit.
    1.0 means maximal variation.
    """
    if frame_matrix.shape[0] == 0:
        return np.ones(frame_matrix.shape[1], dtype=np.float32)

    p1 = np.mean(frame_matrix, axis=0)
    p0 = 1.0 - p1

    entropies = np.zeros_like(p1, dtype=np.float32)
    valid_mask = (p0 > 1e-12) & (p1 > 1e-12)

    entropies[valid_mask] = - (
        p0[valid_mask] * np.log2(p0[valid_mask]) +
        p1[valid_mask] * np.log2(p1[valid_mask])
    )
    return np.clip(entropies, 0.0, 1.0)


def compute_xor_frame_differences(frame_matrix: np.ndarray) -> np.ndarray:
    """
    Compute XOR difference between adjacent frames: F[i] XOR F[i+1].
    Returns 2D uint8 array of shape [num_frames - 1, frame_length].
    """
    if frame_matrix.shape[0] < 2:
        return np.zeros((0, frame_matrix.shape[1]), dtype=np.uint8)

    return np.bitwise_xor(frame_matrix[:-1], frame_matrix[1:])


def compute_frame_hamming_distances(frame_matrix: np.ndarray) -> Tuple[float, float, float]:
    """
    Compute average inter-frame Hamming distances:
    (whole frame, prefix (first 10%), suffix (last 10%)).
    """
    if frame_matrix.shape[0] < 2:
        return 0.0, 0.0, 0.0

    fl = frame_matrix.shape[1]
    p_len = max(8, fl // 10)
    s_len = max(8, fl // 10)

    xor_diffs = np.bitwise_xor(frame_matrix[:-1], frame_matrix[1:])
    whole_dist = float(np.mean(xor_diffs))
    prefix_dist = float(np.mean(xor_diffs[:, :p_len]))
    suffix_dist = float(np.mean(xor_diffs[:, -s_len:]))

    return whole_dist, prefix_dist, suffix_dist


def find_optimal_frame_offset(
    bits: np.ndarray,
    frame_length: int,
    max_search_offset: int = 128
) -> Tuple[int, float]:
    """
    Find starting bit offset that maximizes structural prefix stability (e.g. aligns prefix sync).
    """
    n = len(bits)
    search_limit = min(max_search_offset, frame_length, n - 2 * frame_length)
    if search_limit <= 0:
        return 0, 0.0

    best_offset = 0
    best_score = -1.0
    prefix_len = min(16, max(8, frame_length // 16))

    for off in range(search_limit):
        mat = build_frame_matrix(bits, frame_length, offset=off, max_frames=32)
        if mat.shape[0] < 3:
            continue

        stab = compute_position_stability(mat)
        prefix_stab = float(np.mean(stab[:prefix_len]))

        if prefix_stab > best_score:
            best_score = prefix_stab
            best_offset = off

    return best_offset, max(0.0, best_score)


def detect_structural_regions(
    position_stability: np.ndarray,
    position_entropy: np.ndarray,
    fixed_threshold: float = 0.85,
    variable_threshold: float = 0.60,
    min_region_len: int = 8
) -> List[StructuralRegion]:
    """
    Segment candidate frame into structural regions based on positional stability and entropy.
    """
    fl = len(position_stability)
    if fl == 0:
        return []

    bit_types = []
    for i in range(fl):
        s = position_stability[i]
        if s >= fixed_threshold:
            bit_types.append("fixed")
        elif s >= variable_threshold:
            bit_types.append("semi_fixed")
        else:
            bit_types.append("variable")

    regions: List[StructuralRegion] = []
    curr_type = bit_types[0]
    curr_start = 0

    for i in range(1, fl):
        if bit_types[i] != curr_type:
            curr_end = i
            r_stab = float(np.mean(position_stability[curr_start:curr_end]))
            r_ent = float(np.mean(position_entropy[curr_start:curr_end]))

            assigned_type = curr_type
            if curr_start >= fl - 32 and assigned_type in ["semi_fixed", "fixed"]:
                assigned_type = "checksum"

            regions.append(StructuralRegion(
                start_bit=curr_start,
                end_bit=curr_end,
                region_type=assigned_type,
                mean_stability=r_stab,
                mean_entropy=r_ent
            ))

            curr_type = bit_types[i]
            curr_start = i

    r_stab = float(np.mean(position_stability[curr_start:fl]))
    r_ent = float(np.mean(position_entropy[curr_start:fl]))
    assigned_type = curr_type
    if curr_start >= fl - 32 and assigned_type in ["semi_fixed", "fixed"]:
        assigned_type = "checksum"

    regions.append(StructuralRegion(
        start_bit=curr_start,
        end_bit=fl,
        region_type=assigned_type,
        mean_stability=r_stab,
        mean_entropy=r_ent
    ))

    return regions
