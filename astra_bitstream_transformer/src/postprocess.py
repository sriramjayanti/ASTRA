"""
postprocess.py
Region segmentation, small-region filtering, uncertainty estimation, and confidence thresholding.
"""

from typing import List, Dict, Optional, Any, Tuple
import numpy as np

from .models import StructureLabel, RegionPrediction, ID_TO_LABEL, LABEL_NAMES


def apply_unknown_confidence_threshold(
    predicted_labels: np.ndarray,
    label_probabilities: np.ndarray,
    unknown_threshold: float = 0.40
) -> np.ndarray:
    """
    If maximum probability for a bit position is below threshold, reclassify as UNKNOWN (0).
    """
    max_probs = np.max(label_probabilities, axis=-1)
    refined = predicted_labels.copy()
    low_conf_mask = max_probs < unknown_threshold
    refined[low_conf_mask] = StructureLabel.UNKNOWN.value
    return refined


def extract_contiguous_regions(
    predicted_labels: np.ndarray,
    label_probabilities: np.ndarray
) -> List[RegionPrediction]:
    """
    Convert 1D per-bit labels into a list of contiguous RegionPrediction dataclass instances.
    """
    n = len(predicted_labels)
    if n == 0:
        return []

    regions: List[RegionPrediction] = []
    curr_label_id = int(predicted_labels[0])
    curr_start = 0

    for i in range(1, n):
        lbl_id = int(predicted_labels[i])
        if lbl_id != curr_label_id:
            # Finalize previous region
            curr_end = i
            r_probs = label_probabilities[curr_start:curr_end, curr_label_id]
            mean_p = float(np.mean(r_probs))
            min_p = float(np.min(r_probs))

            regions.append(RegionPrediction(
                start_bit=curr_start,
                end_bit=curr_end,
                label=ID_TO_LABEL.get(curr_label_id, "UNKNOWN"),
                label_id=curr_label_id,
                mean_probability=mean_p,
                min_probability=min_p
            ))

            curr_label_id = lbl_id
            curr_start = i

    # Last region
    r_probs = label_probabilities[curr_start:n, curr_label_id]
    mean_p = float(np.mean(r_probs))
    min_p = float(np.min(r_probs))

    regions.append(RegionPrediction(
        start_bit=curr_start,
        end_bit=n,
        label=ID_TO_LABEL.get(curr_label_id, "UNKNOWN"),
        label_id=curr_label_id,
        mean_probability=mean_p,
        min_probability=min_p
    ))

    return regions


def filter_small_regions(
    regions: List[RegionPrediction],
    min_region_sizes: Optional[Dict[str, int]] = None,
    default_min_len: int = 4
) -> List[RegionPrediction]:
    """
    Filter out isolated single-bit or very small label spikes that violate class-specific minimum lengths,
    merging them into the adjacent dominant region.
    """
    if len(regions) <= 1:
        return regions

    min_sizes = min_region_sizes or {
        "UNKNOWN": 4,
        "SYNC": 8,
        "HEADER": 8,
        "PAYLOAD": 16,
        "CRC": 4,
        "PADDING": 4
    }

    cleaned: List[RegionPrediction] = []
    for r in regions:
        required_min = min_sizes.get(r.label, default_min_len)

        if r.bit_length < required_min and len(cleaned) > 0:
            # Merge into previous region
            prev = cleaned[-1]
            merged_len = prev.bit_length + r.bit_length
            prev.end_bit = r.end_bit
            prev.bit_length = merged_len
            prev.mean_probability = (prev.mean_probability * prev.bit_length + r.mean_probability * r.bit_length) / max(1, merged_len)
            prev.min_probability = min(prev.min_probability, r.min_probability)
        else:
            cleaned.append(r)

    # Re-merge adjacent identical labels
    merged_adjacent: List[RegionPrediction] = []
    for r in cleaned:
        if merged_adjacent and merged_adjacent[-1].label_id == r.label_id:
            prev = merged_adjacent[-1]
            merged_len = prev.bit_length + r.bit_length
            prev.end_bit = r.end_bit
            prev.bit_length = merged_len
            prev.mean_probability = (prev.mean_probability + r.mean_probability) / 2.0
            prev.min_probability = min(prev.min_probability, r.min_probability)
        else:
            merged_adjacent.append(r)

    return merged_adjacent


def compute_uncertainty_metrics(
    label_probabilities: np.ndarray
) -> Tuple[float, List[Dict[str, Any]]]:
    """
    Compute Shannon entropy across label probabilities per bit and identify high-uncertainty spans.
    """
    if len(label_probabilities) == 0:
        return 0.0, []

    # Normalized entropy in [0, 1]
    eps = 1e-12
    p = np.clip(label_probabilities, eps, 1.0)
    entropy_per_bit = -np.sum(p * np.log2(p), axis=-1) / np.log2(label_probabilities.shape[-1])

    mean_uncertainty = float(np.mean(entropy_per_bit))

    # Find spans where entropy > 0.60
    high_ent_mask = entropy_per_bit > 0.60
    uncertain_spans = []

    if np.any(high_ent_mask):
        changes = np.diff(high_ent_mask.astype(np.int32))
        starts = np.where(changes == 1)[0] + 1
        ends = np.where(changes == -1)[0] + 1

        if high_ent_mask[0]:
            starts = np.insert(starts, 0, 0)
        if high_ent_mask[-1]:
            ends = np.append(ends, len(high_ent_mask))

        for s, e in zip(starts, ends):
            if e - s >= 4:
                uncertain_spans.append({
                    "start_bit": int(s),
                    "end_bit": int(e),
                    "bit_length": int(e - s),
                    "mean_entropy": round(float(np.mean(entropy_per_bit[s:e])), 4)
                })

    return mean_uncertainty, uncertain_spans
