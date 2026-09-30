"""
merge.py
Merging overlapping window probabilities and predictions across long bitstreams.
"""

from typing import List, Tuple
import numpy as np


def merge_overlapping_predictions(
    window_probs: np.ndarray,
    window_slices: List[Tuple[int, int]],
    total_length: int,
    num_classes: int = 6,
    window_length: int = 2048,
    weighting: str = "triangular"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Aggregate overlapping window probability distributions into a single full-stream sequence.

    Args:
        window_probs: Numpy array of shape [num_windows, window_length, num_classes]
        window_slices: List of (start_bit, end_bit) corresponding to each window
        total_length: Total length of original bitstream
        num_classes: Number of structure classes
        window_length: Window size L
        weighting: Weighting strategy ('triangular', 'hann', or 'uniform')

    Returns:
        Tuple:
          - full_probabilities: [total_length, num_classes] (float32)
          - predicted_labels: [total_length] (int64)
    """
    full_prob_acc = np.zeros((total_length, num_classes), dtype=np.float32)
    weight_acc = np.zeros(total_length, dtype=np.float32)

    # Compute 1D window weight curve
    if weighting == "triangular":
        half_w = window_length / 2.0
        w_curve = 1.0 - np.abs(np.arange(window_length) - half_w) / half_w
        w_curve = np.clip(w_curve, 0.1, 1.0)
    elif weighting == "hann":
        w_curve = 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(window_length) / max(1, window_length - 1)))
        w_curve = np.clip(w_curve, 0.1, 1.0)
    else:
        w_curve = np.ones(window_length, dtype=np.float32)

    for w_idx, (start, end) in enumerate(window_slices):
        actual_len = end - start
        if actual_len <= 0 or start >= total_length:
            continue

        probs_slice = window_probs[w_idx, :actual_len, :] # [actual_len, num_classes]
        w_slice = w_curve[:actual_len, np.newaxis] # [actual_len, 1]

        full_prob_acc[start:end, :] += probs_slice * w_slice
        weight_acc[start:end] += w_slice.squeeze(-1)

    # Normalize by accumulated weights
    weight_acc = np.clip(weight_acc, 1e-6, None)[:, np.newaxis]
    full_probabilities = full_prob_acc / weight_acc

    # Ensure valid probability distribution per bit
    row_sums = np.sum(full_probabilities, axis=-1, keepdims=True)
    full_probabilities = full_probabilities / np.clip(row_sums, 1e-6, None)

    predicted_labels = np.argmax(full_probabilities, axis=-1).astype(np.int64)
    return full_probabilities, predicted_labels
