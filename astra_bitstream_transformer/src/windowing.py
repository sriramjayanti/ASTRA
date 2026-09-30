"""
windowing.py
Window slicing and batch preparation for fixed and variable length bitstreams.
"""

from typing import List, Tuple, Optional
import numpy as np
import torch


def slice_sequence_windows(
    channels: np.ndarray,
    labels: Optional[np.ndarray] = None,
    window_length: int = 2048,
    overlap_fraction: float = 0.5
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], List[Tuple[int, int]]]:
    """
    Slice multi-channel bitstream array [C, N] into overlapping windows [num_windows, C, window_length].

    Args:
        channels: 2D numpy array [C, N]
        labels: Optional 1D numpy array [N] of label IDs
        window_length: Window size in bits (L)
        overlap_fraction: Fraction of window overlap (e.g. 0.5)

    Returns:
        Tuple:
          - window_channels: [num_windows, C, window_length] (float32)
          - padding_masks: [num_windows, window_length] (bool: True = padded position)
          - window_labels: [num_windows, window_length] (int64) or None
          - window_slices: List of (start_idx, end_idx) in original bitstream
    """
    C, N = channels.shape
    step = max(1, int(window_length * (1.0 - overlap_fraction)))

    window_list = []
    mask_list = []
    label_list = []
    slice_list = []

    start = 0
    while start < N or (start == 0 and N == 0):
        end = min(start + window_length, N)
        actual_len = end - start

        # Channel slice
        win_c = np.zeros((C, window_length), dtype=np.float32)
        if actual_len > 0:
            win_c[:, :actual_len] = channels[:, start:end]

        # Padding mask: True where padded
        pad_mask = np.zeros(window_length, dtype=bool)
        if actual_len < window_length:
            pad_mask[actual_len:] = True

        window_list.append(win_c)
        mask_list.append(pad_mask)
        slice_list.append((start, end))

        # Labels slice
        if labels is not None:
            win_l = np.full(window_length, fill_value=-100, dtype=np.int64)
            if actual_len > 0:
                win_l[:actual_len] = labels[start:end]
            label_list.append(win_l)

        if end >= N:
            break
        start += step

    arr_windows = np.stack(window_list, axis=0)
    arr_masks = np.stack(mask_list, axis=0)
    arr_labels = np.stack(label_list, axis=0) if labels is not None else None

    return arr_windows, arr_masks, arr_labels, slice_list
