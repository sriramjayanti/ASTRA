"""
ASTRA Fusion Validation and Alignment Routines.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Union
import numpy as np
import torch

from .models import (
    ClassMappingMismatchError,
    InvalidProbabilityError,
    InvalidWeightError,
    SourceAlignmentError,
)


def validate_class_alignment(names_1d: Sequence[str], names_2d: Sequence[str]) -> None:
    """
    Validates that both branches share an identical class alphabet, identical ordering,
    and identical length.
    
    Raises:
        ClassMappingMismatchError: If any mismatch in count, names, or ordering is found.
    """
    if len(names_1d) != len(names_2d):
        raise ClassMappingMismatchError(
            f"Class count mismatch: 1D has {len(names_1d)} classes, 2D has {len(names_2d)} classes."
        )

    for i, (c1, c2) in enumerate(zip(names_1d, names_2d)):
        if str(c1).strip().lower() != str(c2).strip().lower():
            raise ClassMappingMismatchError(
                f"Class alphabet mismatch at index {i}: 1D='{c1}' vs 2D='{c2}'. "
                f"Full 1D: {list(names_1d)}, Full 2D: {list(names_2d)}"
            )


def validate_source_alignment(
    signal_id_1d: Optional[str],
    start_1d: Optional[int],
    end_1d: Optional[int],
    signal_id_2d: Optional[str],
    start_2d: Optional[int],
    end_2d: Optional[int],
) -> None:
    """
    Ensures that both predictions originate from the SAME physical IQ signal and sample window.
    
    Raises:
        SourceAlignmentError: If source IDs or window ranges do not match when provided.
    """
    # If both provide signal IDs, they must be identical
    if signal_id_1d is not None and signal_id_2d is not None:
        if str(signal_id_1d).strip() != str(signal_id_2d).strip():
            raise SourceAlignmentError(
                f"Cannot fuse predictions from different source signals: "
                f"1D source='{signal_id_1d}' vs 2D source='{signal_id_2d}'"
            )

    # If both provide start offsets, they must match
    if start_1d is not None and start_2d is not None:
        if start_1d != start_2d:
            raise SourceAlignmentError(
                f"Window start mismatch: 1D start={start_1d} vs 2D start={start_2d}"
            )

    # If both provide end offsets, they must match
    if end_1d is not None and end_2d is not None:
        if end_1d != end_2d:
            raise SourceAlignmentError(
                f"Window end mismatch: 1D end={end_1d} vs 2D end={end_2d}"
            )


def validate_probabilities(
    probs: Union[np.ndarray, torch.Tensor, Sequence[float]],
    tolerance: float = 1e-3,
) -> np.ndarray:
    """
    Validates that probabilities are finite, non-negative, and sum to approximately 1.0.
    
    Returns:
        np.ndarray: Validated float32 probability array [..., num_classes].
    
    Raises:
        InvalidProbabilityError: If invalid, NaN, Inf, or unnormalized.
    """
    if isinstance(probs, torch.Tensor):
        p_arr = probs.detach().cpu().numpy()
    else:
        p_arr = np.asarray(probs, dtype=np.float32)

    if not np.all(np.isfinite(p_arr)):
        raise InvalidProbabilityError("Probabilities contain NaN or Inf values.")

    if np.any(p_arr < -1e-5):
        raise InvalidProbabilityError(f"Probabilities contain negative values (min={p_arr.min()}).")

    # Clip very tiny negative numerical precision artifacts to 0
    p_arr = np.clip(p_arr, 0.0, None)

    # Check sum along the last dimension
    sums = np.sum(p_arr, axis=-1)
    if not np.all(np.abs(sums - 1.0) <= tolerance):
        raise InvalidProbabilityError(
            f"Probabilities do not sum to 1.0 within tolerance {tolerance}. "
            f"Observed sum range: [{sums.min():.4f}, {sums.max():.4f}]"
        )

    return p_arr


def validate_weights(w_1d: float, w_2d: float, tolerance: float = 1e-4) -> None:
    """
    Validates that fusion weights are non-negative and sum to 1.0.
    
    Raises:
        InvalidWeightError: If weights violate convex combination constraints.
    """
    if w_1d < 0.0 or w_2d < 0.0:
        raise InvalidWeightError(f"Fusion weights must be non-negative: w_1d={w_1d}, w_2d={w_2d}")

    total = w_1d + w_2d
    if abs(total - 1.0) > tolerance:
        raise InvalidWeightError(
            f"Fusion weights must sum to 1.0 (within {tolerance}). Got sum={total:.5f} (w_1d={w_1d}, w_2d={w_2d})"
        )
