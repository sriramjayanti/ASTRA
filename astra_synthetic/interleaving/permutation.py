"""
Permutation and index-mapping mathematics for ASTRA Interleaving Engine.
Provides rigorous forward application, exact inversion, and formal permutation validation.
"""

from __future__ import annotations

import numpy as np


def validate_permutation(permutation: np.ndarray, expected_length: int | None = None) -> None:
    """Validate that an array is a valid mathematical permutation of indices 0..N-1.

    Args:
        permutation: 1D integer NumPy array.
        expected_length: Optional expected length N.

    Raises:
        ValueError: If permutation contains duplicates, out-of-bounds indices, or wrong length.
    """
    if not isinstance(permutation, np.ndarray):
        raise TypeError(f"Permutation must be a NumPy array, got {type(permutation).__name__}")
    if permutation.ndim != 1:
        raise ValueError(f"Permutation must be 1D, got shape {permutation.shape}")

    n = len(permutation)
    if expected_length is not None and n != expected_length:
        raise ValueError(f"Permutation length ({n}) does not match expected length ({expected_length})")

    if n == 0:
        return

    # Check bounds
    min_val = np.min(permutation)
    max_val = np.max(permutation)
    if min_val < 0 or max_val >= n:
        raise ValueError(f"Permutation indices out of range [0, {n-1}]: min={min_val}, max={max_val}")

    # Check uniqueness (every index 0..N-1 must appear exactly once)
    unique_count = len(np.unique(permutation))
    if unique_count != n:
        raise ValueError(f"Permutation contains duplicate/missing indices (unique count={unique_count} != N={n})")


def invert_permutation(permutation: np.ndarray) -> np.ndarray:
    """Compute the exact inverse permutation.

    If output[i] = input[permutation[i]], then recovered[j] = output[inv_perm[j]] restores input.

    Args:
        permutation: 1D NumPy array of indices 0..N-1.

    Returns:
        1D NumPy array of inverse indices.
    """
    validate_permutation(permutation)
    n = len(permutation)
    inv = np.empty(n, dtype=np.int64)
    inv[permutation] = np.arange(n, dtype=np.int64)
    return inv


def apply_permutation(bits: np.ndarray, permutation: np.ndarray) -> np.ndarray:
    """Apply index permutation to reorder a 1D bit array.

    Args:
        bits: 1D NumPy uint8 array of length N.
        permutation: 1D NumPy array of length N where output[i] = bits[permutation[i]].

    Returns:
        Permuted 1D NumPy uint8 array of length N.
    """
    if len(bits) != len(permutation):
        raise ValueError(f"Bit array length ({len(bits)}) must match permutation length ({len(permutation)})")
    return bits[permutation]
