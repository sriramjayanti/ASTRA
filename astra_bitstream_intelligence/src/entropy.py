"""
entropy.py
Shannon entropy, block/n-gram entropy, sliding-window entropy, and change-point analysis.
"""

from typing import List, Dict
import numpy as np


def binary_entropy(bits: np.ndarray) -> float:
    """
    Calculate global Shannon binary entropy for 1D bitstream.
    H = -p0 * log2(p0) - p1 * log2(p1)
    Returns value in [0.0, 1.0].
    """
    if bits is None or len(bits) == 0:
        return 0.0

    p1 = np.mean(bits)
    p0 = 1.0 - p1

    if p0 <= 1e-12 or p1 <= 1e-12:
        return 0.0

    h = - (p0 * np.log2(p0) + p1 * np.log2(p1))
    return float(np.clip(h, 0.0, 1.0))


def block_entropy(bits: np.ndarray, ngram_size: int = 8) -> float:
    """
    Calculate normalized block/n-gram entropy for N-bit blocks.
    Normalized by ngram_size so the result is in [0.0, 1.0].

    Args:
        bits: 1D uint8 array of bits.
        ngram_size: Size of block in bits (e.g. 1, 2, 4, 8, 16).

    Returns:
        Normalized block entropy in [0.0, 1.0].
    """
    if bits is None or len(bits) < ngram_size or ngram_size <= 0:
        return binary_entropy(bits)

    if ngram_size == 1:
        return binary_entropy(bits)

    # Trim to multiple of ngram_size
    n_blocks = len(bits) // ngram_size
    if n_blocks == 0:
        return binary_entropy(bits)

    truncated = bits[:n_blocks * ngram_size].reshape(n_blocks, ngram_size)

    # Convert binary blocks to integer symbols
    powers = 2 ** np.arange(ngram_size - 1, -1, -1, dtype=np.uint32)
    symbols = np.dot(truncated.astype(np.uint32), powers)

    _, counts = np.unique(symbols, return_counts=True)
    probs = counts / n_blocks

    h = -np.sum(probs * np.log2(probs))
    # Maximum possible entropy for ngram_size is ngram_size bits
    normalized_h = h / float(ngram_size)
    return float(np.clip(normalized_h, 0.0, 1.0))


def compute_all_ngram_entropies(bits: np.ndarray, ngram_sizes: List[int] = None) -> Dict[str, float]:
    """Compute block entropies for multiple specified n-gram sizes."""
    if ngram_sizes is None:
        ngram_sizes = [1, 2, 4, 8]

    res = {}
    for n in ngram_sizes:
        if len(bits) >= n:
            res[f"{n}bit"] = block_entropy(bits, n)
        else:
            res[f"{n}bit"] = binary_entropy(bits)
    return res


def sliding_entropy(bits: np.ndarray, window_size: int = 128, step: int = 16) -> List[float]:
    """
    Compute sliding-window binary entropy across the bitstream.

    Args:
        bits: 1D uint8 array of bits.
        window_size: Window size in bits.
        step: Step size in bits.

    Returns:
        List of entropy values per window.
    """
    n = len(bits)
    if n < window_size:
        return [binary_entropy(bits)]

    entropy_profile = []
    for start in range(0, n - window_size + 1, step):
        w = bits[start:start + window_size]
        entropy_profile.append(binary_entropy(w))

    return entropy_profile


def detect_entropy_change_points(
    entropy_profile: List[float],
    threshold: float = 0.15,
    step: int = 16
) -> List[int]:
    """
    Detect bit positions where local entropy transitions sharply.

    Args:
        entropy_profile: List of windowed entropy values.
        threshold: Absolute change threshold between consecutive windows.
        step: Step size used to generate entropy_profile.

    Returns:
        List of candidate bit offset change points.
    """
    if len(entropy_profile) < 2:
        return []

    profile = np.array(entropy_profile, dtype=np.float32)
    diffs = np.abs(np.diff(profile))

    # Indices where step change exceeds threshold
    change_idx = np.where(diffs >= threshold)[0]

    # Convert window indices to approximate bit offsets
    bit_offsets = [(int(idx) + 1) * step for idx in change_idx]
    return bit_offsets
