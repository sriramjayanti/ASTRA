"""
byte_alignment.py
Exploration and scoring of 8 bit offsets (0..7) for byte-level structural alignment.
"""

from typing import List, Tuple
import numpy as np

from .models import ByteAlignmentResult


def bits_to_bytes(bits: np.ndarray, offset: int = 0) -> np.ndarray:
    """Convert bitstream to uint8 bytes starting from bit offset."""
    n = len(bits)
    if offset >= n:
        return np.array([], dtype=np.uint8)

    trimmed = bits[offset:]
    n_bytes = len(trimmed) // 8
    if n_bytes == 0:
        return np.array([], dtype=np.uint8)

    byte_bits = trimmed[:n_bytes * 8].reshape(n_bytes, 8)
    powers = 2 ** np.arange(7, -1, -1, dtype=np.uint8)
    byte_vals = np.dot(byte_bits.astype(np.uint8), powers)
    return byte_vals


def compute_byte_entropy(byte_vals: np.ndarray) -> float:
    """Compute normalized Shannon entropy of 8-bit byte sequence (0.0 to 1.0)."""
    if len(byte_vals) == 0:
        return 1.0

    _, counts = np.unique(byte_vals, return_counts=True)
    probs = counts / float(len(byte_vals))
    h = -np.sum(probs * np.log2(probs))
    # Maximum possible byte entropy is 8 bits
    return float(np.clip(h / 8.0, 0.0, 1.0))


def analyze_byte_offsets(bits: np.ndarray) -> ByteAlignmentResult:
    """
    Evaluate all 8 bit offsets (0..7) to determine optimal byte alignment.

    Args:
        bits: 1D uint8 binary array.

    Returns:
        ByteAlignmentResult with per-offset metrics and identified best offset.
    """
    n = len(bits)
    if n < 16:
        return ByteAlignmentResult(
            best_offset=0,
            offset_entropies=[1.0] * 8,
            zero_byte_fractions=[0.0] * 8,
            printable_fractions=[0.0] * 8,
            structural_scores=[0.0] * 8
        )

    entropies: List[float] = []
    zero_fractions: List[float] = []
    printable_fractions: List[float] = []
    structural_scores: List[float] = []

    for offset in range(8):
        byte_vals = bits_to_bytes(bits, offset=offset)
        if len(byte_vals) == 0:
            entropies.append(1.0)
            zero_fractions.append(0.0)
            printable_fractions.append(0.0)
            structural_scores.append(0.0)
            continue

        # 1. Byte Entropy (lower indicates more structure/redundancy)
        h = compute_byte_entropy(byte_vals)
        entropies.append(h)

        # 2. Zero-byte (0x00) frequency (common in structured framing/padding)
        zero_frac = float(np.mean(byte_vals == 0x00))
        zero_fractions.append(zero_frac)

        # 3. Printable ASCII characters (0x20 - 0x7E, plus \r, \n, \t) - weak diagnostic
        printable = ((byte_vals >= 0x20) & (byte_vals <= 0x7E)) | (byte_vals == 0x0A) | (byte_vals == 0x0D) | (byte_vals == 0x09)
        print_frac = float(np.mean(printable))
        printable_fractions.append(print_frac)

        # 4. Structural score: rewards non-random entropy deviations and consistent zero bytes
        # High score if entropy is lower than 1.0 or zero bytes are elevated
        entropy_dev = max(0.0, 1.0 - h)
        score = float(0.6 * entropy_dev + 0.4 * min(1.0, zero_frac * 5.0))
        structural_scores.append(score)

    best_offset = int(np.argmax(structural_scores))

    return ByteAlignmentResult(
        best_offset=best_offset,
        offset_entropies=entropies,
        zero_byte_fractions=zero_fractions,
        printable_fractions=printable_fractions,
        structural_scores=structural_scores
    )
