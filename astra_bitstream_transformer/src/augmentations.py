"""
augmentations.py
Bitstream data augmentations: residual BER noise injection, random boundary offsets, and channel perturbations.
"""

from typing import Tuple, Optional
import numpy as np


def apply_ber_augmentation(
    bits: np.ndarray,
    ber: float = 0.001,
    rng: Optional[np.random.RandomState] = None
) -> np.ndarray:
    """
    Randomly flip bits with probability `ber` while leaving truth labels unchanged.
    """
    if ber <= 0.0 or len(bits) == 0:
        return bits.copy()

    if rng is None:
        rng = np.random.RandomState()

    flip_mask = rng.rand(len(bits)) < ber
    noisy_bits = np.bitwise_xor(bits.astype(np.uint8), flip_mask.astype(np.uint8))
    return noisy_bits


def apply_offset_augmentation(
    bits: np.ndarray,
    labels: np.ndarray,
    max_offset: int = 64,
    rng: Optional[np.random.RandomState] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepend and append random noise bits (labeled UNKNOWN = 0) so the model does not assume
    sync always starts at position 0.
    """
    if max_offset <= 0 or len(bits) == 0:
        return bits, labels

    if rng is None:
        rng = np.random.RandomState()

    prefix_len = rng.randint(0, max_offset + 1)
    suffix_len = rng.randint(0, max_offset + 1)

    prefix_bits = rng.randint(0, 2, size=prefix_len, dtype=np.uint8) if prefix_len > 0 else np.array([], dtype=np.uint8)
    suffix_bits = rng.randint(0, 2, size=suffix_len, dtype=np.uint8) if suffix_len > 0 else np.array([], dtype=np.uint8)

    prefix_labels = np.zeros(prefix_len, dtype=np.int64)
    suffix_labels = np.zeros(suffix_len, dtype=np.int64)

    augmented_bits = np.concatenate((prefix_bits, bits, suffix_bits))
    augmented_labels = np.concatenate((prefix_labels, labels, suffix_labels))

    return augmented_bits, augmented_labels


def add_confidence_noise(
    confidence: np.ndarray,
    noise_std: float = 0.05,
    rng: Optional[np.random.RandomState] = None
) -> np.ndarray:
    """Add small Gaussian noise to soft confidence values in [0.0, 1.0]."""
    if rng is None:
        rng = np.random.RandomState()
    noisy = confidence + rng.normal(0.0, noise_std, size=confidence.shape).astype(np.float32)
    return np.clip(noisy, 0.0, 1.0)
