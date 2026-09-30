"""
utils.py
Synthetic framed bitstream generation, hashing, formatting, and test utility functions.
"""

from typing import List, Tuple, Optional
import hashlib
import numpy as np


def compute_bitstream_hash(bits: np.ndarray) -> str:
    """Compute SHA-256 hash of bitstream array for deduplication and caching."""
    if bits is None or len(bits) == 0:
        return ""
    return hashlib.sha256(bits.astype(np.uint8).tobytes()).hexdigest()[:16]


def generate_synthetic_framed_bitstream(
    frame_length: int = 512,
    num_frames: int = 32,
    sync_word_hex: str = "EB90",
    header_length: int = 32,
    seed: int = 42,
    ber: float = 0.0,
    prefix_noise_bits: int = 0
) -> Tuple[np.ndarray, dict]:
    """
    Generate synthetic structured bitstream with known framing truth:
    [Sync Word][Header][Variable Payload][CRC/Checksum].

    Args:
        frame_length: Frame length in bits.
        num_frames: Total frames to generate.
        sync_word_hex: Hex string of sync pattern.
        header_length: Length of fixed/semi-fixed header in bits.
        seed: Random seed for reproducibility.
        ber: Bit error rate to inject.
        prefix_noise_bits: Number of random noise bits to prepend before frames.

    Returns:
        Tuple[bits, ground_truth_dict]
    """
    rng = np.random.RandomState(seed)

    # Convert sync hex to bits
    from .cross_correlation import hex_to_bits
    sync_bits = hex_to_bits(sync_word_hex)
    sync_len = len(sync_bits)

    assert frame_length > sync_len + header_length, "Frame length must exceed sync + header length."
    payload_len = frame_length - sync_len - header_length

    # Semi-fixed header template
    header_template = rng.randint(0, 2, size=header_length, dtype=np.uint8)

    all_frames = []
    for f_idx in range(num_frames):
        # 1. Sync word
        frame = list(sync_bits)

        # 2. Header: template with incrementing frame counter in first 8 bits
        hdr = header_template.copy()
        hdr[:8] = [(f_idx >> (7 - b)) & 1 for b in range(8)]
        frame.extend(hdr)

        # 3. Variable high-entropy payload
        payload = rng.randint(0, 2, size=payload_len, dtype=np.uint8)
        frame.extend(payload)

        all_frames.extend(frame)

    raw_bits = np.array(all_frames, dtype=np.uint8)

    # Optional prefix noise (to test offset recovery)
    if prefix_noise_bits > 0:
        noise = rng.randint(0, 2, size=prefix_noise_bits, dtype=np.uint8)
        raw_bits = np.concatenate((noise, raw_bits))

    # Optional Bit Error Rate
    if ber > 0.0:
        error_mask = rng.rand(len(raw_bits)) < ber
        raw_bits = np.bitwise_xor(raw_bits, error_mask.astype(np.uint8))

    truth = {
        "frame_length": frame_length,
        "num_frames": num_frames,
        "sync_word_hex": sync_word_hex,
        "sync_length": sync_len,
        "header_length": header_length,
        "payload_length": payload_len,
        "prefix_noise_bits": prefix_noise_bits,
        "ber": ber,
        "total_bits": len(raw_bits)
    }

    return raw_bits, truth
