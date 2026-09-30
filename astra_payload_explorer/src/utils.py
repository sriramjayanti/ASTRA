"""
utils.py
Utilities and synthetic framed payload test generators for Stage 14.
"""

from typing import Tuple, Dict, Any, Optional, Union
import numpy as np

from .byte_alignment import bytes_to_bits, hex_to_bits
from .endian import encode_integer_with_endian


def build_test_frame(
    sync_bits: np.ndarray,
    header_bits: np.ndarray,
    payload_bits: np.ndarray,
    crc_bits: Optional[np.ndarray] = None
) -> np.ndarray:
    """Concatenate structural regions into a single binary frame."""
    parts = [sync_bits, header_bits, payload_bits]
    if crc_bits is not None:
        parts.append(crc_bits)
    return np.concatenate(parts).astype(np.uint8)


def generate_synthetic_stream(
    num_frames: int = 4,
    frame_length_bits: int = 128,
    sync_pattern: str = "1010101010101010",
    header_length_bits: int = 32,
    crc_length_bits: int = 16,
    payload_text: Optional[str] = None,
    seed: int = 42
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Generate synthetic multi-frame bitstream with known sync, header, payload, and CRC.
    """
    rng = np.random.RandomState(seed)

    if all(c in "01" for c in sync_pattern):
        sync_bits = np.array([int(c) for c in sync_pattern], dtype=np.uint8)
    else:
        sync_bits = hex_to_bits(sync_pattern)

    sync_len = len(sync_bits)
    p_len = max(0, frame_length_bits - sync_len - header_length_bits - crc_length_bits)

    frames = []
    for f_idx in range(num_frames):
        # Header: 8-bit version (1), 8-bit flags (0), 16-bit sequence (f_idx)
        if header_length_bits >= 32:
            hdr = np.zeros(header_length_bits, dtype=np.uint8)
            hdr[:8] = encode_integer_with_endian(1, 8, "big")
            hdr[8:16] = encode_integer_with_endian(0, 8, "big")
            hdr[16:32] = encode_integer_with_endian(f_idx, 16, "big")
        else:
            hdr = np.zeros(header_length_bits, dtype=np.uint8)

        # Payload
        if payload_text is not None:
            p_bytes = payload_text.encode("utf-8")
            p_bits_raw = bytes_to_bits(p_bytes)
            p_bits = np.zeros(p_len, dtype=np.uint8)
            copy_len = min(len(p_bits_raw), p_len)
            p_bits[:copy_len] = p_bits_raw[:copy_len]
        else:
            p_bits = rng.randint(0, 2, size=p_len, dtype=np.uint8)

        # CRC
        crc_bits = np.zeros(crc_length_bits, dtype=np.uint8) if crc_length_bits > 0 else np.array([], dtype=np.uint8)

        frame = build_test_frame(sync_bits, hdr, p_bits, crc_bits if crc_length_bits > 0 else None)
        frames.append(frame)

    stream = np.concatenate(frames)
    info = {
        "num_frames": num_frames,
        "frame_length_bits": frame_length_bits,
        "sync_pattern": sync_pattern,
        "sync_len": sync_len,
        "header_length_bits": header_length_bits,
        "crc_length_bits": crc_length_bits,
        "payload_length_bits": p_len
    }
    return stream, info


def generate_synthetic_payload_stream(
    payload_text: str = "hi hello",
    frame_length: int = 512,
    num_frames: int = 8,
    sync_word_hex: str = "EB90",
    header_length: int = 32,
    crc_length: int = 16,
    prefix_noise_bits: int = 0,
    seed: int = 42
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Generate synthetic framed bitstream containing exact known payload bytes (e.g. 'hi hello').
    """
    return generate_synthetic_stream(
        num_frames=num_frames,
        frame_length_bits=frame_length,
        sync_pattern=sync_word_hex,
        header_length_bits=header_length,
        crc_length_bits=crc_length,
        payload_text=payload_text,
        seed=seed
    )
