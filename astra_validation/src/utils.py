"""
utils.py
Synthetic validation data generation, test vector generators, and dataset formatting utilities.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import hashlib
import json
from .models import CRCProfile
from .crc import compute_crc, load_crc_profiles, bits_to_bytes
from .sync_word import hex_to_bits


def generate_synthetic_frame(
    sync_pattern_hex: str = "EB90",
    version: int = 1,
    sequence_id: int = 0,
    payload_bytes: Optional[bytes] = None,
    payload_len: int = 56,
    flags: int = 0,
    crc_profile_name: str = "crc16_ccitt_false",
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate a complete single synthetic telemetry frame containing:
    [Sync Word (16b)] + [Version (4b) + SeqID (12b) + PayloadLen (8b) + Flags (8b)] + [Payload (56B)] + [CRC-16 (16b)]
    Total length = 512 bits (64 bytes).
    """
    if seed is not None:
        np.random.seed(seed)

    # 1. Sync word (16 bits)
    sync_bits = hex_to_bits(sync_pattern_hex)

    # 2. Header (32 bits = 4 bytes: 4b version, 12b seq, 8b len, 8b flags)
    hdr_bytes = bytes([
        ((version & 0x0F) << 4) | ((sequence_id >> 8) & 0x0F),
        sequence_id & 0xFF,
        payload_len & 0xFF,
        flags & 0xFF,
    ])
    hdr_bits = np.unpackbits(np.frombuffer(hdr_bytes, dtype=np.uint8))

    # 3. Payload (56 bytes = 448 bits)
    if payload_bytes is None:
        payload_bytes = np.random.randint(0, 256, size=payload_len, dtype=np.uint8).tobytes()
    else:
        payload_bytes = payload_bytes[:payload_len]
        if len(payload_bytes) < payload_len:
            payload_bytes += b'\x00' * (payload_len - len(payload_bytes))

    payload_bits = np.unpackbits(np.frombuffer(payload_bytes, dtype=np.uint8))

    # Combine for CRC
    protected_bits = np.concatenate([sync_bits, hdr_bits, payload_bits])

    # 4. CRC (16 bits)
    profiles = load_crc_profiles()
    profile = profiles.get(crc_profile_name)
    if profile is None:
        profile = CRCProfile(name="CRC-16", width=16, poly=0x1021, init=0xFFFF, refin=False, refout=False, xorout=0x0000)

    crc_val = compute_crc(protected_bits, profile)

    crc_bits = np.array([(crc_val >> i) & 1 for i in range(profile.width - 1, -1, -1)], dtype=np.uint8)
    full_frame = np.concatenate([protected_bits, crc_bits])
    return full_frame


def generate_synthetic_stream(
    num_frames: int = 8,
    sync_pattern_hex: str = "EB90",
    version: int = 1,
    start_seq_id: int = 100,
    payload_len: int = 56,
    crc_profile_name: str = "crc16_ccitt_false",
    bit_error_rate: float = 0.0,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate a concatenated multi-frame stream with monotonic sequence IDs and optional BER corruption.
    """
    np.random.seed(seed)
    frames = []
    for f in range(num_frames):
        frame = generate_synthetic_frame(
            sync_pattern_hex=sync_pattern_hex,
            version=version,
            sequence_id=start_seq_id + f,
            payload_len=payload_len,
            crc_profile_name=crc_profile_name,
            seed=seed + f * 17,
        )
        frames.append(frame)

    stream = np.concatenate(frames)
    if bit_error_rate > 0.0:
        error_mask = (np.random.rand(len(stream)) < bit_error_rate).astype(np.uint8)
        stream = stream ^ error_mask

    return stream


def compute_bitstream_hash(bits: Union[np.ndarray, List[int]]) -> str:
    """Compute deterministic SHA-256 hash of bitstream for candidate deduplication."""
    bits = np.asarray(bits, dtype=np.uint8).ravel()
    b_bytes = np.packbits(bits).tobytes()
    return hashlib.sha256(b_bytes).hexdigest()[:16]
