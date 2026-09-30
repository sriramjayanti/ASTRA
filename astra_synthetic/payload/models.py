"""
Data models and core bit/byte conversion utilities for ASTRA Payload Engine.
"""

from __future__ import annotations

import math
import hashlib
from dataclasses import dataclass, field
from typing import Any
import numpy as np


def bytes_to_bits(data: bytes) -> np.ndarray:
    """Convert bytes to a 1D uint8 NumPy array of bits (0 or 1) using MSB-first ordering.

    Args:
        data: Raw byte string.

    Returns:
        np.ndarray of dtype np.uint8 with values in {0, 1}.
    """
    if not isinstance(data, (bytes, bytearray)):
        raise TypeError(f"Expected bytes or bytearray, got {type(data).__name__}")
    if len(data) == 0:
        return np.empty(0, dtype=np.uint8)
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8))


def bits_to_bytes(bits: np.ndarray, padding: bool = False) -> bytes:
    """Convert a 1D uint8 NumPy array of bits (0 or 1) to bytes using MSB-first ordering.

    Args:
        bits: 1D array containing only 0 and 1.
        padding: If True and len(bits) is not a multiple of 8, pad with zeros at the end.
                 If False and len(bits) % 8 != 0, raises ValueError.

    Returns:
        Packed bytes.
    """
    if not isinstance(bits, np.ndarray):
        bits = np.asarray(bits, dtype=np.uint8)
    
    if bits.ndim != 1:
        raise ValueError(f"Expected 1D bit array, got array with shape {bits.shape}")
    
    if len(bits) == 0:
        return b""
    
    remainder = len(bits) % 8
    if remainder != 0:
        if padding:
            pad_len = 8 - remainder
            bits = np.pad(bits, (0, pad_len), mode="constant", constant_values=0)
        else:
            raise ValueError(
                f"Bit array length ({len(bits)}) is not a multiple of 8. "
                "Set padding=True to pad with trailing zeros."
            )
            
    return np.packbits(bits).tobytes()


def hex_to_bits(hex_string: str) -> np.ndarray:
    """Convert a hexadecimal string to a 1D uint8 NumPy array of bits.

    Args:
        hex_string: Hexadecimal string (e.g., 'DEADBEEF', '0x1A2B', 'a3 71').

    Returns:
        np.ndarray of dtype np.uint8 containing bits (MSB-first).
    """
    if not isinstance(hex_string, str):
        raise TypeError(f"Expected str, got {type(hex_string).__name__}")
    
    cleaned = hex_string.strip()
    if cleaned.lower().startswith("0x"):
        cleaned = cleaned[2:]
    cleaned = "".join(cleaned.split())
    
    if len(cleaned) == 0:
        return np.empty(0, dtype=np.uint8)
    
    if len(cleaned) % 2 != 0:
        raise ValueError(
            f"Hex string length must be even, got {len(cleaned)} characters: '{cleaned}'"
        )
    
    try:
        raw_bytes = bytes.fromhex(cleaned)
    except ValueError as e:
        raise ValueError(f"Invalid hexadecimal string: '{hex_string}'") from e
        
    return bytes_to_bits(raw_bytes)


def bits_to_hex(bits: np.ndarray, padding: bool = False) -> str:
    """Convert a 1D NumPy bit array to an uppercase hexadecimal string.

    Args:
        bits: 1D array containing only 0 and 1.
        padding: If True, pad to nearest byte multiple before converting.

    Returns:
        Uppercase hexadecimal string.
    """
    raw_bytes = bits_to_bytes(bits, padding=padding)
    return raw_bytes.hex().upper()


def text_to_bits(text: str, encoding: str = "utf-8") -> np.ndarray:
    """Encode text string into bytes and convert to MSB-first bits.

    Args:
        text: Input text string.
        encoding: Character encoding (default: 'utf-8').

    Returns:
        np.ndarray of uint8 bits.
    """
    if not isinstance(text, str):
        raise TypeError(f"Expected str, got {type(text).__name__}")
    try:
        raw_bytes = text.encode(encoding)
    except Exception as e:
        raise ValueError(f"Failed to encode text with encoding '{encoding}': {e}") from e
    return bytes_to_bits(raw_bytes)


def calculate_entropy(bits: np.ndarray) -> float:
    """Calculate binary Shannon entropy H(X) = -p0*log2(p0) - p1*log2(p1).

    Args:
        bits: 1D array of binary bits (0 and 1).

    Returns:
        Shannon entropy in the range [0.0, 1.0].
    """
    if len(bits) == 0:
        return 0.0
    
    n_total = len(bits)
    n_ones = int(np.count_nonzero(bits))
    n_zeros = n_total - n_ones
    
    if n_ones == 0 or n_zeros == 0:
        return 0.0
    
    p0 = n_zeros / n_total
    p1 = n_ones / n_total
    
    h = - (p0 * math.log2(p0) + p1 * math.log2(p1))
    # Clamp to [0.0, 1.0] to handle any float precision imprecision
    return float(max(0.0, min(1.0, round(h, 6))))


@dataclass
class PayloadRecord:
    """Represents a generated original ground truth source payload.

    Attributes:
        payload_id: Unique identifier (e.g. 'payload_000001').
        payload_type: Generation mode (e.g. 'random_bits', 'text', etc.).
        payload_bits: 1D np.ndarray of uint8 bits (0 and 1).
        payload_bytes: Packed byte representation.
        bit_length: Total number of bits in payload_bits.
        byte_length: Total number of bytes in payload_bytes.
        seed: Random seed used (if applicable).
        source_text: Source text if payload_type is text.
        pattern: Pattern string if payload_type is repeated_pattern.
        entropy_estimate: Binary Shannon entropy [0.0, 1.0].
        sha256: Hexadecimal SHA-256 hash of payload_bytes.
        metadata: Generation parameters and audit metadata dictionary.
    """
    payload_id: str
    payload_type: str
    payload_bits: np.ndarray
    payload_bytes: bytes
    bit_length: int
    byte_length: int
    seed: int | None = None
    source_text: str | None = None
    pattern: str | None = None
    entropy_estimate: float = 0.0
    sha256: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Ensure payload_bits is a 1D uint8 numpy array
        if not isinstance(self.payload_bits, np.ndarray):
            self.payload_bits = np.asarray(self.payload_bits, dtype=np.uint8)
        elif self.payload_bits.dtype != np.uint8:
            self.payload_bits = self.payload_bits.astype(np.uint8)
            
        if not self.sha256 and self.payload_bytes is not None:
            self.sha256 = hashlib.sha256(self.payload_bytes).hexdigest()

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, PayloadRecord):
            return False
        return (
            self.payload_id == other.payload_id
            and self.payload_type == other.payload_type
            and np.array_equal(self.payload_bits, other.payload_bits)
            and self.payload_bytes == other.payload_bytes
            and self.bit_length == other.bit_length
            and self.byte_length == other.byte_length
            and self.seed == other.seed
            and self.source_text == other.source_text
            and self.pattern == other.pattern
            and math.isclose(self.entropy_estimate, other.entropy_estimate, abs_tol=1e-5)
            and self.sha256 == other.sha256
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert record metadata to a JSON-serializable dictionary."""
        return {
            "payload_id": self.payload_id,
            "payload_type": self.payload_type,
            "bit_length": self.bit_length,
            "byte_length": self.byte_length,
            "seed": self.seed,
            "source_text": self.source_text,
            "pattern": self.pattern,
            "entropy_estimate": self.entropy_estimate,
            "sha256": self.sha256,
            "metadata": self.metadata,
        }
