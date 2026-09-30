"""
CRC (Cyclic Redundancy Check) module for ASTRA Framing Engine.
Implements exact, standardized CRC profiles with bit-level and byte-level computation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np


@dataclass(frozen=True)
class CRCProfile:
    """Specification of a Cyclic Redundancy Check (CRC) algorithm."""
    name: str
    width: int            # CRC width in bits (8, 16, 32, etc.)
    poly: int             # Generator polynomial (without implicit MSB)
    init: int             # Initial register value
    refin: bool           # Reflect input bytes / bits before processing
    refout: bool          # Reflect output register before final XOR
    xorout: int           # Value XORed with final register value

    def to_dict(self) -> dict[str, str | int | bool]:
        return {
            "name": self.name,
            "width": self.width,
            "poly_hex": f"0x{self.poly:X}",
            "init_hex": f"0x{self.init:X}",
            "refin": self.refin,
            "refout": self.refout,
            "xorout_hex": f"0x{self.xorout:X}",
        }


# Standard CRC Profiles
CRC_PROFILES: dict[str, CRCProfile] = {
    # CRC-8 / ITU / ATM: x^8 + x^2 + x + 1 (Poly 0x07, init 0x00, vector '123456789' -> 0xF4)
    "crc8": CRCProfile(
        name="CRC-8",
        width=8,
        poly=0x07,
        init=0x00,
        refin=False,
        refout=False,
        xorout=0x00,
    ),
    # CRC-16-CCITT / FALSE: x^16 + x^12 + x^5 + 1 (Poly 0x1021, init 0xFFFF, vector '123456789' -> 0x29B1)
    "crc16_ccitt": CRCProfile(
        name="CRC-16-CCITT",
        width=16,
        poly=0x1021,
        init=0xFFFF,
        refin=False,
        refout=False,
        xorout=0x0000,
    ),
    # Alternative CRC-16-CCITT (init 0x0000)
    "crc16_ccitt_zero": CRCProfile(
        name="CRC-16-CCITT-ZERO",
        width=16,
        poly=0x1021,
        init=0x0000,
        refin=False,
        refout=False,
        xorout=0x0000,
    ),
    # Standard IEEE 802.3 CRC-32 (Poly 0x04C11DB7, init 0xFFFFFFFF, refin=True, refout=True, xorout=0xFFFFFFFF, vector '123456789' -> 0xCBF43926)
    "crc32": CRCProfile(
        name="CRC-32",
        width=32,
        poly=0x04C11DB7,
        init=0xFFFFFFFF,
        refin=True,
        refout=True,
        xorout=0xFFFFFFFF,
    ),
    # CRC-32-POSIX / MPEG-2: Poly 0x04C11DB7 unreflected
    "crc32_posix": CRCProfile(
        name="CRC-32-POSIX",
        width=32,
        poly=0x04C11DB7,
        init=0xFFFFFFFF,
        refin=False,
        refout=False,
        xorout=0xFFFFFFFF,
    ),
}


def _reflect(val: int, width: int) -> int:
    """Reflect (reverse) the low `width` bits of `val`."""
    res = 0
    for i in range(width):
        if (val >> i) & 1:
            res |= (1 << (width - 1 - i))
    return res


def get_crc_profile(name_or_profile: str | CRCProfile) -> CRCProfile:
    """Retrieve CRCProfile by name (case-insensitive) or return existing profile."""
    if isinstance(name_or_profile, CRCProfile):
        return name_or_profile
    
    key = str(name_or_profile).strip().lower().replace("-", "_")
    if key in CRC_PROFILES:
        return CRC_PROFILES[key]
    
    available = ", ".join(CRC_PROFILES.keys())
    raise ValueError(f"Unknown CRC profile '{name_or_profile}'. Available standard profiles: {available}")


def compute_crc(
    bits: np.ndarray,
    crc_type: str | CRCProfile = "crc16_ccitt",
) -> tuple[int, np.ndarray]:
    """Compute CRC over a 1D NumPy uint8 bit array according to profile parameters.

    Args:
        bits: 1D NumPy array of uint8 values (0 and 1) in MSB-first bit order.
        crc_type: Name of profile or CRCProfile instance.

    Returns:
        tuple (crc_int_value, crc_bits_array) where crc_bits_array is MSB-first uint8 array.
    """
    profile = get_crc_profile(crc_type)
    
    if not isinstance(bits, np.ndarray):
        bits = np.asarray(bits, dtype=np.uint8)
    if bits.ndim != 1:
        raise ValueError(f"Bits must be 1D array, got shape {bits.shape}")

    width = profile.width
    poly = profile.poly
    init = profile.init
    xorout = profile.xorout
    mask = (1 << width) - 1

    if profile.refin:
        ref_poly = _reflect(poly, width)
        reg = init
        # Group bits into bytes if multiple of 8, process each byte LSB to MSB as standard
        if len(bits) % 8 == 0:
            for byte_idx in range(len(bits) // 8):
                byte_bits = bits[byte_idx * 8 : (byte_idx + 1) * 8]
                # In MSB-first array: index 0 is MSB (bit 7), index 7 is LSB (bit 0)
                # Reflected transmission takes LSB first
                for b in reversed(byte_bits):
                    cur = (reg ^ int(b)) & 1
                    reg >>= 1
                    if cur:
                        reg ^= ref_poly
        else:
            for b in bits:
                cur = (reg ^ int(b)) & 1
                reg >>= 1
                if cur:
                    reg ^= ref_poly

        if not profile.refout:
            reg = _reflect(reg, width)
    else:
        # Standard unreflected direct CRC computation
        reg = init
        for b in bits:
            msb = ((reg >> (width - 1)) ^ (int(b) & 1)) & 1
            reg = (reg << 1) & mask
            if msb:
                reg ^= poly

        if profile.refout:
            reg = _reflect(reg, width)

    crc_val = (reg ^ xorout) & mask

    # Convert crc_val to MSB-first bit array of length `width`
    crc_bits = np.array(
        [(crc_val >> (width - 1 - i)) & 1 for i in range(width)],
        dtype=np.uint8,
    )

    return crc_val, crc_bits


def verify_crc(
    data_bits: np.ndarray,
    expected_crc_bits: np.ndarray,
    crc_type: str | CRCProfile = "crc16_ccitt",
) -> bool:
    """Verify that computed CRC over data_bits matches expected_crc_bits.

    Args:
        data_bits: Data bits that CRC was computed over.
        expected_crc_bits: Expected CRC bit array.
        crc_type: CRC profile name or instance.

    Returns:
        True if matches, False otherwise.
    """
    _, computed_bits = compute_crc(data_bits, crc_type=crc_type)
    return np.array_equal(computed_bits, expected_crc_bits)
