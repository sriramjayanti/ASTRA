"""
crc.py
Cyclic Redundancy Check (CRC) verification engine for ASTRA Stage 10.
Supports parameterized CRC calculation (CRC-8, CRC-16, CRC-32) using both
bitwise and table-driven implementations, profile registries, and multi-frame searching.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import yaml
import os
from .models import CRCProfile, CRCResult, EvidenceCheckState


def reflect_bits(val: int, width: int) -> int:
    """Reflect (reverse) bit order of an integer of given width."""
    res = 0
    for i in range(width):
        if (val >> i) & 1:
            res |= (1 << (width - 1 - i))
    return res


def bits_to_bytes(bits: Union[np.ndarray, List[int]]) -> bytes:
    """Convert a 1D bit array into a bytes object (MSB first per byte)."""
    bits = np.asarray(bits, dtype=np.uint8).ravel()
    # Pad to multiple of 8 if necessary
    pad_len = (8 - (len(bits) % 8)) % 8
    if pad_len > 0:
        bits = np.pad(bits, (0, pad_len), mode='constant', constant_values=0)
    
    byte_vals = np.packbits(bits)
    return bytes(byte_vals)


class CRCCalculator:
    """
    Parametric CRC Engine supporting standard Rocksoft model:
    width, poly, init, refin, refout, xorout.
    Precomputes 256-entry lookup tables for 8, 16, and 32-bit widths.
    """

    def __init__(self, profile: CRCProfile):
        self.profile = profile
        self.width = profile.width
        self.poly = profile.poly
        self.init = profile.init
        self.refin = profile.refin
        self.refout = profile.refout
        self.xorout = profile.xorout
        self.mask = (1 << self.width) - 1
        self._table = self._generate_table()

    def _generate_table(self) -> List[int]:
        """Generate 256-entry CRC lookup table."""
        table = []
        top_bit = 1 << (self.width - 1)
        for byte in range(256):
            cur = (byte << (self.width - 8)) if self.width >= 8 else (byte >> (8 - self.width))
            cur &= self.mask
            for _ in range(8):
                if cur & top_bit:
                    cur = ((cur << 1) ^ self.poly) & self.mask
                else:
                    cur = (cur << 1) & self.mask
            table.append(cur)
        return table

    def compute(self, data: Union[bytes, bytearray, np.ndarray, List[int]]) -> int:
        """
        Compute CRC over input byte sequence or bitstream according to Rocksoft parameterized model.
        """
        if isinstance(data, (np.ndarray, list)):
            arr = np.asarray(data)
            if arr.ndim > 0 and len(arr) > 0 and np.all((arr == 0) | (arr == 1)):
                byte_data = bits_to_bytes(arr)
            else:
                byte_data = bytes(arr.astype(np.uint8))
        else:
            byte_data = bytes(data)

        crc = self.init & self.mask
        for b in byte_data:
            cur_b = reflect_bits(b, 8) if self.refin else b
            if self.width >= 8:
                pos = ((crc >> (self.width - 8)) ^ cur_b) & 0xFF
                crc = ((crc << 8) ^ self._table[pos]) & self.mask
            else:
                pos = (crc ^ (cur_b >> (8 - self.width))) & 0xFF
                crc = self._table[pos] & self.mask

        if self.refout:
            crc = reflect_bits(crc, self.width)

        crc = (crc ^ self.xorout) & self.mask
        return crc

    def verify_check_vector(self) -> bool:
        """Verify CRC against standard ASCII test vector '123456789'."""
        if self.profile.check is None:
            return True
        check_data = b"123456789"
        computed = self.compute(check_data)
        return computed == self.profile.check


def load_crc_profiles(yaml_path: Optional[str] = None) -> Dict[str, CRCProfile]:
    """
    Load CRC profiles from yaml configuration file.
    """
    if yaml_path is None or not os.path.exists(yaml_path):
        # Default fallback location
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        yaml_path = os.path.join(base_dir, "configs", "crc_profiles.yaml")

    if not os.path.exists(yaml_path):
        # Built-in fallback catalog
        return {
            "crc8_atm": CRCProfile(name="CRC-8/ATM", width=8, poly=0x07, init=0x00, refin=False, refout=False, xorout=0x00, check=0xF4),
            "crc16_ccitt_false": CRCProfile(name="CRC-16/CCITT-FALSE", width=16, poly=0x1021, init=0xFFFF, refin=False, refout=False, xorout=0x0000, check=0x29B1),
            "crc16_ibm": CRCProfile(name="CRC-16/IBM", width=16, poly=0x8005, init=0x0000, refin=True, refout=True, xorout=0x0000, check=0xBB3D),
            "crc32_ieee": CRCProfile(name="CRC-32/IEEE", width=32, poly=0x04C11DB7, init=0xFFFFFFFF, refin=True, refout=True, xorout=0xFFFFFFFF, check=0xCBF43926),
        }

    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    profiles = {}
    for key, p in data.get("profiles", {}).items():
        profiles[key] = CRCProfile(
            name=p.get("name", key),
            width=int(p.get("width", 16)),
            poly=int(p.get("poly", 0)),
            init=int(p.get("init", 0)),
            refin=bool(p.get("refin", False)),
            refout=bool(p.get("refout", False)),
            xorout=int(p.get("xorout", 0)),
            check=int(p.get("check")) if p.get("check") is not None else None,
            description=p.get("description", ""),
        )
    return profiles


def compute_crc(data: Union[bytes, np.ndarray, List[int]], profile: CRCProfile) -> int:
    """Compute CRC for given data and profile."""
    calc = CRCCalculator(profile)
    return calc.compute(data)


def check_crc_frame(
    frame_bits: np.ndarray,
    profile: CRCProfile,
    crc_offset_from_end: int = 16,
) -> Tuple[bool, int, int]:
    """
    Check CRC on a single frame where CRC field is at the end of the frame.
    Returns: (pass_boolean, calculated_crc, observed_crc)
    """
    frame_bits = np.asarray(frame_bits, dtype=np.uint8).ravel()
    width = profile.width
    if len(frame_bits) <= width:
        return False, 0, 0

    payload_bits = frame_bits[:-width]
    crc_bits = frame_bits[-width:]

    # Extract observed CRC integer
    observed_crc = 0
    for b in crc_bits:
        observed_crc = (observed_crc << 1) | int(b)

    calc = CRCCalculator(profile)
    calc_crc = calc.compute(payload_bits)

    passed = (calc_crc == observed_crc)
    return passed, calc_crc, observed_crc


def search_crc_candidates(
    bitstream: np.ndarray,
    profiles: Dict[str, CRCProfile],
    candidate_lengths: List[int],
    max_offsets: int = 8,
) -> List[CRCResult]:
    """
    Multi-hypothesis blind CRC search across candidate frame lengths and profiles.
    Evaluates multi-frame consistency to avoid single-frame false positives.
    """
    bitstream = np.asarray(bitstream, dtype=np.uint8).ravel()
    total_bits = len(bitstream)
    results = []

    if total_bits < 32:
        return results

    for prof_key, profile in profiles.items():
        best_pass_rate = 0.0
        best_checked = 0
        best_passed = 0
        last_calc = None
        last_obs = None

        for frame_len in candidate_lengths:
            if frame_len <= profile.width or frame_len > total_bits:
                continue

            for offset in range(min(max_offsets, frame_len)):
                usable_bits = total_bits - offset
                num_frames = usable_bits // frame_len
                if num_frames < 1:
                    continue

                passed_frames = 0
                for f_idx in range(num_frames):
                    start = offset + f_idx * frame_len
                    end = start + frame_len
                    frame = bitstream[start:end]
                    passed, c_crc, o_crc = check_crc_frame(frame, profile, profile.width)
                    if passed:
                        passed_frames += 1
                    last_calc = c_crc
                    last_obs = o_crc

                pass_rate = passed_frames / num_frames
                if pass_rate > best_pass_rate or (pass_rate == best_pass_rate and num_frames > best_checked):
                    best_pass_rate = pass_rate
                    best_checked = num_frames
                    best_passed = passed_frames

        check_state = EvidenceCheckState.NOT_TESTED
        confidence = 0.0
        if best_checked > 0:
            if best_pass_rate >= 0.8:
                check_state = EvidenceCheckState.PASS
                # Higher confidence for wider CRC and more frames
                confidence = min(1.0, (profile.width / 32.0) * 0.5 + (best_checked / 5.0) * 0.5)
            elif best_pass_rate > 0.0:
                check_state = EvidenceCheckState.FAIL
                confidence = best_pass_rate * 0.5
            else:
                check_state = EvidenceCheckState.FAIL
                confidence = 0.0

        results.append(
            CRCResult(
                profile_name=prof_key,
                width=profile.width,
                checked_frames=best_checked,
                passed_frames=best_passed,
                pass_rate=best_pass_rate,
                calculated_crc=last_calc,
                observed_crc=last_obs,
                check_state=check_state,
                confidence=confidence,
            )
        )

    return results
