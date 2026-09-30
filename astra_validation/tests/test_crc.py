"""
test_crc.py
Tests for CRC calculation, profile registries, and test vectors.
"""

import pytest
import numpy as np
from astra_validation.src.crc import (
    load_crc_profiles,
    CRCCalculator,
    compute_crc,
    check_crc_frame,
    search_crc_candidates,
)
from astra_validation.src.models import EvidenceCheckState


def test_known_crc_vectors():
    """Verify all configured CRC profiles match standard ASCII '123456789' check vectors."""
    profiles = load_crc_profiles()
    assert len(profiles) >= 4

    for name, prof in profiles.items():
        calc = CRCCalculator(prof)
        assert calc.verify_check_vector(), f"Profile {name} failed check vector verification!"


def test_crc_positive_frame():
    """Verify CRC pass on a synthetic protected frame."""
    profiles = load_crc_profiles()
    prof = profiles["crc16_ccitt_false"]
    
    # 240-bit payload + 16-bit CRC = 256 bits
    np.random.seed(42)
    payload_bits = np.random.randint(0, 2, size=240, dtype=np.uint8)
    calc = CRCCalculator(prof)
    crc_val = calc.compute(payload_bits)

    crc_bits = np.array([(crc_val >> i) & 1 for i in range(15, -1, -1)], dtype=np.uint8)
    full_frame = np.concatenate([payload_bits, crc_bits])

    passed, c_crc, o_crc = check_crc_frame(full_frame, prof, 16)
    assert passed is True
    assert c_crc == crc_val
    assert o_crc == crc_val


def test_crc_negative_corrupted_frame():
    """Verify CRC fails when a single bit is flipped."""
    profiles = load_crc_profiles()
    prof = profiles["crc16_ccitt_false"]

    np.random.seed(123)
    payload_bits = np.random.randint(0, 2, size=240, dtype=np.uint8)
    calc = CRCCalculator(prof)
    crc_val = calc.compute(payload_bits)

    crc_bits = np.array([(crc_val >> i) & 1 for i in range(15, -1, -1)], dtype=np.uint8)
    full_frame = np.concatenate([payload_bits, crc_bits])

    # Flip 1 bit in payload
    full_frame[10] ^= 1

    passed, c_crc, o_crc = check_crc_frame(full_frame, prof, 16)
    assert passed is False
    assert c_crc != o_crc


def test_multi_hypothesis_crc_search():
    """Verify blind search identifies correct CRC profile and frame size."""
    profiles = load_crc_profiles()
    prof = profiles["crc16_ccitt_false"]

    # 4 frames of 128 bits
    frames = []
    np.random.seed(99)
    for _ in range(4):
        p = np.random.randint(0, 2, size=112, dtype=np.uint8)
        crc_val = compute_crc(p, prof)
        c_bits = np.array([(crc_val >> i) & 1 for i in range(15, -1, -1)], dtype=np.uint8)
        frames.append(np.concatenate([p, c_bits]))

    stream = np.concatenate(frames)
    results = search_crc_candidates(stream, profiles, candidate_lengths=[64, 128, 256])

    # Find the result for crc16_ccitt_false
    match = next(r for r in results if r.profile_name == "crc16_ccitt_false")
    assert match.passed_frames == 4
    assert match.pass_rate == 1.0
    assert match.check_state == EvidenceCheckState.PASS
