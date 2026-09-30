"""
length_checks.py
Length-field consistency validation for ASTRA Stage 10.
Compares payload/frame length values declared in candidate headers against observed frame boundaries.
"""

from typing import List, Optional, Union
import numpy as np
from .models import LengthConsistencyResult, EvidenceCheckState
from .headers import extract_field_uint


def validate_length_field(
    frames: List[np.ndarray],
    offset_bits: int,
    length_bits: int,
    unit: str = "bytes",
    observed_frame_length_bits: Optional[int] = None,
    header_overhead_bytes: int = 6,
    crc_overhead_bytes: int = 2,
) -> LengthConsistencyResult:
    """
    Validate that the declared length in candidate frames is consistent with the observed frame length.
    """
    if not frames:
        return LengthConsistencyResult(check_state=EvidenceCheckState.NOT_TESTED)

    declared_lengths = []
    for f in frames:
        if len(f) >= offset_bits + length_bits:
            val = extract_field_uint(f, offset_bits, length_bits)
            declared_lengths.append(val)

    if not declared_lengths:
        return LengthConsistencyResult(check_state=EvidenceCheckState.NOT_TESTED)

    median_declared = int(np.median(declared_lengths))
    if observed_frame_length_bits is None:
        observed_frame_length_bits = len(frames[0])

    if unit == "bytes":
        declared_total_bits = (median_declared + header_overhead_bytes + crc_overhead_bytes) * 8
        observed_bytes = observed_frame_length_bits // 8
    else:
        declared_total_bits = median_declared
        observed_bytes = observed_frame_length_bits

    diff = abs(declared_total_bits - observed_frame_length_bits)
    is_valid = (diff <= 32)  # Allow small alignment variance / padding

    check_state = EvidenceCheckState.PASS if is_valid else EvidenceCheckState.FAIL

    return LengthConsistencyResult(
        declared_length=median_declared,
        observed_length=observed_bytes,
        is_valid=is_valid,
        check_state=check_state,
    )
