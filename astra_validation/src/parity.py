"""
parity.py
Parity checking mechanisms for ASTRA Stage 10 — Validation Engine.
Supports even, odd, and 2D block parity consistency evaluations.
"""

from typing import List, Optional, Tuple, Union
import numpy as np
from .models import ParityResult, EvidenceCheckState


def check_even_parity(
    bits: Union[np.ndarray, List[int]],
    block_size: int = 8,
) -> ParityResult:
    """
    Check if every block_size block has even parity (sum of bits % 2 == 0).
    """
    bits = np.asarray(bits, dtype=np.uint8).ravel()
    total_bits = len(bits)
    num_blocks = total_bits // block_size

    if num_blocks == 0:
        return ParityResult(parity_type="even", check_state=EvidenceCheckState.NOT_TESTED)

    usable_bits = bits[:num_blocks * block_size]
    reshaped = usable_bits.reshape(num_blocks, block_size)
    block_sums = np.sum(reshaped, axis=1)
    passed_mask = (block_sums % 2 == 0)
    passed_count = int(np.sum(passed_mask))
    pass_rate = passed_count / num_blocks

    check_state = EvidenceCheckState.NOT_TESTED
    if pass_rate >= 0.95:
        check_state = EvidenceCheckState.PASS
    elif pass_rate <= 0.70:
        check_state = EvidenceCheckState.FAIL
    else:
        check_state = EvidenceCheckState.NOT_TESTED

    return ParityResult(
        parity_type="even",
        blocks_tested=num_blocks,
        blocks_passed=passed_count,
        pass_rate=pass_rate,
        check_state=check_state,
    )


def check_odd_parity(
    bits: Union[np.ndarray, List[int]],
    block_size: int = 8,
) -> ParityResult:
    """
    Check if every block_size block has odd parity (sum of bits % 2 == 1).
    """
    bits = np.asarray(bits, dtype=np.uint8).ravel()
    total_bits = len(bits)
    num_blocks = total_bits // block_size

    if num_blocks == 0:
        return ParityResult(parity_type="odd", check_state=EvidenceCheckState.NOT_TESTED)

    usable_bits = bits[:num_blocks * block_size]
    reshaped = usable_bits.reshape(num_blocks, block_size)
    block_sums = np.sum(reshaped, axis=1)
    passed_mask = (block_sums % 2 == 1)
    passed_count = int(np.sum(passed_mask))
    pass_rate = passed_count / num_blocks

    check_state = EvidenceCheckState.NOT_TESTED
    if pass_rate >= 0.95:
        check_state = EvidenceCheckState.PASS
    elif pass_rate <= 0.70:
        check_state = EvidenceCheckState.FAIL
    else:
        check_state = EvidenceCheckState.NOT_TESTED

    return ParityResult(
        parity_type="odd",
        blocks_tested=num_blocks,
        blocks_passed=passed_count,
        pass_rate=pass_rate,
        check_state=check_state,
    )


def check_block_parity(
    bits: Union[np.ndarray, List[int]],
    rows: int = 8,
    cols: int = 8,
) -> ParityResult:
    """
    Check 2D block parity (rows x cols) where the last row and column are parity checks.
    """
    bits = np.asarray(bits, dtype=np.uint8).ravel()
    block_len = rows * cols
    num_blocks = len(bits) // block_len

    if num_blocks == 0:
        return ParityResult(parity_type="block_2d", check_state=EvidenceCheckState.NOT_TESTED)

    passed_count = 0
    for b in range(num_blocks):
        mat = bits[b * block_len:(b + 1) * block_len].reshape(rows, cols)
        row_parity_valid = np.all(np.sum(mat, axis=1) % 2 == 0)
        col_parity_valid = np.all(np.sum(mat, axis=0) % 2 == 0)
        if row_parity_valid and col_parity_valid:
            passed_count += 1

    pass_rate = passed_count / num_blocks
    check_state = EvidenceCheckState.PASS if pass_rate >= 0.90 else (
        EvidenceCheckState.FAIL if pass_rate <= 0.50 else EvidenceCheckState.NOT_TESTED
    )

    return ParityResult(
        parity_type=f"block_2d_{rows}x{cols}",
        blocks_tested=num_blocks,
        blocks_passed=passed_count,
        pass_rate=pass_rate,
        check_state=check_state,
    )
