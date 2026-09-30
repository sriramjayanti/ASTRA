"""
test_parity.py
Unit tests for even, odd, and 2D block parity checkers.
"""

import pytest
import numpy as np
from astra_validation.src.parity import check_even_parity, check_odd_parity, check_block_parity
from astra_validation.src.models import EvidenceCheckState


def test_even_parity_pass():
    # 4 blocks of 8 bits with even parity
    blocks = []
    for _ in range(4):
        b = np.array([1, 0, 1, 0, 1, 1, 0, 0], dtype=np.uint8) # 4 ones -> sum 4 -> even
        blocks.append(b)
    stream = np.concatenate(blocks)
    res = check_even_parity(stream, block_size=8)
    assert res.blocks_tested == 4
    assert res.blocks_passed == 4
    assert res.pass_rate == 1.0
    assert res.check_state == EvidenceCheckState.PASS


def test_even_parity_fail():
    # 4 blocks of 8 bits with odd parity
    blocks = []
    for _ in range(4):
        b = np.array([1, 0, 1, 0, 1, 0, 0, 0], dtype=np.uint8) # 3 ones -> sum 3 -> odd
        blocks.append(b)
    stream = np.concatenate(blocks)
    res = check_even_parity(stream, block_size=8)
    assert res.blocks_passed == 0
    assert res.pass_rate == 0.0
    assert res.check_state == EvidenceCheckState.FAIL


def test_odd_parity_pass():
    blocks = []
    for _ in range(4):
        b = np.array([1, 0, 1, 0, 1, 0, 0, 0], dtype=np.uint8) # 3 ones -> sum 3 -> odd
        blocks.append(b)
    stream = np.concatenate(blocks)
    res = check_odd_parity(stream, block_size=8)
    assert res.blocks_tested == 4
    assert res.blocks_passed == 4
    assert res.pass_rate == 1.0
    assert res.check_state == EvidenceCheckState.PASS


def test_block_2d_parity():
    # 8x8 matrix where row and col parities are all even
    mat = np.zeros((8, 8), dtype=np.uint8)
    mat[0, :2] = [1, 1]
    mat[1, :2] = [1, 1]
    res = check_block_parity(mat.ravel(), rows=8, cols=8)
    assert res.blocks_passed == 1
    assert res.check_state == EvidenceCheckState.PASS
