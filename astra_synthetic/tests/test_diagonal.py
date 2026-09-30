"""
Unit tests for Diagonal Matrix Interleaver and Deinterleaver in ASTRA Synthetic Engine 4.
Validates top-left to bottom-right anti-diagonal traversal, rectangular and square matrices,
multi-block segmenting, padding removal, and exact round-trip inversion.
"""

import numpy as np
import pytest

from astra_synthetic.interleaving.diagonal import (
    DiagonalInterleaver,
    compute_diagonal_permutation,
    deinterleave_diagonal,
    interleave_diagonal,
)
from astra_synthetic.interleaving.permutation import (
    apply_permutation,
    invert_permutation,
    validate_permutation,
)


class TestDiagonalInterleaver:
    """Test suite for diagonal interleaving logic."""

    def test_known_small_example_3x3(self):
        """
        Matrix (3x3):
        0 1 2
        3 4 5
        6 7 8

        Diagonals (s = r + c):
        s=0: (0,0) -> 0
        s=1: (0,1)->1, (1,0)->3
        s=2: (0,2)->2, (1,1)->4, (2,0)->6
        s=3: (1,2)->5, (2,1)->7
        s=4: (2,2)->8

        Permutation: [0, 1, 3, 2, 4, 6, 5, 7, 8]
        """
        perm, inv_perm = compute_diagonal_permutation(rows=3, columns=3, direction="top_left_to_bottom_right")
        expected_perm = [0, 1, 3, 2, 4, 6, 5, 7, 8]
        np.testing.assert_array_equal(perm, expected_perm)
        validate_permutation(perm, 9)

        # Invert permutation
        inp = np.array([1, 0, 1, 1, 1, 0, 0, 0, 1], dtype=np.uint8)
        interleaved = apply_permutation(inp, perm)
        recovered = apply_permutation(interleaved, inv_perm)
        np.testing.assert_array_equal(recovered, inp)

    def test_known_example_4x4(self):
        """Test 4x4 matrix top-left to bottom-right traversal."""
        perm, inv_perm = compute_diagonal_permutation(rows=4, columns=4, direction="top_left_to_bottom_right")
        assert len(perm) == 16
        validate_permutation(perm, 16)
        # First element must be (0,0)=0, last element must be (3,3)=15
        assert perm[0] == 0
        assert perm[-1] == 15

    def test_diagonal_roundtrip_8x8(self):
        """Test round-trip reconstruction on 8x8 diagonal interleaver."""
        rng = np.random.default_rng(456)
        inp = rng.integers(0, 2, size=64, dtype=np.uint8)

        interleaved, pad_bits, boundaries = interleave_diagonal(
            inp, rows=8, cols=8, direction="top_left_to_bottom_right"
        )
        assert len(interleaved) == 64
        assert len(pad_bits) == 0

        recovered = deinterleave_diagonal(
            interleaved, rows=8, cols=8, direction="top_left_to_bottom_right", original_bit_length=64
        )
        np.testing.assert_array_equal(recovered, inp)

    def test_diagonal_multi_block_with_padding(self):
        """Test 16x16 diagonal interleaver with 600 input bits (requires 3 blocks of 256 bits = 768 bits)."""
        rng = np.random.default_rng(789)
        inp = rng.integers(0, 2, size=600, dtype=np.uint8)

        interleaved, pad_bits, boundaries = interleave_diagonal(
            inp, rows=16, cols=16, direction="top_left_to_bottom_right", padding_mode="zeros"
        )
        assert len(interleaved) == 768
        assert len(pad_bits) == 168
        assert len(boundaries) == 3

        recovered = deinterleave_diagonal(
            interleaved, rows=16, cols=16, direction="top_left_to_bottom_right", original_bit_length=600
        )
        np.testing.assert_array_equal(recovered, inp)

    def test_invalid_diagonal_direction(self):
        """Verify errors on unsupported diagonal direction."""
        with pytest.raises(ValueError):
            compute_diagonal_permutation(rows=8, columns=8, direction="spiral")
