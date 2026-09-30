"""
Unit tests for Block Interleaver and Deinterleaver in ASTRA Synthetic Engine 4.
Validates matrix permutations, row-major / column-major reads, padding handling,
rectangular blocks, and exact round-trip reconstruction.
"""

import numpy as np
import pytest

from astra_synthetic.interleaving.block import (
    BlockInterleaver,
    compute_block_permutation,
    deinterleave_block,
    interleave_block,
)
from astra_synthetic.interleaving.padding import pad_to_multiple, unpad
from astra_synthetic.interleaving.permutation import (
    apply_permutation,
    invert_permutation,
    validate_permutation,
)


class TestBlockInterleaver:
    """Test suite for block interleaving logic."""

    def test_known_small_example_2x3(self):
        """
        Input: [1, 0, 1, 0, 1, 1]
        Matrix (2 rows, 3 cols):
        Row 0: [1, 0, 1]
        Row 1: [0, 1, 1]
        Read column-wise:
        Col 0: [1, 0]
        Col 1: [0, 1]
        Col 2: [1, 1]
        Output: [1, 0, 0, 1, 1, 1]
        """
        inp = np.array([1, 0, 1, 0, 1, 1], dtype=np.uint8)
        perm, inv_perm = compute_block_permutation(rows=2, columns=3, write_order="row_major", read_order="column_major")
        # Indices in row major:
        # [0, 1, 2]
        # [3, 4, 5]
        # Read column major: 0, 3, 1, 4, 2, 5
        np.testing.assert_array_equal(perm, [0, 3, 1, 4, 2, 5])
        validate_permutation(perm, 6)

        interleaved = apply_permutation(inp, perm)
        expected = np.array([1, 0, 0, 1, 1, 1], dtype=np.uint8)
        np.testing.assert_array_equal(interleaved, expected)

        # Invert
        recovered = apply_permutation(interleaved, inv_perm)
        np.testing.assert_array_equal(recovered, inp)

    def test_block_8x8_square_roundtrip(self):
        """Test square 8x8 block interleaver round-trip."""
        rng = np.random.default_rng(12345)
        inp = rng.integers(0, 2, size=64, dtype=np.uint8)

        interleaved, pad_bits, boundaries = interleave_block(
            inp, rows=8, cols=8, write_order="row_major", read_order="column_major"
        )
        assert len(interleaved) == 64
        assert len(pad_bits) == 0
        assert len(boundaries) == 1

        recovered = deinterleave_block(
            interleaved,
            rows=8,
            cols=8,
            write_order="row_major",
            read_order="column_major",
            original_bit_length=64,
        )
        np.testing.assert_array_equal(recovered, inp)

    def test_rectangular_block_16x32(self):
        """Test rectangular 16x32 block interleaver with multi-block data."""
        block_size = 16 * 32  # 512 bits
        rng = np.random.default_rng(999)
        inp = rng.integers(0, 2, size=1024, dtype=np.uint8)  # Exactly 2 blocks

        interleaved, pad_bits, boundaries = interleave_block(
            inp, rows=16, cols=32, write_order="row_major", read_order="column_major"
        )
        assert len(interleaved) == 1024
        assert len(pad_bits) == 0
        assert len(boundaries) == 2

        recovered = deinterleave_block(
            interleaved,
            rows=16,
            cols=32,
            write_order="row_major",
            read_order="column_major",
            original_bit_length=1024,
        )
        np.testing.assert_array_equal(recovered, inp)

    def test_block_padding_and_removal(self):
        """Test block interleaver when input size is not a multiple of block size."""
        # 100 bits into 8x8 (64 bits block) -> 2 blocks = 128 bits total (28 bits padding)
        rng = np.random.default_rng(42)
        inp = rng.integers(0, 2, size=100, dtype=np.uint8)

        for pad_mode in ["zeros", "ones", "random"]:
            interleaved, pad_bits, boundaries = interleave_block(
                inp,
                rows=8,
                cols=8,
                write_order="row_major",
                read_order="column_major",
                padding_mode=pad_mode,
                seed=42,
            )
            assert len(interleaved) == 128
            assert len(pad_bits) == 28
            assert len(boundaries) == 2
            assert boundaries[0]["input_length"] == 64
            assert boundaries[1]["input_length"] == 36
            assert boundaries[1]["padded_length"] == 64

            recovered = deinterleave_block(
                interleaved,
                rows=8,
                cols=8,
                write_order="row_major",
                read_order="column_major",
                original_bit_length=100,
            )
            np.testing.assert_array_equal(recovered, inp)

    def test_reverse_combination_col_write_row_read(self):
        """Test column-major write and row-major read order."""
        perm, inv_perm = compute_block_permutation(rows=3, columns=4, write_order="column_major", read_order="row_major")
        validate_permutation(perm, 12)
        rng = np.random.default_rng(777)
        inp = rng.integers(0, 2, size=12, dtype=np.uint8)

        interleaved, _, _ = interleave_block(
            inp, rows=3, cols=4, write_order="column_major", read_order="row_major"
        )
        recovered = deinterleave_block(
            interleaved,
            rows=3,
            cols=4,
            write_order="column_major",
            read_order="row_major",
            original_bit_length=12,
        )
        np.testing.assert_array_equal(recovered, inp)

    def test_invalid_block_dimensions(self):
        """Verify errors on invalid rows/columns."""
        with pytest.raises(ValueError):
            compute_block_permutation(rows=0, columns=8)
        with pytest.raises(ValueError):
            compute_block_permutation(rows=8, columns=-1)
        with pytest.raises(ValueError):
            compute_block_permutation(rows=8, columns=8, write_order="diagonal")
