"""
test_block.py
Unit tests for rectangular block interleaver/deinterleaver family.
"""

import unittest
import numpy as np
from astra_interleaver.src.block import (
    create_block_mapping,
    interleave_block,
    deinterleave_block
)


class TestBlockInterleaver(unittest.TestCase):
    def test_block_mapping_validity(self):
        mapping = create_block_mapping(rows=8, cols=16, orientation="row_to_column")
        self.assertTrue(mapping.is_valid())
        self.assertEqual(mapping.length, 128)

    def test_block_roundtrip_all_orientations(self):
        for orient in ["row_to_column", "column_to_row"]:
            for r, c in [(4, 8), (8, 16), (16, 32)]:
                N = r * c * 3  # 3 full blocks
                raw_bits = np.random.randint(0, 2, size=N, dtype=np.uint8)
                llrs = np.random.randn(N).astype(np.float32)
                
                int_b, int_l = interleave_block(raw_bits, llrs, r, c, orient)
                deint_b, deint_l, mapping, rem = deinterleave_block(int_b, int_l, r, c, orient)
                
                self.assertEqual(rem, 0)
                np.testing.assert_array_equal(deint_b, raw_bits)
                np.testing.assert_array_almost_equal(deint_l, llrs)


if __name__ == "__main__":
    unittest.main()
