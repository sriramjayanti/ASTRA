"""
test_helical.py
Unit tests for helical / diagonal permutation interleaver family.
"""

import unittest
import numpy as np
from astra_interleaver.src.helical import (
    create_helical_mapping,
    interleave_helical,
    deinterleave_helical
)


class TestHelicalInterleaver(unittest.TestCase):
    def test_helical_mapping_bijection(self):
        for r in [4, 8, 16]:
            for c in [8, 16]:
                for s in [1, 2, 3]:
                    mapping = create_helical_mapping(r, c, s, "row_diagonal")
                    self.assertTrue(mapping.is_valid())
                    self.assertEqual(len(np.unique(mapping.inverse_indices)), r * c)

    def test_helical_roundtrip(self):
        for orient in ["row_diagonal", "diagonal_row"]:
            raw_bits = np.random.randint(0, 2, size=512, dtype=np.uint8)
            llrs = np.random.randn(512).astype(np.float32)
            int_b, int_l = interleave_helical(raw_bits, llrs, rows=16, cols=16, step=3, orientation=orient)
            deint_b, deint_l, _, rem = deinterleave_helical(int_b, int_l, rows=16, cols=16, step=3, orientation=orient)
            self.assertEqual(rem, 0)
            np.testing.assert_array_equal(deint_b, raw_bits)
            np.testing.assert_array_almost_equal(deint_l, llrs)


if __name__ == "__main__":
    unittest.main()
