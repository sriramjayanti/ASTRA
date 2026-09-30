"""
test_pseudo_random.py
Unit tests for pseudo-random permutation interleaver family.
"""

import unittest
import numpy as np
from astra_interleaver.src.pseudo_random import (
    create_pseudorandom_mapping,
    interleave_pseudorandom,
    deinterleave_pseudorandom
)


class TestPseudoRandomInterleaver(unittest.TestCase):
    def test_pseudorandom_mapping(self):
        for algo in ["pcg64", "fisher_yates"]:
            mapping = create_pseudorandom_mapping(length=256, seed=42, algorithm=algo)
            self.assertTrue(mapping.is_valid())
            self.assertEqual(len(mapping.inverse_indices), 256)

    def test_pseudorandom_roundtrip(self):
        raw_bits = np.random.randint(0, 2, size=1024, dtype=np.uint8)
        llrs = np.random.randn(1024).astype(np.float32)
        int_b, int_l = interleave_pseudorandom(raw_bits, llrs, length=512, seed=1337)
        deint_b, deint_l, _, rem = deinterleave_pseudorandom(int_b, int_l, length=512, seed=1337)
        self.assertEqual(rem, 0)
        np.testing.assert_array_equal(deint_b, raw_bits)
        np.testing.assert_array_almost_equal(deint_l, llrs)


if __name__ == "__main__":
    unittest.main()
