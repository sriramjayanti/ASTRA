"""
test_identity.py
Unit tests for Identity / NO_INTERLEAVER pass-through family.
"""

import unittest
import numpy as np
from astra_interleaver.src.identity import (
    create_identity_mapping,
    deinterleave_identity,
    interleave_identity
)
from astra_interleaver.src.models import InterleaverFamily


class TestIdentityInterleaver(unittest.TestCase):
    def test_identity_mapping(self):
        mapping = create_identity_mapping(128)
        self.assertEqual(mapping.length, 128)
        self.assertEqual(mapping.family, InterleaverFamily.IDENTITY.value)
        self.assertTrue(mapping.is_valid())
        np.testing.assert_array_equal(mapping.forward_indices, np.arange(128))
        np.testing.assert_array_equal(mapping.inverse_indices, np.arange(128))

    def test_identity_roundtrip(self):
        bits = np.random.randint(0, 2, size=256, dtype=np.uint8)
        llrs = np.random.randn(256).astype(np.float32)
        int_bits, int_llrs = interleave_identity(bits, llrs)
        deint_bits, deint_llrs, mapping = deinterleave_identity(int_bits, int_llrs)
        
        np.testing.assert_array_equal(deint_bits, bits)
        np.testing.assert_array_almost_equal(deint_llrs, llrs)


if __name__ == "__main__":
    unittest.main()
