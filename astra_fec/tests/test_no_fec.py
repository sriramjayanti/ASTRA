"""
test_no_fec.py
Unit tests for NO_FEC uncoded pass-through handler.
"""

import unittest
import numpy as np
from astra_fec.src.no_fec import decode_no_fec


class TestNoFEC(unittest.TestCase):
    def test_no_fec_passthrough(self):
        bits = np.random.randint(0, 2, size=256, dtype=np.uint8)
        llrs = np.random.randn(256).astype(np.float32)
        res = decode_no_fec(bits, llrs)
        self.assertTrue(res.success)
        np.testing.assert_array_equal(res.decoded_bits, bits)
        np.testing.assert_array_almost_equal(res.decoded_soft_info, llrs)
        self.assertEqual(res.metrics["code_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
