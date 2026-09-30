"""
test_viterbi.py
Unit tests for Hard and Soft Viterbi decoders.
"""

import unittest
import numpy as np
from astra_fec.src.convolutional import encode_convolutional
from astra_fec.src.viterbi import viterbi_decode_hard, viterbi_decode_soft


class TestViterbi(unittest.TestCase):
    def test_viterbi_hard_and_soft(self):
        msg = np.random.randint(0, 2, size=128, dtype=np.uint8)
        encoded = encode_convolutional(msg, constraint_length=7, generators_octal=[171, 133], termination="terminated")
        
        # Hard Viterbi test
        dec_hard, pm_hard, _, term_hard = viterbi_decode_hard(encoded, constraint_length=7, generators_octal=[171, 133])
        self.assertTrue(term_hard)
        np.testing.assert_array_equal(dec_hard, msg)
        
        # Soft Viterbi test
        llrs = np.where(encoded == 0, 6.0, -6.0).astype(np.float32)
        dec_soft, pm_soft, _, term_soft = viterbi_decode_soft(llrs, constraint_length=7, generators_octal=[171, 133])
        self.assertTrue(term_soft)
        np.testing.assert_array_equal(dec_soft, msg)


if __name__ == "__main__":
    unittest.main()
