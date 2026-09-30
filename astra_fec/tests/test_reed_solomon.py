"""
test_reed_solomon.py
Unit tests for Reed-Solomon codec and error correction.
"""

import unittest
import numpy as np
from astra_fec.src.reed_solomon import ReedSolomonCodec, decode_reed_solomon_profile
from astra_fec.src.profiles import get_profile_registry
from astra_fec.src.utils import generate_synthetic_fec_stream


class TestReedSolomon(unittest.TestCase):
    def test_rs_codec_correction(self):
        codec = ReedSolomonCodec(n=255, k=223, m=8, prim_poly=0x187, fcr=112)
        msg_syms = list(np.random.randint(0, 256, size=223))
        cw = codec.encode(msg_syms)
        
        # Corrupt 8 symbols (well within t=16)
        corrupted = list(cw)
        for i in range(8):
            corrupted[i * 20] ^= 0x3C
            
        success, decoded, n_err = codec.decode_codeword(corrupted)
        self.assertTrue(success)
        self.assertEqual(n_err, 8)
        self.assertEqual(decoded, msg_syms)


if __name__ == "__main__":
    unittest.main()
