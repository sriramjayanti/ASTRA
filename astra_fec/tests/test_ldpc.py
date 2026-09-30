"""
test_ldpc.py
Unit tests for LDPC codec and Min-Sum decoding.
"""

import unittest
import numpy as np
from astra_fec.src.profiles import get_profile_registry
from astra_fec.src.ldpc import construct_qc_ldpc_matrix, LDPCCodec, decode_ldpc_profile
from astra_fec.src.utils import generate_synthetic_fec_stream


class TestLDPC(unittest.TestCase):
    def test_ldpc_codec_roundtrip(self):
        H = construct_qc_ldpc_matrix(n=128, k=64, block_size=16)
        codec = LDPCCodec(H, alpha=0.80, max_iter=30)
        
        info = np.random.randint(0, 2, size=64, dtype=np.uint8)
        cw = codec.encode(info)
        
        # Verify H * cw == 0 mod 2
        syn = np.dot(H, cw.astype(np.int32)) % 2
        self.assertEqual(np.sum(syn), 0)
        
        # Decode soft LLRs
        llrs = np.where(cw == 0, 6.0, -6.0).astype(np.float32)
        conv, dec_bits, iters, _, _, _ = codec.decode_min_sum(llrs)
        self.assertTrue(conv)
        np.testing.assert_array_equal(dec_bits[:64], info)


if __name__ == "__main__":
    unittest.main()
