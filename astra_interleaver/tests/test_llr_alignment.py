"""
test_llr_alignment.py
Unit tests verifying strict identical permutation alignment between hard bits and soft LLRs.
"""

import unittest
import numpy as np
from astra_interleaver.src.permutation import apply_inverse_mapping
from astra_interleaver.src.block import interleave_block, deinterleave_block
from astra_interleaver.src.helical import interleave_helical, deinterleave_helical
from astra_interleaver.src.pseudo_random import interleave_pseudorandom, deinterleave_pseudorandom


class TestLLRAlignment(unittest.TestCase):
    def test_bit_llr_coupling_preservation(self):
        # Create paired bit and unique LLR signatures
        N = 256
        bits = np.random.randint(0, 2, size=N, dtype=np.uint8)
        # Each LLR encodes its original index
        llrs = np.array([float(i) * (-1.0 if bits[i] == 1 else 1.0) for i in range(N)], dtype=np.float32)
        
        # Test across block, helical, PR
        for family in ["block", "helical", "pr"]:
            if family == "block":
                int_b, int_l = interleave_block(bits, llrs, rows=16, cols=16)
                deint_b, deint_l, _, _ = deinterleave_block(int_b, int_l, rows=16, cols=16)
            elif family == "helical":
                int_b, int_l = interleave_helical(bits, llrs, rows=16, cols=16, step=3)
                deint_b, deint_l, _, _ = deinterleave_helical(int_b, int_l, rows=16, cols=16, step=3)
            else:
                int_b, int_l = interleave_pseudorandom(bits, llrs, length=256, seed=42)
                deint_b, deint_l, _, _ = deinterleave_pseudorandom(int_b, int_l, length=256, seed=42)
                
            np.testing.assert_array_equal(deint_b, bits)
            np.testing.assert_array_almost_equal(deint_l, llrs)
            
            # Check intermediate interleaved state also maintains exact bit-sign to LLR match
            for i in range(N):
                b_val = int_b[i]
                l_val = int_l[i]
                if b_val == 0:
                    self.assertGreaterEqual(l_val, 0.0)
                else:
                    self.assertLessEqual(l_val, 0.0)


if __name__ == "__main__":
    unittest.main()
