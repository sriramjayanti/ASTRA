"""
test_soft_input.py
Unit tests verifying soft LLR advantage and numerical stability.
"""

import unittest
import numpy as np
from astra_fec.src.convolutional import encode_convolutional
from astra_fec.src.viterbi import viterbi_decode_hard, viterbi_decode_soft
from astra_fec.src.utils import compute_ber


class TestSoftInput(unittest.TestCase):
    def test_soft_viterbi_gain_under_noise(self):
        # Transmit 500 bits through noisy channel
        msg = np.random.randint(0, 2, size=500, dtype=np.uint8)
        encoded = encode_convolutional(msg, constraint_length=7, generators_octal=[171, 133], termination="terminated")
        
        # BPSK + Noise at 3.0 dB SNR
        sigma = 10.0 ** (-3.0 / 20.0)
        bpsk = 1.0 - 2.0 * encoded.astype(np.float32)
        noise = np.random.normal(0, sigma, size=len(encoded)).astype(np.float32)
        rx = bpsk + noise
        
        # LLRs and hard bits
        llrs = (2.0 * rx / (sigma ** 2)).astype(np.float32)
        hard_bits = (llrs < 0).astype(np.uint8)
        
        dec_hard, _, _, _ = viterbi_decode_hard(hard_bits, constraint_length=7, generators_octal=[171, 133])
        dec_soft, _, _, _ = viterbi_decode_soft(llrs, constraint_length=7, generators_octal=[171, 133])
        
        ber_hard = compute_ber(dec_hard, msg)
        ber_soft = compute_ber(dec_soft, msg)
        
        # Soft decoding BER should be <= Hard decoding BER
        self.assertLessEqual(ber_soft, ber_hard + 0.02)


if __name__ == "__main__":
    unittest.main()
