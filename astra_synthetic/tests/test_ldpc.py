"""
Unit tests for ASTRA LDPC Systematic Block Encoder and Syndrome Checker.
Validates parity-check syndrome nullity H*c^T=0, clean systematic decoding, and block segmentation.
"""

import numpy as np
import pytest

from astra_synthetic.fec.ldpc import LDPCCode, LDPC_PROFILES, generate_systematic_ldpc_profile


class TestLDPCCode:
    """Test systematic LDPC encoding, syndrome verification, and block decoding."""

    def test_matrix_dimensions_and_syndrome(self):
        """TEST 9: LDPC valid syndrome."""
        ldpc = LDPCCode(profile="ldpc_n128_k64_r12")
        assert ldpc.n == 128
        assert ldpc.k == 64
        assert ldpc.H.shape == (64, 128)
        assert ldpc.G.shape == (64, 128)

        # Encode a single 64-bit block
        rng = np.random.default_rng(42)
        m = rng.integers(0, 2, size=64, dtype=np.uint8)
        codeword = (m @ ldpc.G) % 2

        assert len(codeword) == 128
        assert ldpc.is_valid_codeword(codeword) is True
        syndrome = ldpc.compute_syndrome(codeword)
        assert np.all(syndrome == 0)

    def test_corrupted_codeword_fails_syndrome(self):
        ldpc = LDPCCode(profile="ldpc_n128_k64_r12")
        m = np.ones(64, dtype=np.uint8)
        codeword = (m @ ldpc.G) % 2
        
        # Corrupt 1 bit
        corrupted = codeword.copy()
        corrupted[5] ^= 1
        assert ldpc.is_valid_codeword(corrupted) is False
        assert np.sum(ldpc.compute_syndrome(corrupted)) > 0

    def test_ldpc_clean_round_trip_and_multi_block(self):
        """TEST 10: LDPC clean round-trip with multi-block segmentation."""
        ldpc = LDPCCode(profile="ldpc_n128_k64_r12")
        rng = np.random.default_rng(100)
        # 150 bits -> ceil(150/64) = 3 blocks of 64 bits (total 192 bits input, 384 bits encoded)
        in_bits = rng.integers(0, 2, size=150, dtype=np.uint8)

        encoded, pad_bits, pad_len, block_meta, params = ldpc.encode(in_bits)
        assert len(block_meta) == 3
        assert len(encoded) == 3 * 128
        assert pad_len == (64 * 3) - 150  # 42 bits padding

        # Check all block syndromes are valid
        for b in block_meta:
            assert b["parity_check_valid"] is True
            assert b["syndrome_weight"] == 0

        # Reference decode
        decoded = ldpc.decode_reference(encoded, original_bit_length=150, block_boundaries=block_meta)
        assert np.array_equal(decoded, in_bits)
