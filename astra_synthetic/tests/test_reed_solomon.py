"""
Unit tests for ASTRA Reed-Solomon Block Encoder and Decoder.
Validates clean round-trip, multi-block segmentation, correctable symbol error recovery, and uncorrectable errors.
"""

import numpy as np
import pytest
import reedsolo

from astra_synthetic.payload.models import text_to_bits, bits_to_bytes
from astra_synthetic.fec.reed_solomon import ReedSolomonCode


class TestReedSolomonCode:
    """Test Reed-Solomon encoding, padding, and correction capabilities."""

    def test_rs_clean_round_trip(self):
        """TEST 5: RS clean round-trip with byte alignment."""
        rs = ReedSolomonCode(n=255, k=223)
        rng = np.random.default_rng(42)
        # 1128 bits (141 bytes) -> fits in 1 RS block of k=223 bytes
        in_bits = rng.integers(0, 2, size=1128, dtype=np.uint8)

        encoded, pad_bits, pad_len, block_meta, params = rs.encode(in_bits)
        assert len(encoded) == 255 * 8
        assert len(block_meta) == 1

        decoded = rs.decode(encoded, original_bit_length=1128, block_boundaries=block_meta)
        assert np.array_equal(decoded, in_bits)

    def test_rs_multi_block_segmentation(self):
        """Test input larger than k symbols requiring multi-block segmentation."""
        rs = ReedSolomonCode(n=64, k=48)
        rng = np.random.default_rng(999)
        # 100 bytes (800 bits) -> ceil(100/48) = 3 blocks
        in_bits = rng.integers(0, 2, size=800, dtype=np.uint8)

        encoded, _, _, block_meta, _ = rs.encode(in_bits)
        assert len(block_meta) == 3
        assert len(encoded) == 3 * 64 * 8

        decoded = rs.decode(encoded, original_bit_length=800, block_boundaries=block_meta)
        assert np.array_equal(decoded, in_bits)

    def test_rs_correctable_symbol_errors(self):
        """TEST 6: RS corrects up to t symbol errors."""
        # RS(64, 48) has 16 parity symbols, t = 8 symbol errors
        rs = ReedSolomonCode(n=64, k=48)
        rng = np.random.default_rng(555)
        in_bits = rng.integers(0, 2, size=48 * 8, dtype=np.uint8)

        encoded, _, _, block_meta, _ = rs.encode(in_bits)

        # Corrupt 5 distinct bytes (within t=8 capacity) by flipping 1 bit in each byte
        corrupted = encoded.copy()
        for byte_idx in [2, 10, 25, 40, 55]:
            corrupted[byte_idx * 8] ^= 1

        decoded = rs.decode(corrupted, original_bit_length=48 * 8, block_boundaries=block_meta)
        assert np.array_equal(decoded, in_bits)

    def test_rs_uncorrectable_errors_raise(self):
        """TEST 7: RS uncorrectable error behavior (> t errors)."""
        rs = ReedSolomonCode(n=64, k=48)  # t = 8
        in_bits = np.ones(48 * 8, dtype=np.uint8)

        encoded, _, _, block_meta, _ = rs.encode(in_bits)

        # Corrupt 15 distinct bytes (> 8) by flipping 1 bit in each
        corrupted = encoded.copy()
        for byte_idx in range(15):
            corrupted[byte_idx * 8] ^= 1

        with pytest.raises(reedsolo.ReedSolomonError):
            rs.decode(corrupted, original_bit_length=48 * 8, block_boundaries=block_meta)
