"""
Unit tests for ASTRA Sync Word Generator.
Validates fixed hex/binary sync words, seed-reproducible random sync generation, and pool selection.
"""

import numpy as np
import pytest

from astra_synthetic.payload.models import hex_to_bits
from astra_synthetic.framing.sync import SyncWordGenerator


class TestSyncGenerator:
    """Test sync word creation across all modes."""

    def test_fixed_hex_sync(self):
        """TEST 12: fixed sync works"""
        gen = SyncWordGenerator({"mode": "fixed", "format": "hex", "value": "1ACFFC1D"})
        bits, val, length, mode = gen.generate()

        assert mode == "fixed"
        assert length == 32
        assert val == "1ACFFC1D"
        expected_bits = hex_to_bits("1ACFFC1D")
        assert np.array_equal(bits, expected_bits)

    def test_fixed_binary_sync(self):
        gen = SyncWordGenerator({"mode": "fixed", "format": "binary", "value": "1010111100001101"})
        bits, val, length, mode = gen.generate()

        assert length == 16
        assert val == "1010111100001101"
        assert np.array_equal(bits, np.array([1,0,1,0,1,1,1,1,0,0,0,0,1,1,0,1], dtype=np.uint8))

    def test_random_sync_reproducibility(self):
        """TEST 13: random sync reproducible with same seed"""
        gen = SyncWordGenerator({"mode": "random", "length_bits": 48})
        
        bits1, val1, len1, mode1 = gen.generate(seed=777)
        bits2, val2, len2, mode2 = gen.generate(seed=777)
        bits_diff, _, _, _ = gen.generate(seed=888)

        assert len1 == 48
        assert np.array_equal(bits1, bits2)
        assert val1 == val2
        assert not np.array_equal(bits1, bits_diff)

    def test_pool_sync_selection(self):
        pool = ["1ACFFC1D", "D391D391", "A5A5A5A5"]
        gen = SyncWordGenerator({"mode": "pool", "format": "hex", "values": pool})
        
        bits, val, length, mode = gen.generate(seed=42)
        assert mode == "pool"
        assert val in pool
        assert length == 32
        assert np.array_equal(bits, hex_to_bits(val))

    def test_invalid_sync_config(self):
        gen = SyncWordGenerator({"mode": "invalid_mode"})
        with pytest.raises(ValueError, match="Unknown sync mode"):
            gen.generate()
