"""
Reproducibility and determinism tests for ASTRA PayloadGenerator.
Validates that seed control produces bit-exact identical payloads across all modes.
"""

import numpy as np
import pytest

from astra_synthetic.payload.generator import PayloadGenerator


class TestReproducibility:
    """Test seed determinism and independence."""

    def test_same_seed_identical_output(self):
        """TEST 3: same seed → identical output"""
        gen1 = PayloadGenerator()
        gen2 = PayloadGenerator()

        p1 = gen1.generate(payload_type="random_bits", bit_length=2048, seed=12345)
        p2 = gen2.generate(payload_type="random_bits", bit_length=2048, seed=12345)

        assert np.array_equal(p1.payload_bits, p2.payload_bits)
        assert p1.payload_bytes == p2.payload_bytes
        assert p1.sha256 == p2.sha256
        assert p1.entropy_estimate == p2.entropy_estimate

    def test_different_seeds_different_output(self):
        """TEST 4: different seeds → different random output"""
        gen = PayloadGenerator()
        p1 = gen.generate(payload_type="random_bits", bit_length=2048, seed=111)
        p2 = gen.generate(payload_type="random_bits", bit_length=2048, seed=222)

        assert not np.array_equal(p1.payload_bits, p2.payload_bits)
        assert p1.sha256 != p2.sha256

    def test_biased_random_reproducibility(self):
        gen = PayloadGenerator()
        p1 = gen.generate(payload_type="biased_random", bit_length=1024, probability_one=0.08, seed=999)
        p2 = gen.generate(payload_type="biased_random", bit_length=1024, probability_one=0.08, seed=999)

        assert np.array_equal(p1.payload_bits, p2.payload_bits)
        assert p1.sha256 == p2.sha256

    def test_batch_reproducibility_with_master_seed(self):
        """TEST 15: batch generation reproducible with same master seed"""
        config = {
            "payload_generator": {
                "version": "1.0.0",
                "master_seed": 777,
                "length": {"mode": "random", "min_bits": 128, "max_bits": 512},
                "types": {
                    "random_bits": {"enabled": True, "probability": 0.5},
                    "repeated_pattern": {"enabled": True, "probability": 0.5, "patterns": ["10110011"]},
                },
            }
        }

        gen1 = PayloadGenerator(config)
        gen2 = PayloadGenerator(config)

        batch1 = gen1.generate_batch(count=50)
        batch2 = gen2.generate_batch(count=50)

        assert len(batch1) == len(batch2) == 50
        for b1, b2 in zip(batch1, batch2):
            assert b1.payload_type == b2.payload_type
            assert b1.bit_length == b2.bit_length
            assert np.array_equal(b1.payload_bits, b2.payload_bits)
            assert b1.sha256 == b2.sha256

    def test_generator_reset(self):
        gen = PayloadGenerator()
        p_initial = gen.generate(payload_type="random_bits", bit_length=512, seed=42)
        
        # Advance generator
        _ = gen.generate_batch(count=10)
        
        # Reset with seed 42
        gen.reset(seed=42)
        p_after_reset = gen.generate(payload_type="random_bits", bit_length=512, seed=42)
        
        assert np.array_equal(p_initial.payload_bits, p_after_reset.payload_bits)
