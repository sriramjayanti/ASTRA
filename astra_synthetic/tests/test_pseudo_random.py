"""
Unit tests for Pseudo-Random Interleaver and Deinterleaver in ASTRA Synthetic Engine 4.
Validates deterministic seeded permutations, inverse permutations, seed uniqueness,
multi-block segmenting, padding handling, and exact round-trip inversion.
"""

import numpy as np
import pytest

from astra_synthetic.interleaving.permutation import (
    apply_permutation,
    invert_permutation,
    validate_permutation,
)
from astra_synthetic.interleaving.pseudo_random import (
    PseudoRandomInterleaver,
    deinterleave_pseudo_random,
    generate_pseudo_random_permutation,
    interleave_pseudo_random,
)


class TestPseudoRandomInterleaver:
    """Test suite for pseudo-random interleaving logic."""

    def test_same_seed_same_permutation(self):
        """Verify identical seed + block_size reproduces the exact same permutation."""
        perm1, _, _ = generate_pseudo_random_permutation(block_size=256, seed=42)
        perm2, _, _ = generate_pseudo_random_permutation(block_size=256, seed=42)
        np.testing.assert_array_equal(perm1, perm2)

    def test_different_seed_different_permutation(self):
        """Verify different seeds produce distinct permutations."""
        perm1, _, _ = generate_pseudo_random_permutation(block_size=256, seed=42)
        perm2, _, _ = generate_pseudo_random_permutation(block_size=256, seed=999)
        assert not np.array_equal(perm1, perm2)

    def test_permutation_validity(self):
        """Verify permutation is a bijection on range(0, N)."""
        perm, inv_perm, perm_sha = generate_pseudo_random_permutation(block_size=512, seed=123)
        assert len(perm) == 512
        validate_permutation(perm, 512)
        assert sorted(perm.tolist()) == list(range(512))

    def test_inverse_permutation_math(self):
        """Verify output[inverse_permutation] recovers original input for arbitrary random vectors."""
        block_size = 128
        perm, inv_perm, perm_sha = generate_pseudo_random_permutation(block_size=block_size, seed=555)
        validate_permutation(inv_perm, block_size)

        rng = np.random.default_rng(888)
        inp = rng.integers(0, 2, size=block_size, dtype=np.uint8)

        interleaved = apply_permutation(inp, perm)
        recovered = apply_permutation(interleaved, inv_perm)
        np.testing.assert_array_equal(recovered, inp)

    def test_pseudo_random_roundtrip_pr_256(self):
        """Test complete round-trip through PseudoRandomInterleaver."""
        rng = np.random.default_rng(777)
        inp = rng.integers(0, 2, size=256, dtype=np.uint8)

        interleaved, pad_bits, boundaries, perm = interleave_pseudo_random(inp, block_size=256, seed=42)
        assert len(interleaved) == 256
        assert len(pad_bits) == 0

        recovered = deinterleave_pseudo_random(interleaved, block_size=256, seed=42, original_bit_length=256)
        np.testing.assert_array_equal(recovered, inp)

    def test_pseudo_random_multi_block_with_padding(self):
        """Test multi-block input (e.g. 700 bits with block_size=256 -> 3 blocks = 768 bits)."""
        rng = np.random.default_rng(321)
        inp = rng.integers(0, 2, size=700, dtype=np.uint8)

        interleaved, pad_bits, boundaries, perm = interleave_pseudo_random(
            inp, block_size=256, seed=100, padding_mode="zeros"
        )
        assert len(interleaved) == 768
        assert len(pad_bits) == 68
        assert len(boundaries) == 3
        assert boundaries[0]["input_length"] == 256
        assert boundaries[1]["input_length"] == 256
        assert boundaries[2]["input_length"] == 188
        assert boundaries[2]["padded_length"] == 256

        recovered = deinterleave_pseudo_random(interleaved, block_size=256, seed=100, original_bit_length=700)
        np.testing.assert_array_equal(recovered, inp)

    def test_invalid_parameters(self):
        """Verify errors on non-positive block sizes."""
        with pytest.raises(ValueError):
            generate_pseudo_random_permutation(block_size=0, seed=1)
        with pytest.raises(ValueError):
            generate_pseudo_random_permutation(block_size=-10, seed=1)
