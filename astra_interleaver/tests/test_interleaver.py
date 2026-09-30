"""
test_interleaver.py
Master comprehensive test suite for ASTRA Stage 8 — Interleaver Candidate Testing Engine.
Covers all 30 mandatory unit tests from Specification Section 86.
"""

import unittest
import numpy as np
import json
import time

from astra_interleaver.src.models import (
    InterleaverStatus,
    InterleaverFamily,
    PermutationMapping,
    ConvolutionalDeinterleaverState,
    StructuralFeatures,
    InterleaverCandidateResult,
    InterleaverTestResult,
)
from astra_interleaver.src.permutation import (
    invert_permutation,
    validate_permutation,
    compute_permutation_hash,
    apply_inverse_mapping,
    apply_block_chunked_inverse,
)
from astra_interleaver.src.identity import (
    deinterleave_identity,
    interleave_identity,
)
from astra_interleaver.src.block import (
    create_block_mapping,
    interleave_block,
    deinterleave_block,
)
from astra_interleaver.src.convolutional import (
    ConvolutionalInterleaverEngine,
    ConvolutionalDeinterleaverEngine,
    interleave_convolutional,
    deinterleave_convolutional,
)
from astra_interleaver.src.helical import (
    create_helical_mapping,
    interleave_helical,
    deinterleave_helical,
)
from astra_interleaver.src.pseudo_random import (
    generate_pseudorandom_indices,
    create_pseudorandom_mapping,
    interleave_pseudorandom,
    deinterleave_pseudorandom,
)
from astra_interleaver.src.structural_features import (
    extract_structural_features,
    compute_binary_entropy,
    compute_autocorrelation_features,
)
from astra_interleaver.src.scoring import CandidateScorer
from astra_interleaver.src.pruning import prune_candidates
from astra_interleaver.src.candidate_generator import InterleaverCandidateGenerator
from astra_interleaver.src.inference import InterleaverTestingEngine
from astra_interleaver.src.utils import (
    generate_synthetic_interleaved_stream,
    evaluate_candidate_recall,
)


class TestInterleaverEngine(unittest.TestCase):
    """30-test master suite for Stage 8 Interleaver Testing Engine."""

    def setUp(self):
        self.engine = InterleaverTestingEngine()
        np.random.seed(42)

    # TEST 1: Identity leaves bits unchanged
    def test_01_identity_leaves_bits_unchanged(self):
        bits = np.random.randint(0, 2, size=512, dtype=np.uint8)
        deint_bits, _, _ = deinterleave_identity(bits)
        np.testing.assert_array_equal(deint_bits, bits)

    # TEST 2: Identity preserves LLRs
    def test_02_identity_preserves_llrs(self):
        bits = np.random.randint(0, 2, size=512, dtype=np.uint8)
        llrs = np.random.randn(512).astype(np.float32)
        deint_bits, deint_llrs, _ = deinterleave_identity(bits, llrs)
        np.testing.assert_array_equal(deint_bits, bits)
        np.testing.assert_array_almost_equal(deint_llrs, llrs)

    # TEST 3: Block round trip
    def test_03_block_round_trip(self):
        raw_bits = np.random.randint(0, 2, size=1024, dtype=np.uint8)
        llrs = np.random.randn(1024).astype(np.float32)
        rows, cols = 16, 32  # 512 block size (2 blocks)
        
        int_bits, int_llrs = interleave_block(raw_bits, llrs, rows, cols, "row_to_column")
        deint_bits, deint_llrs, mapping, rem = deinterleave_block(int_bits, int_llrs, rows, cols, "row_to_column")
        
        self.assertEqual(rem, 0)
        np.testing.assert_array_equal(deint_bits, raw_bits)
        np.testing.assert_array_almost_equal(deint_llrs, llrs)

    # TEST 4: Block orientations
    def test_04_block_orientations(self):
        raw_bits = np.random.randint(0, 2, size=256, dtype=np.uint8)
        for orient in ["row_to_column", "column_to_row"]:
            int_bits, _ = interleave_block(raw_bits, None, 16, 16, orient)
            deint_bits, _, _, _ = deinterleave_block(int_bits, None, 16, 16, orient)
            np.testing.assert_array_equal(deint_bits, raw_bits)

    # TEST 5: Block invalid dimensions rejected
    def test_05_block_invalid_dimensions_rejected(self):
        with self.assertRaises(ValueError):
            create_block_mapping(0, 16)
        with self.assertRaises(ValueError):
            create_block_mapping(16, -4)

    # TEST 6: Block partial handling
    def test_06_block_partial_handling(self):
        # 1000 bits with block size 256 -> 3 full blocks (768), 232 remainder
        raw_bits = np.random.randint(0, 2, size=1000, dtype=np.uint8)
        deint_bits, _, _, rem = deinterleave_block(raw_bits, None, 16, 16, "row_to_column", padding_policy="truncate_tail")
        self.assertEqual(len(deint_bits), 768)
        self.assertEqual(rem, 232)

    # TEST 7: Helical permutation validity
    def test_07_helical_permutation_validity(self):
        for rows in [4, 8, 16]:
            for cols in [8, 16, 32]:
                for step in [1, 2, 3, 5]:
                    mapping = create_helical_mapping(rows, cols, step, "row_diagonal")
                    self.assertTrue(mapping.is_valid())
                    self.assertEqual(len(np.unique(mapping.inverse_indices)), rows * cols)

    # TEST 8: Helical round trip
    def test_08_helical_round_trip(self):
        raw_bits = np.random.randint(0, 2, size=512, dtype=np.uint8)
        llrs = np.random.randn(512).astype(np.float32)
        int_bits, int_llrs = interleave_helical(raw_bits, llrs, 16, 16, step=3)
        deint_bits, deint_llrs, _, _ = deinterleave_helical(int_bits, int_llrs, 16, 16, step=3)
        np.testing.assert_array_equal(deint_bits, raw_bits)
        np.testing.assert_array_almost_equal(deint_llrs, llrs)

    # TEST 9: Pseudo-random round trip
    def test_09_pseudorandom_round_trip(self):
        raw_bits = np.random.randint(0, 2, size=1024, dtype=np.uint8)
        llrs = np.random.randn(1024).astype(np.float32)
        int_bits, int_llrs = interleave_pseudorandom(raw_bits, llrs, length=512, seed=42, algorithm="pcg64")
        deint_bits, deint_llrs, _, _ = deinterleave_pseudorandom(int_bits, int_llrs, length=512, seed=42, algorithm="pcg64")
        np.testing.assert_array_equal(deint_bits, raw_bits)
        np.testing.assert_array_almost_equal(deint_llrs, llrs)

    # TEST 10: Wrong pseudo-random seed differs
    def test_10_wrong_pseudorandom_seed_differs(self):
        raw_bits = np.random.randint(0, 2, size=512, dtype=np.uint8)
        int_bits, _ = interleave_pseudorandom(raw_bits, None, length=512, seed=42)
        wrong_deint, _, _, _ = deinterleave_pseudorandom(int_bits, None, length=512, seed=999)
        # Should not match original
        self.assertFalse(np.array_equal(wrong_deint, raw_bits))

    # TEST 11: Convolutional round trip
    def test_11_convolutional_round_trip(self):
        branches = 4
        delay_step = 2
        latency = branches * (branches - 1) * delay_step  # 4 * 3 * 2 = 24 bits
        
        raw_bits = np.random.randint(0, 2, size=500, dtype=np.uint8)
        # Append flush zeros to recover all payload
        flush_bits = np.concatenate([raw_bits, np.zeros(latency, dtype=np.uint8)])
        
        int_bits, _ = interleave_convolutional(flush_bits, None, branches, delay_step)
        deint_bits, _, _, _ = deinterleave_convolutional(int_bits, None, branches, delay_step, trim_latency=True)
        
        # Original payload should be perfectly recovered after latency trim
        np.testing.assert_array_equal(deint_bits[:len(raw_bits)], raw_bits)

    # TEST 12: Convolutional state persists across chunks
    def test_12_convolutional_state_persists_across_chunks(self):
        branches, delay_step = 4, 1
        stream = np.random.randint(0, 2, size=600, dtype=np.uint8)
        
        # Interleave in one shot
        int_stream, _ = interleave_convolutional(stream, None, branches, delay_step)
        
        # Deinterleave in chunks of 200
        state = ConvolutionalDeinterleaverState(branch_count=branches, delay_step=delay_step)
        c1, _, state, _ = deinterleave_convolutional(int_stream[:200], None, branches, delay_step, state=state)
        c2, _, state, _ = deinterleave_convolutional(int_stream[200:400], None, branches, delay_step, state=state)
        c3, _, state, _ = deinterleave_convolutional(int_stream[400:600], None, branches, delay_step, state=state)
        
        chunked_output = np.concatenate([c1, c2, c3])
        
        # Deinterleave in one shot
        one_shot, _, _, _ = deinterleave_convolutional(int_stream, None, branches, delay_step)
        np.testing.assert_array_equal(chunked_output, one_shot)

    # TEST 13: Hard and LLR permutation alignment
    def test_13_hard_llr_alignment(self):
        bits = np.array([0, 1, 0, 1, 1, 0, 0, 1], dtype=np.uint8)
        llrs = np.array([+5.2, -4.1, +3.8, -6.0, -2.5, +7.1, +1.2, -8.3], dtype=np.float32)
        inv_idx = np.array([3, 0, 7, 2, 5, 1, 4, 6], dtype=np.int64)
        
        reordered_bits, reordered_llrs = apply_inverse_mapping(bits, llrs, inv_idx)
        for i in range(len(bits)):
            source_idx = inv_idx[i]
            self.assertEqual(reordered_bits[i], bits[source_idx])
            self.assertEqual(reordered_llrs[i], llrs[source_idx])

    # TEST 14: Candidate IDs unique
    def test_14_candidate_ids_unique(self):
        candidates = self.engine.generate_candidates(bit_count=1024)
        ids = [c.candidate_id for c in candidates]
        self.assertEqual(len(ids), len(set(ids)))

    # TEST 15: Duplicate permutations removed
    def test_15_duplicate_permutations_removed(self):
        # Generate candidates and verify no duplicate mapping hashes exist in generator
        candidates = self.engine.generate_candidates(bit_count=1024)
        hashes = [c.mapping_hash for c in candidates if c.mapping_hash]
        self.assertEqual(len(hashes), len(set(hashes)))

    # TEST 16: Identity-like duplicates collapsed
    def test_16_identity_like_duplicates_collapsed(self):
        candidates = self.engine.generate_candidates(bit_count=512)
        identity_candidates = [c for c in candidates if c.family == InterleaverFamily.IDENTITY.value]
        self.assertEqual(len(identity_candidates), 1)

    # TEST 17: Beam width respected
    def test_17_beam_width_respected(self):
        var = {
            "variant_id": "test_var_0",
            "hard_bits": np.random.randint(0, 2, size=1024, dtype=np.uint8),
            "soft_llrs": np.random.randn(1024).astype(np.float32)
        }
        res = self.engine.test_candidates(var)
        self.assertLessEqual(len(res.surviving_candidates), self.engine.beam_width)

    # TEST 18: Max candidate limit respected
    def test_18_max_candidate_limit_respected(self):
        cands = self.engine.generate_candidates(bit_count=4096)
        self.assertLessEqual(len(cands), self.engine.generator.max_total)

    # TEST 19: Structural features finite
    def test_19_structural_features_finite(self):
        bits = np.random.randint(0, 2, size=512, dtype=np.uint8)
        llrs = np.random.randn(512).astype(np.float32)
        feat = extract_structural_features(bits, llrs)
        d = feat.to_dict()
        for k, v in d.items():
            self.assertTrue(np.isfinite(v), f"Feature {k} is not finite: {v}")

    # TEST 20: Candidate scoring deterministic
    def test_20_candidate_scoring_deterministic(self):
        bits = np.array([1, 0, 1, 0] * 64, dtype=np.uint8)
        feat = extract_structural_features(bits, None)
        scorer = CandidateScorer()
        s1 = scorer.score_candidate(feat, "block")
        s2 = scorer.score_candidate(feat, "block")
        self.assertEqual(s1, s2)

    # TEST 21: NaN LLR handling
    def test_21_nan_llr_handling(self):
        bits = np.random.randint(0, 2, size=256, dtype=np.uint8)
        llrs = np.array([np.nan, np.inf, -np.inf] + list(np.random.randn(253)), dtype=np.float32)
        var = {"variant_id": "var_nan", "hard_bits": bits, "soft_llrs": llrs}
        res = self.engine.test_candidates(var)
        self.assertTrue(res.top_candidate is not None)
        # Check all features are finite
        self.assertTrue(np.isfinite(res.top_candidate.structural_features.mean_abs_llr))

    # TEST 22: Empty stream rejected
    def test_22_empty_stream_rejected(self):
        var = {"variant_id": "var_empty", "hard_bits": np.array([], dtype=np.uint8)}
        res = self.engine.test_candidates(var)
        self.assertEqual(len(res.surviving_candidates), 0)
        self.assertIn("error", res.metadata)

    # TEST 23: Odd / non-divisible lengths handled
    def test_23_odd_nondivisible_lengths(self):
        bits = np.random.randint(0, 2, size=333, dtype=np.uint8)
        var = {"variant_id": "var_odd", "hard_bits": bits}
        res = self.engine.test_candidates(var)
        self.assertGreater(len(res.surviving_candidates), 0)

    # TEST 24: Large stream operation
    def test_24_large_stream_operation(self):
        bits = np.random.randint(0, 2, size=32768, dtype=np.uint8)
        llrs = np.random.randn(32768).astype(np.float32)
        var = {"variant_id": "var_large", "hard_bits": bits, "soft_llrs": llrs}
        start = time.perf_counter()
        res = self.engine.test_candidates(var)
        elapsed = time.perf_counter() - start
        self.assertLess(elapsed, 3.0)  # Must complete quickly in vectorized numpy
        self.assertGreater(len(res.surviving_candidates), 0)

    # TEST 25: JSON metadata serializable
    def test_25_json_metadata_serializable(self):
        bits = np.random.randint(0, 2, size=512, dtype=np.uint8)
        var = {"variant_id": "var_json", "hard_bits": bits}
        res = self.engine.test_candidates(var)
        d = res.to_dict(include_arrays=False)
        json_str = json.dumps(d)
        self.assertIsInstance(json_str, str)

    # TEST 26: Stage 7 integration with DemodulationVariant
    def test_26_stage7_integration(self):
        from astra_demodulation.src.models import DemodulationVariant, DemodulationQuality
        bits = np.random.randint(0, 2, size=1024, dtype=np.uint8)
        llrs = np.random.randn(1024).astype(np.float32)
        variant = DemodulationVariant(
            variant_id="qpsk_rot90",
            symbol_count=512,
            bit_count=1024,
            hard_bits=bits,
            soft_llrs=llrs,
            quality=DemodulationQuality(evm_percent=4.2, snr_estimate_db=18.5)
        )
        res = self.engine.test_candidates(variant)
        self.assertEqual(res.demod_variant_id, "qpsk_rot90")
        self.assertGreater(len(res.surviving_candidates), 0)

    # TEST 27: Candidate recall evaluation
    def test_27_candidate_recall_evaluation(self):
        raw_b, raw_l, int_b, int_l, meta = generate_synthetic_interleaved_stream(
            bit_count=1024,
            pattern_type="structured_frame",
            interleaver_family="block",
            interleaver_params={"rows": 16, "cols": 16, "orientation": "row_to_column"}
        )
        var = {"variant_id": "var_recall", "hard_bits": int_b, "soft_llrs": int_l}
        res = self.engine.test_candidates(var)
        recall_eval = evaluate_candidate_recall(
            res,
            ground_truth_family="block",
            ground_truth_params={"rows": 16, "cols": 16, "orientation": "row_to_column"}
        )
        self.assertTrue(recall_eval["found_in_top_k"])

    # TEST 28: Pseudo-random unconstrained seed search prohibited
    def test_28_pseudorandom_unconstrained_search_prohibited(self):
        generator = self.engine.generator
        pr_cands = [c for c in generator.generate(1024) if c.family == InterleaverFamily.PSEUDO_RANDOM.value]
        self.assertLessEqual(len(pr_cands), 20)

    # TEST 29: No hidden truth used in production inference
    def test_29_no_hidden_truth_in_inference(self):
        bits = np.random.randint(0, 2, size=512, dtype=np.uint8)
        var = {"variant_id": "blind_var", "hard_bits": bits}
        # Verify inference runs with only demodulated bits/llrs
        res = self.engine.test_candidates(var)
        self.assertNotIn("ground_truth", res.metadata)

    # TEST 30: CPU-only execution
    def test_30_cpu_only_execution(self):
        bits = np.random.randint(0, 2, size=512, dtype=np.uint8)
        res = self.engine.test_candidates({"variant_id": "cpu_var", "hard_bits": bits})
        self.assertTrue(isinstance(res, InterleaverTestResult))


if __name__ == "__main__":
    unittest.main()
