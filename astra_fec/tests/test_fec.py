"""
test_fec.py
Master comprehensive test suite for ASTRA Stage 9 — FEC Candidate Testing Engine.
Covers all 30 mandatory unit tests from Specification Section 92.
"""

import unittest
import numpy as np
import json
import time

from astra_fec.src.models import (
    FECStatus,
    FECFamily,
    FECProfile,
    DecoderResult,
    FECCandidateResult,
    FECTestResult,
)
from astra_fec.src.profiles import get_profile_registry
from astra_fec.src.no_fec import decode_no_fec
from astra_fec.src.convolutional import (
    encode_convolutional,
    depuncture_stream,
    ConvolutionalTrellis,
)
from astra_fec.src.viterbi import (
    viterbi_decode_hard,
    viterbi_decode_soft,
    decode_convolutional_profile,
)
from astra_fec.src.reed_solomon import (
    GaloisField,
    ReedSolomonCodec,
    bits_to_symbols,
    symbols_to_bits,
    decode_reed_solomon_profile,
)
from astra_fec.src.ldpc import (
    construct_qc_ldpc_matrix,
    LDPCCodec,
    decode_ldpc_profile,
)
from astra_fec.src.concatenated import decode_concatenated_profile
from astra_fec.src.candidate_generator import FECCandidateGenerator
from astra_fec.src.inference import FECTestingEngine
from astra_fec.src.utils import (
    generate_synthetic_fec_stream,
    compute_ber,
    evaluate_fec_candidate_recall,
)


class TestFECEngine(unittest.TestCase):
    """30-test master suite for Stage 9 FEC Testing Engine."""

    def setUp(self):
        self.engine = FECTestingEngine()
        self.registry = get_profile_registry()
        np.random.seed(42)

    # TEST 1: NO_FEC identity
    def test_01_no_fec_identity(self):
        bits = np.random.randint(0, 2, size=512, dtype=np.uint8)
        llrs = np.random.randn(512).astype(np.float32)
        res = decode_no_fec(bits, llrs)
        self.assertTrue(res.success)
        np.testing.assert_array_equal(res.decoded_bits, bits)
        np.testing.assert_array_almost_equal(res.decoded_soft_info, llrs)

    # TEST 2: Conv noise-free round trip
    def test_02_conv_noise_free_round_trip(self):
        msg = np.random.randint(0, 2, size=200, dtype=np.uint8)
        encoded = encode_convolutional(msg, constraint_length=7, generators_octal=[171, 133], termination="terminated")
        decoded, metric, norm_metric, term = viterbi_decode_hard(encoded, constraint_length=7, generators_octal=[171, 133])
        self.assertTrue(term)
        self.assertEqual(metric, 0.0)
        np.testing.assert_array_equal(decoded, msg)

    # TEST 3: Hard Viterbi error correction
    def test_03_hard_viterbi_correction(self):
        msg = np.random.randint(0, 2, size=150, dtype=np.uint8)
        encoded = encode_convolutional(msg, constraint_length=7, generators_octal=[171, 133], termination="terminated")
        
        # Inject sparse bit errors (e.g. 3 errors spread out)
        noisy = encoded.copy()
        noisy[10] ^= 1
        noisy[50] ^= 1
        noisy[120] ^= 1
        
        decoded, metric, norm_metric, term = viterbi_decode_hard(noisy, constraint_length=7, generators_octal=[171, 133])
        np.testing.assert_array_equal(decoded, msg)

    # TEST 4: Soft Viterbi
    def test_04_soft_viterbi(self):
        msg = np.random.randint(0, 2, size=150, dtype=np.uint8)
        encoded = encode_convolutional(msg, constraint_length=7, generators_octal=[171, 133], termination="terminated")
        
        # Synthetic LLRs with noise (+5.0 for 0, -5.0 for 1)
        llrs = np.where(encoded == 0, 5.0, -5.0).astype(np.float32)
        llrs += np.random.normal(0, 1.0, size=len(encoded)).astype(np.float32)
        
        decoded, metric, norm_metric, term = viterbi_decode_soft(llrs, constraint_length=7, generators_octal=[171, 133])
        np.testing.assert_array_equal(decoded, msg)

    # TEST 5: LLR sign compatibility (positive favors 0, negative favors 1)
    def test_05_llr_sign_compatibility(self):
        msg = np.array([0, 1, 0, 1, 1, 0], dtype=np.uint8)
        encoded = encode_convolutional(msg, constraint_length=3, generators_octal=[7, 5], termination="terminated")
        
        # Exact positive LLR for 0, negative LLR for 1
        llrs = np.where(encoded == 0, 10.0, -10.0).astype(np.float32)
        decoded, metric, _, _ = viterbi_decode_soft(llrs, constraint_length=3, generators_octal=[7, 5])
        np.testing.assert_array_equal(decoded, msg)
        self.assertAlmostEqual(metric, 0.0, places=3)

    # TEST 6: Terminated convolutional code
    def test_06_terminated_convolutional_code(self):
        msg = np.random.randint(0, 2, size=100, dtype=np.uint8)
        encoded = encode_convolutional(msg, constraint_length=5, generators_octal=[23, 35], termination="terminated")
        decoded, _, _, term_match = viterbi_decode_hard(encoded, constraint_length=5, generators_octal=[23, 35], termination="terminated")
        self.assertTrue(term_match)
        self.assertEqual(len(decoded), 100)
        np.testing.assert_array_equal(decoded, msg)

    # TEST 7: Continuous convolutional mode
    def test_07_continuous_convolutional_mode(self):
        msg = np.random.randint(0, 2, size=120, dtype=np.uint8)
        encoded = encode_convolutional(msg, constraint_length=5, generators_octal=[23, 35], termination="continuous")
        decoded, _, _, term_match = viterbi_decode_hard(encoded, constraint_length=5, generators_octal=[23, 35], termination="continuous")
        self.assertTrue(term_match)
        # Verify message decoded
        np.testing.assert_array_equal(decoded, msg)

    # TEST 8: Punctured convolutional decode
    def test_08_punctured_convolutional_decode(self):
        msg = np.random.randint(0, 2, size=150, dtype=np.uint8)
        prof = self.registry.get_profile("conv_k7_r23_punctured")
        encoded = encode_convolutional(
            msg, constraint_length=7, generators_octal=[171, 133],
            termination="terminated", puncturing_pattern=prof.puncturing_pattern
        )
        res = decode_convolutional_profile(hard_bits=encoded, soft_llrs=None, profile=prof, prefer_soft=False)
        self.assertTrue(res.success)
        np.testing.assert_array_equal(res.decoded_bits, msg)

    # TEST 9: Neutral LLR depuncturing
    def test_09_neutral_llr_depuncturing(self):
        hard_bits = np.array([1, 1, 1], dtype=np.uint8)
        soft_llrs = np.array([-5.0, -5.0, -5.0], dtype=np.float32)
        pattern = [1, 1, 0, 1]  # 4th position punctured
        
        dep_hard, dep_soft = depuncture_stream(hard_bits, soft_llrs, pattern)
        self.assertEqual(len(dep_hard), 4)
        self.assertEqual(dep_soft[2], 0.0)  # Neutral LLR inserted

    # TEST 10: Wrong conv profile produces weaker evidence
    def test_10_wrong_conv_profile_weaker_evidence(self):
        msg = np.random.randint(0, 2, size=200, dtype=np.uint8)
        prof_nasa = self.registry.get_profile("conv_k7_r12_nasa")
        prof_k5 = self.registry.get_profile("conv_k5_r12_standard")
        
        encoded = encode_convolutional(msg, constraint_length=7, generators_octal=[171, 133])
        res_correct = decode_convolutional_profile(encoded, None, prof_nasa, prefer_soft=False)
        res_wrong = decode_convolutional_profile(encoded, None, prof_k5, prefer_soft=False)
        
        self.assertLess(
            res_correct.metrics["normalized_path_metric"],
            res_wrong.metrics["normalized_path_metric"]
        )

    # TEST 11: RS noise-free round trip
    def test_11_rs_noise_free_round_trip(self):
        prof = self.registry.get_profile("rs_255_223_ccsds")
        codec = ReedSolomonCodec(n=255, k=223, m=8, prim_poly=0x187, fcr=112)
        msg_syms = list(np.random.randint(0, 256, size=223))
        cw = codec.encode(msg_syms)
        
        success, decoded_msg, n_err = codec.decode_codeword(cw)
        self.assertTrue(success)
        self.assertEqual(n_err, 0)
        self.assertEqual(decoded_msg, msg_syms)

    # TEST 12: RS corrects within capability
    def test_12_rs_corrects_within_capability(self):
        prof = self.registry.get_profile("rs_255_223_ccsds")
        codec = ReedSolomonCodec(n=255, k=223, m=8, prim_poly=0x187, fcr=112)
        msg_syms = list(np.random.randint(0, 256, size=223))
        cw = codec.encode(msg_syms)
        
        # Inject t=16 symbol errors (t=16 for RS(255, 223))
        corrupted = list(cw)
        for i in range(16):
            corrupted[i * 10] ^= 0x55
            
        success, decoded_msg, n_err = codec.decode_codeword(corrupted)
        self.assertTrue(success)
        self.assertEqual(n_err, 16)
        self.assertEqual(decoded_msg, msg_syms)

    # TEST 13: RS flags beyond capability
    def test_13_rs_flags_beyond_capability(self):
        codec = ReedSolomonCodec(n=255, k=223, m=8, prim_poly=0x187, fcr=112)
        msg_syms = list(np.random.randint(0, 256, size=223))
        cw = codec.encode(msg_syms)
        
        # Inject 20 errors (exceeds t=16)
        corrupted = list(cw)
        for i in range(20):
            corrupted[i * 5] ^= 0xAA
            
        success, _, n_err = codec.decode_codeword(corrupted)
        self.assertFalse(success)
        self.assertEqual(n_err, -1)

    # TEST 14: Shortened RS
    def test_14_shortened_rs(self):
        prof = self.registry.get_profile("rs_32_24_shortened")
        raw_msg, hard_bits, _, _ = generate_synthetic_fec_stream(
            fec_family="reed_solomon",
            profile_id="rs_32_24_shortened",
            snr_db=30.0
        )
        res = decode_reed_solomon_profile(hard_bits, prof)
        self.assertTrue(res.success)
        np.testing.assert_array_equal(res.decoded_bits, raw_msg)

    # TEST 15: LDPC clean decode
    def test_15_ldpc_clean_decode(self):
        prof = self.registry.get_profile("ldpc_128_r12")
        raw_msg, hard_bits, llrs, _ = generate_synthetic_fec_stream(
            fec_family="ldpc",
            profile_id="ldpc_128_r12",
            snr_db=30.0
        )
        res = decode_ldpc_profile(hard_bits, llrs, prof)
        self.assertTrue(res.success)
        np.testing.assert_array_equal(res.decoded_bits, raw_msg)

    # TEST 16: LDPC noisy decode
    def test_16_ldpc_noisy_decode(self):
        prof = self.registry.get_profile("ldpc_128_r12")
        raw_msg, hard_bits, llrs, _ = generate_synthetic_fec_stream(
            fec_family="ldpc",
            profile_id="ldpc_128_r12",
            snr_db=12.0
        )
        res = decode_ldpc_profile(hard_bits, llrs, prof)
        self.assertTrue(res.success)
        np.testing.assert_array_equal(res.decoded_bits, raw_msg)

    # TEST 17: LDPC early stopping
    def test_17_ldpc_early_stopping(self):
        prof = self.registry.get_profile("ldpc_128_r12")
        raw_msg, hard_bits, llrs, _ = generate_synthetic_fec_stream(
            fec_family="ldpc",
            profile_id="ldpc_128_r12",
            snr_db=25.0
        )
        res = decode_ldpc_profile(hard_bits, llrs, prof)
        self.assertTrue(res.success)
        self.assertLess(res.metrics["iterations_used"], 30)

    # TEST 18: LDPC non-convergence
    def test_18_ldpc_non_convergence(self):
        prof = self.registry.get_profile("ldpc_128_r12")
        # Severe random noise (SNR = -10 dB)
        raw_msg, hard_bits, llrs, _ = generate_synthetic_fec_stream(
            fec_family="ldpc",
            profile_id="ldpc_128_r12",
            snr_db=-10.0
        )
        res = decode_ldpc_profile(hard_bits, llrs, prof, max_iter=10)
        self.assertFalse(res.success)
        self.assertFalse(res.metrics["converged"])

    # TEST 19: Concatenated round trip
    def test_19_concatenated_round_trip(self):
        prof = self.registry.get_profile("concat_rs32_convk5")
        outer_prof = self.registry.get_profile(prof.outer_profile)
        inner_prof = self.registry.get_profile(prof.inner_profile)
        
        # Outer encode
        raw_msg, hard_bits, _, _ = generate_synthetic_fec_stream(
            fec_family="reed_solomon",
            profile_id=outer_prof.profile_id,
            snr_db=30.0
        )
        # Inner encode
        inner_encoded = encode_convolutional(
            hard_bits,
            constraint_length=inner_prof.constraint_length,
            generators_octal=inner_prof.generators_octal
        )
        
        res = decode_concatenated_profile(inner_encoded, None, prof, prefer_soft=False)
        self.assertTrue(res.success)
        np.testing.assert_array_equal(res.decoded_bits, raw_msg)

    # TEST 20: Correct concatenated decode order (Viterbi -> RS)
    def test_20_concatenated_decode_order(self):
        prof = self.registry.get_profile("concat_rs32_convk5")
        res = decode_concatenated_profile(np.zeros(300, dtype=np.uint8), None, prof)
        self.assertIn("inner_viterbi_metric", res.metrics)
        self.assertIn("outer_rs_success", res.metrics)

    # TEST 21: Candidate IDs unique
    def test_21_candidate_ids_unique(self):
        cands = self.engine.generate_candidates(bit_count=2048)
        ids = [c.candidate_id for c in cands]
        self.assertEqual(len(ids), len(set(ids)))

    # TEST 22: Beam width enforced
    def test_22_beam_width_enforced(self):
        raw_msg, hard_bits, llrs, _ = generate_synthetic_fec_stream(fec_family="none", msg_len=512)
        res = self.engine.test_candidates({"hard_bits": hard_bits, "soft_llrs": llrs})
        self.assertLessEqual(len(res.surviving_candidates), self.engine.beam_width)

    # TEST 23: Length-incompatible candidates rejected safely
    def test_23_length_incompatible_candidates(self):
        short_bits = np.array([0, 1, 0, 1], dtype=np.uint8)
        res = self.engine.test_candidates({"hard_bits": short_bits})
        self.assertGreater(len(res.surviving_candidates), 0)

    # TEST 24: Alignment offset handling
    def test_24_alignment_offset_handling(self):
        cands = self.engine.generate_candidates(bit_count=4096)
        offsets = [c.bit_offset for c in cands]
        self.assertIn(0, offsets)
        self.assertTrue(any(o > 0 for o in offsets))

    # TEST 25: Hard/LLR alignment retained
    def test_25_hard_llr_alignment_retained(self):
        raw_msg, hard_bits, llrs, _ = generate_synthetic_fec_stream(
            fec_family="convolutional",
            profile_id="conv_k7_r12_nasa",
            snr_db=15.0
        )
        res = self.engine.test_candidates({"hard_bits": hard_bits, "soft_llrs": llrs})
        self.assertTrue(res.top_candidate is not None)

    # TEST 26: NaN LLR rejected / handled safely
    def test_26_nan_llr_handled_safely(self):
        bits = np.random.randint(0, 2, size=512, dtype=np.uint8)
        llrs = np.array([np.nan, np.inf] + list(np.random.randn(510)), dtype=np.float32)
        res = self.engine.test_candidates({"hard_bits": bits, "soft_llrs": llrs})
        self.assertGreater(len(res.surviving_candidates), 0)

    # TEST 27: Empty stream rejected
    def test_27_empty_stream_rejected(self):
        res = self.engine.test_candidates({"hard_bits": np.array([], dtype=np.uint8)})
        self.assertEqual(len(res.surviving_candidates), 0)
        self.assertIn("error", res.metadata)

    # TEST 28: Candidate lineage preserved
    def test_28_candidate_lineage_preserved(self):
        cand_input = {
            "candidate_id": "cand_qpsk",
            "demod_variant_id": "rot90",
            "interleaver_candidate_id": "int_block_r16_c32_0012",
            "hard_bits": np.random.randint(0, 2, size=1024, dtype=np.uint8),
            "soft_llrs": np.random.randn(1024).astype(np.float32)
        }
        res = self.engine.test_candidates(cand_input)
        top = res.top_candidate
        self.assertEqual(top.candidate_id, "cand_qpsk")
        self.assertEqual(top.demod_variant_id, "rot90")
        self.assertEqual(top.interleaver_candidate_id, "int_block_r16_c32_0012")
        self.assertTrue(top.path_id.startswith("path_cand_qpsk_rot90_int_block_r16_c32_0012_"))

    # TEST 29: Stage 8 integration
    def test_29_stage8_integration(self):
        from astra_interleaver.src.models import InterleaverCandidateResult, InterleaverStatus
        int_res = InterleaverCandidateResult(
            candidate_id="cand_stage8_demo",
            demod_variant_id="rot0",
            interleaver_candidate_id="int_none_0001",
            interleaver_family="identity",
            parameters={"type": "identity"},
            deinterleaved_hard_bits=np.random.randint(0, 2, size=1024, dtype=np.uint8),
            deinterleaved_soft_llrs=np.random.randn(1024).astype(np.float32),
            status=InterleaverStatus.INTERLEAVER_PLAUSIBLE
        )
        res = self.engine.test_candidates(int_res)
        self.assertEqual(res.interleaver_candidate_id, "int_none_0001")
        self.assertGreater(len(res.surviving_candidates), 0)

    # TEST 30: JSON metadata serializable
    def test_30_json_metadata_serializable(self):
        bits = np.random.randint(0, 2, size=512, dtype=np.uint8)
        res = self.engine.test_candidates({"hard_bits": bits})
        d = res.to_dict(include_arrays=False)
        json_str = json.dumps(d)
        self.assertIsInstance(json_str, str)


if __name__ == "__main__":
    unittest.main()
