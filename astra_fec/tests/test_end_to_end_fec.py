"""
test_end_to_end_fec.py
End-to-end multi-family FEC Candidate Testing Engine integration.
"""

import unittest
import numpy as np
from astra_fec.src.inference import FECTestingEngine
from astra_fec.src.utils import generate_synthetic_fec_stream, evaluate_fec_candidate_recall


class TestEndToEndFEC(unittest.TestCase):
    def setUp(self):
        self.engine = FECTestingEngine()

    def test_multi_family_candidate_recall(self):
        test_cases = [
            ("none", "none", 10.0),
            ("convolutional", "conv_k7_r12_nasa", 10.0),
            ("reed_solomon", "rs_32_24_shortened", 25.0),
            ("ldpc", "ldpc_128_r12", 20.0),
        ]
        
        for fam, pid, snr in test_cases:
            with self.subTest(family=fam, profile_id=pid):
                raw_msg, hard_bits, llrs, meta = generate_synthetic_fec_stream(
                    msg_len=200 if fam == "convolutional" else 64,
                    fec_family=fam,
                    profile_id=pid,
                    snr_db=snr
                )
                
                input_variant = {
                    "candidate_id": "cand_e2e",
                    "demod_variant_id": "rot0",
                    "interleaver_candidate_id": "int_none_0001",
                    "hard_bits": hard_bits,
                    "soft_llrs": llrs
                }
                
                res = self.engine.test_candidates(input_variant)
                self.assertGreater(len(res.surviving_candidates), 0)
                
                # Check candidate recall in surviving beam
                eval_res = evaluate_fec_candidate_recall(res, pid)
                self.assertTrue(eval_res["found_in_top_k"], f"Failed to recall {pid} in Top-{self.engine.beam_width}")


if __name__ == "__main__":
    unittest.main()
