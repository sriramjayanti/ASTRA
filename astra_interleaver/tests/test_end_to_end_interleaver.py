"""
test_end_to_end_interleaver.py
End-to-end multi-variant testing from Demodulation to Interleaver Candidate Testing Engine.
"""

import unittest
import numpy as np
from astra_interleaver.src.inference import InterleaverTestingEngine
from astra_interleaver.src.utils import generate_synthetic_interleaved_stream, evaluate_candidate_recall


class TestEndToEndInterleaver(unittest.TestCase):
    def setUp(self):
        self.engine = InterleaverTestingEngine()

    def test_full_pipeline_multi_family_recovery(self):
        # Test synthetic recovery across all families
        test_cases = [
            ("identity", {}),
            ("block", {"rows": 16, "cols": 16, "orientation": "row_to_column"}),
            ("block", {"rows": 8, "cols": 32, "orientation": "column_to_row"}),
            ("helical", {"rows": 16, "cols": 16, "step": 3, "orientation": "row_diagonal"}),
            ("pseudo_random", {"length": 256, "seed": 42, "algorithm": "pcg64"}),
        ]
        
        for family, params in test_cases:
            with self.subTest(family=family, params=params):
                raw_b, raw_l, int_b, int_l, meta = generate_synthetic_interleaved_stream(
                    bit_count=1024,
                    pattern_type="structured_frame",
                    interleaver_family=family,
                    interleaver_params=params
                )
                
                var = {
                    "candidate_id": "cand_e2e",
                    "variant_id": f"demod_variant_{family}",
                    "hard_bits": int_b,
                    "soft_llrs": int_l
                }
                
                result = self.engine.test_candidates(var)
                self.assertGreater(len(result.surviving_candidates), 0)
                
                # Verify identity is always present
                has_identity = any(c.interleaver_family == "identity" for c in result.surviving_candidates)
                self.assertTrue(has_identity)
                
                # Check candidate recall
                if family in ("identity", "block", "helical", "pseudo_random"):
                    recall = evaluate_candidate_recall(result, family, params)
                    self.assertTrue(recall["found_in_top_k"], f"Failed to recall {family} in Top-K")


if __name__ == "__main__":
    unittest.main()
