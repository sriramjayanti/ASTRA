"""
test_candidate_generation.py
Unit tests for candidate generation and bounding.
"""

import unittest
from astra_fec.src.candidate_generator import FECCandidateGenerator
from astra_fec.src.profiles import get_profile_registry


class TestCandidateGeneration(unittest.TestCase):
    def test_candidate_generation_limits(self):
        generator = FECCandidateGenerator()
        cands = generator.generate(bit_count=4096)
        
        self.assertGreater(len(cands), 0)
        self.assertLessEqual(len(cands), generator.max_candidates)
        
        # Check NO_FEC is present
        has_none = any(c.family == "none" for c in cands)
        self.assertTrue(has_none)


if __name__ == "__main__":
    unittest.main()
