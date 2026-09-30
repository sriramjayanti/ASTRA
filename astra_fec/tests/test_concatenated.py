"""
test_concatenated.py
Unit tests for concatenated code decoding.
"""

import unittest
import numpy as np
from astra_fec.src.profiles import get_profile_registry
from astra_fec.src.concatenated import decode_concatenated_profile
from astra_fec.src.convolutional import encode_convolutional
from astra_fec.src.utils import generate_synthetic_fec_stream


class TestConcatenated(unittest.TestCase):
    def test_concatenated_decoding(self):
        registry = get_profile_registry()
        prof = registry.get_profile("concat_rs32_convk5")
        outer_prof = registry.get_profile(prof.outer_profile)
        inner_prof = registry.get_profile(prof.inner_profile)
        
        raw_msg, rs_bits, _, _ = generate_synthetic_fec_stream(
            fec_family="reed_solomon",
            profile_id=outer_prof.profile_id,
            snr_db=30.0
        )
        conv_bits = encode_convolutional(
            rs_bits,
            constraint_length=inner_prof.constraint_length,
            generators_octal=inner_prof.generators_octal
        )
        
        res = decode_concatenated_profile(conv_bits, None, prof, prefer_soft=False)
        self.assertTrue(res.success)
        np.testing.assert_array_equal(res.decoded_bits, raw_msg)


if __name__ == "__main__":
    unittest.main()
