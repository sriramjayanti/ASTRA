"""
test_convolutional_profiles.py
Unit tests for registered convolutional code profiles.
"""

import unittest
import numpy as np
from astra_fec.src.profiles import get_profile_registry
from astra_fec.src.convolutional import encode_convolutional
from astra_fec.src.viterbi import decode_convolutional_profile


class TestConvolutionalProfiles(unittest.TestCase):
    def test_all_conv_profiles_roundtrip(self):
        registry = get_profile_registry()
        profiles = registry.list_profiles(family="convolutional")
        
        for prof in profiles:
            with self.subTest(profile_id=prof.profile_id):
                msg = np.random.randint(0, 2, size=100, dtype=np.uint8)
                encoded = encode_convolutional(
                    msg,
                    constraint_length=prof.constraint_length,
                    generators_octal=prof.generators_octal,
                    termination=prof.termination,
                    puncturing_pattern=prof.puncturing_pattern
                )
                res = decode_convolutional_profile(hard_bits=encoded, soft_llrs=None, profile=prof, prefer_soft=False)
                self.assertTrue(res.success)
                np.testing.assert_array_equal(res.decoded_bits, msg)


if __name__ == "__main__":
    unittest.main()
