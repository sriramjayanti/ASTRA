"""
test_convolutional.py
Unit tests for convolutional interleaver/deinterleaver family with streaming states.
"""

import unittest
import numpy as np
from astra_interleaver.src.convolutional import (
    ConvolutionalInterleaverEngine,
    ConvolutionalDeinterleaverEngine,
    interleave_convolutional,
    deinterleave_convolutional
)
from astra_interleaver.src.models import ConvolutionalDeinterleaverState


class TestConvolutionalInterleaver(unittest.TestCase):
    def test_convolutional_roundtrip(self):
        branches, delay_step = 4, 2
        latency = branches * (branches - 1) * delay_step
        raw_bits = np.random.randint(0, 2, size=400, dtype=np.uint8)
        llrs = np.random.randn(400).astype(np.float32)
        
        # Add flush tail
        flush_bits = np.concatenate([raw_bits, np.zeros(latency, dtype=np.uint8)])
        flush_llrs = np.concatenate([llrs, np.zeros(latency, dtype=np.float32)])
        
        int_bits, int_llrs = interleave_convolutional(flush_bits, flush_llrs, branches, delay_step)
        deint_bits, deint_llrs, _, _ = deinterleave_convolutional(int_bits, int_llrs, branches, delay_step, trim_latency=True)
        
        np.testing.assert_array_equal(deint_bits[:len(raw_bits)], raw_bits)
        np.testing.assert_array_almost_equal(deint_llrs[:len(llrs)], llrs)

    def test_convolutional_chunked_streaming(self):
        branches, delay_step = 3, 2
        stream = np.random.randint(0, 2, size=300, dtype=np.uint8)
        int_stream, _ = interleave_convolutional(stream, None, branches, delay_step)
        
        state = ConvolutionalDeinterleaverState(branch_count=branches, delay_step=delay_step)
        c1, _, state, _ = deinterleave_convolutional(int_stream[:100], None, branches, delay_step, state=state)
        c2, _, state, _ = deinterleave_convolutional(int_stream[100:200], None, branches, delay_step, state=state)
        c3, _, state, _ = deinterleave_convolutional(int_stream[200:300], None, branches, delay_step, state=state)
        
        combined = np.concatenate([c1, c2, c3])
        single_pass, _, _, _ = deinterleave_convolutional(int_stream, None, branches, delay_step)
        np.testing.assert_array_equal(combined, single_pass)


if __name__ == "__main__":
    unittest.main()
