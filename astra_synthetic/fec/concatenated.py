"""
Concatenated FEC Encoder (Outer Reed-Solomon + Inner Convolutional) for ASTRA.
Encapsulates two-stage coding, intermediate ground-truth preservation, and reverse reference decoding.
"""

from __future__ import annotations

from typing import Any
import numpy as np

from .reed_solomon import ReedSolomonCode
from .convolutional import ConvolutionalCode


class ConcatenatedCode:
    """Concatenated FEC Encoder: Outer Reed-Solomon + Inner Convolutional."""

    def __init__(
        self,
        outer_rs: ReedSolomonCode | None = None,
        inner_conv: ConvolutionalCode | None = None,
    ):
        """Initialize ConcatenatedCode.

        Args:
            outer_rs: Outer ReedSolomonCode instance (default RS(255, 223)).
            inner_conv: Inner ConvolutionalCode instance (default K=7, rate 1/2).
        """
        self.outer = outer_rs or ReedSolomonCode(n=255, k=223)
        self.inner = inner_conv or ConvolutionalCode(constraint_length=7, generators=(0o171, 0o133))
        self.nominal_rate = self.outer.nominal_rate * self.inner.nominal_rate

    def encode(
        self,
        bits: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, int, list[dict[str, Any]], dict[str, Any], dict[str, np.ndarray]]:
        """Perform two-stage concatenated encoding: Input -> RS -> Convolutional.

        Args:
            bits: 1D NumPy uint8 array of input bits (0 and 1).

        Returns:
            tuple (encoded_bits, padding_bits, total_pad_length, block_boundaries, parameters, intermediate_stages)
        """
        if not isinstance(bits, np.ndarray):
            bits = np.asarray(bits, dtype=np.uint8)

        # Stage 1: Outer Reed-Solomon Encoding
        rs_bits, rs_pad_bits, rs_pad_len, rs_blocks, rs_params = self.outer.encode(bits)

        # Stage 2: Inner Convolutional Encoding
        conv_bits, conv_tail_bits, conv_tail_len, conv_params = self.inner.encode(rs_bits)

        total_pad_len = rs_pad_len + conv_tail_len
        padding_arr = (
            np.concatenate([rs_pad_bits, conv_tail_bits])
            if (len(rs_pad_bits) > 0 or len(conv_tail_bits) > 0)
            else np.empty(0, dtype=np.uint8)
        )

        params = {
            "outer": {
                "type": "reed_solomon",
                **rs_params,
            },
            "inner": {
                "type": "convolutional",
                **conv_params,
            },
            "encoding_order": ["reed_solomon", "convolutional"],
            "nominal_code_rate": round(self.nominal_rate, 6),
        }

        intermediates = {
            "rs_encoded_bits": rs_bits,
            "final_convolutional_bits": conv_bits,
        }

        return conv_bits, padding_arr, total_pad_len, rs_blocks, params, intermediates

    def decode(
        self,
        encoded_bits: np.ndarray,
        original_bit_length: int,
        intermediate_rs_length: int,
        block_boundaries: list[dict[str, Any]] | None = None,
    ) -> np.ndarray:
        """Reference Concatenated Decoder: Viterbi (Inner) -> RS (Outer).

        Args:
            encoded_bits: Received concatenated bitstream.
            original_bit_length: Expected length of original information bits.
            intermediate_rs_length: Expected bit length of RS encoded stream before conv.
            block_boundaries: RS block boundary telemetry.

        Returns:
            Decoded original bit array of length `original_bit_length`.
        """
        # Step 1: Decode Inner Convolutional with Viterbi
        viterbi_decoded_bits = self.inner.decode_viterbi(
            encoded_bits=encoded_bits,
            original_bit_length=intermediate_rs_length,
        )

        # Step 2: Decode Outer Reed-Solomon
        rs_decoded_bits = self.outer.decode(
            encoded_bits=viterbi_decoded_bits,
            original_bit_length=original_bit_length,
            block_boundaries=block_boundaries,
        )

        return rs_decoded_bits
