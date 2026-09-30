"""
Reed-Solomon Block Encoder and Reference Decoder for ASTRA FEC Engine.
Supports RS(255,223), RS(255,239), shortened codes, multi-block segmentation, and bit-level ground truth.
"""

from __future__ import annotations

from typing import Any
import numpy as np
import reedsolo

from ..payload.models import bytes_to_bits, bits_to_bytes
from .padding import pad_to_multiple


class ReedSolomonCode:
    """Configurable Reed-Solomon (RS) symbol encoder and reference decoder over GF(2^8)."""

    def __init__(
        self,
        n: int = 255,
        k: int = 223,
        symbol_size_bits: int = 8,
    ):
        """Initialize ReedSolomonCode.

        Args:
            n: Total codeword symbol length (default 255).
            k: Message symbol length (default 223).
            symbol_size_bits: Number of bits per symbol (8 for GF(2^8)).
        """
        if symbol_size_bits != 8:
            raise ValueError(f"Only 8-bit symbols (GF(2^8)) are supported, got {symbol_size_bits}")
        if not (1 <= k < n <= 255):
            raise ValueError(f"Invalid RS parameters: requires 1 <= k < n <= 255 (got n={n}, k={k})")

        self.n = n
        self.k = k
        self.symbol_size_bits = symbol_size_bits
        self.parity_symbols = n - k
        self.error_correction_capacity = self.parity_symbols // 2
        self.nominal_rate = self.k / self.n

        # Initialize reedsolo codec
        self.codec = reedsolo.RSCodec(nsym=self.parity_symbols)

    def encode(
        self,
        bits: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, int, list[dict[str, Any]], dict[str, Any]]:
        """Encode input bits into RS codeword bitstream.

        Args:
            bits: 1D NumPy uint8 array of input bits (0 and 1).

        Returns:
            tuple (encoded_bits, padding_bits, total_pad_length, block_boundaries, parameters_dict)
        """
        if not isinstance(bits, np.ndarray):
            bits = np.asarray(bits, dtype=np.uint8)
        if bits.ndim != 1:
            raise ValueError(f"Input bits must be 1D, got shape {bits.shape}")

        # 1. Byte alignment padding
        orig_bit_len = len(bits)
        aligned_bits, align_pad_bits, align_pad_len = pad_to_multiple(bits, 8, pad_value=0)
        raw_bytes = bits_to_bytes(aligned_bits, padding=False)

        # 2. Block segmentation into k-byte chunks
        k_bytes = self.k
        total_bytes = len(raw_bytes)
        num_blocks = max(1, (total_bytes + k_bytes - 1) // k_bytes) if total_bytes > 0 else 1

        encoded_blocks: list[bytes] = []
        block_boundaries: list[dict[str, Any]] = []
        all_padding_bits: list[np.ndarray] = []
        if align_pad_len > 0:
            all_padding_bits.append(align_pad_bits)

        total_pad_len = align_pad_len
        current_encoded_bit_offset = 0

        for b_idx in range(num_blocks):
            start_b = b_idx * k_bytes
            end_b = min(start_b + k_bytes, total_bytes)
            chunk = raw_bytes[start_b:end_b] if total_bytes > 0 else b""

            # Pad chunk to exact k bytes if shorter
            chunk_pad_bytes_len = k_bytes - len(chunk)
            if chunk_pad_bytes_len > 0:
                pad_bytes = bytes(chunk_pad_bytes_len)
                padded_chunk = chunk + pad_bytes
                block_pad_bits = np.zeros(chunk_pad_bytes_len * 8, dtype=np.uint8)
                all_padding_bits.append(block_pad_bits)
                total_pad_len += chunk_pad_bytes_len * 8
            else:
                padded_chunk = chunk
                chunk_pad_bytes_len = 0

            # Encode block
            encoded_chunk = bytes(self.codec.encode(padded_chunk))
            encoded_blocks.append(encoded_chunk)

            enc_len_bits = len(encoded_chunk) * 8
            block_boundaries.append({
                "block_index": b_idx,
                "input_byte_start": start_b,
                "original_input_bytes": len(chunk),
                "padded_input_bytes": len(padded_chunk),
                "padding_bytes": chunk_pad_bytes_len,
                "encoded_start_bit": current_encoded_bit_offset,
                "encoded_bit_length": enc_len_bits,
            })
            current_encoded_bit_offset += enc_len_bits

        all_encoded_bytes = b"".join(encoded_blocks)
        encoded_bits = bytes_to_bits(all_encoded_bytes)

        padding_arr = (
            np.concatenate(all_padding_bits) if all_padding_bits else np.empty(0, dtype=np.uint8)
        )

        params = {
            "n": self.n,
            "k": self.k,
            "symbol_size_bits": self.symbol_size_bits,
            "parity_symbols": self.parity_symbols,
            "error_correction_capacity_symbols": self.error_correction_capacity,
            "nominal_code_rate": round(self.nominal_rate, 6),
            "alignment_padding_bits": align_pad_len,
        }

        return encoded_bits, padding_arr, total_pad_len, block_boundaries, params

    def decode(
        self,
        encoded_bits: np.ndarray,
        original_bit_length: int,
        block_boundaries: list[dict[str, Any]] | None = None,
    ) -> np.ndarray:
        """Reference Reed-Solomon Decoder for verification and round-trip assertions.

        Args:
            encoded_bits: 1D NumPy uint8 array of received bits.
            original_bit_length: Expected length of original unpadded information bits.
            block_boundaries: Block boundary metadata list.

        Returns:
            Decoded 1D NumPy uint8 array of length `original_bit_length`.
        """
        # Convert bits to bytes
        enc_bytes = bits_to_bytes(encoded_bits, padding=True)
        n_bytes = self.n

        num_blocks = len(enc_bytes) // n_bytes
        decoded_bytes_list: list[bytes] = []

        for b_idx in range(num_blocks):
            block_bytes = enc_bytes[b_idx * n_bytes : (b_idx + 1) * n_bytes]
            decoded_chunk, _, _ = self.codec.decode(block_bytes)
            decoded_bytes_list.append(bytes(decoded_chunk))

        all_decoded_bytes = b"".join(decoded_bytes_list)
        all_decoded_bits = bytes_to_bits(all_decoded_bytes)

        return all_decoded_bits[:original_bit_length]
