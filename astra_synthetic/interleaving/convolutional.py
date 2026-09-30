"""
Convolutional Interleaver (Ramsey/Forney Multi-Branch Delay Line) for ASTRA Interleaving Engine.
Implements cyclic branch routing, deterministic FIFO delay lines, flush handling, and complementary deinterleaving.
"""

from __future__ import annotations

from typing import Any
import numpy as np


class ConvolutionalInterleaver:
    """Configurable Convolutional Interleaver with complementary reference deinterleaver."""

    def __init__(
        self,
        num_branches: int = 4,
        delay_step: int = 2,
    ):
        """Initialize ConvolutionalInterleaver.

        Args:
            num_branches: Number of commutator branches (B >= 1).
            delay_step: Delay multiplier per branch (M >= 1). Branch i has delay i * M.
        """
        if num_branches < 2:
            raise ValueError(f"num_branches must be >= 2, got {num_branches}")
        if delay_step < 1:
            raise ValueError(f"delay_step must be >= 1, got {delay_step}")

        self.b = num_branches
        self.m = delay_step
        self.branch_delays = [i * self.m for i in range(self.b)]
        self.flush_rounds = (self.b - 1) * self.m
        self.flush_length = self.flush_rounds * self.b
        self.total_pipeline_delay = (self.b - 1) * self.m * self.b

    def interleave(
        self,
        bits: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, int, list[dict[str, Any]], dict[str, Any]]:
        """Interleave input bits through B delay branches with zero-flush tail.

        Args:
            bits: 1D NumPy uint8 array of input bits (0 and 1).

        Returns:
            tuple (interleaved_bits, flush_padding_bits, flush_length, block_boundaries, parameters)
        """
        if not isinstance(bits, np.ndarray):
            bits = np.asarray(bits, dtype=np.uint8)
        if bits.ndim != 1:
            raise ValueError(f"Input bits must be 1D, got shape {bits.shape}")

        # Initialize FIFO delay lines with zeros: branch i has size i * M
        fifos = [[0] * (i * self.m) for i in range(self.b)]
        output_chunks: list[int] = []

        # 1. Process data bits
        for t, b in enumerate(bits):
            branch_idx = t % self.b
            if len(fifos[branch_idx]) > 0:
                out_bit = fifos[branch_idx].pop(0)
                fifos[branch_idx].append(int(b) & 1)
                output_chunks.append(out_bit)
            else:
                output_chunks.append(int(b) & 1)

        # 2. Flush remaining data bits out of delay lines
        flush_bits = np.zeros(self.flush_length, dtype=np.uint8)
        for t, b in enumerate(flush_bits):
            branch_idx = (len(bits) + t) % self.b
            if len(fifos[branch_idx]) > 0:
                out_bit = fifos[branch_idx].pop(0)
                fifos[branch_idx].append(int(b) & 1)
                output_chunks.append(out_bit)
            else:
                output_chunks.append(int(b) & 1)

        interleaved_arr = np.array(output_chunks, dtype=np.uint8)

        block_boundaries = [{
            "block_id": 0,
            "input_start": 0,
            "input_length": len(bits),
            "output_start": 0,
            "output_length": len(interleaved_arr),
            "flush_length": self.flush_length,
            "pipeline_latency": self.total_pipeline_delay,
        }]

        params = {
            "num_branches": self.b,
            "delay_step": self.m,
            "branch_delays": self.branch_delays,
            "flush_length": self.flush_length,
            "total_pipeline_delay": self.total_pipeline_delay,
            "routing_rule": "cyclic_round_robin",
        }

        return interleaved_arr, flush_bits, self.flush_length, block_boundaries, params

    def deinterleave(
        self,
        interleaved_bits: np.ndarray,
        original_bit_length: int,
        block_boundaries: list[dict[str, Any]] | None = None,
    ) -> np.ndarray:
        """Reference Convolutional Deinterleaver using complementary delay lines.

        Branch i has delay (B - 1 - i) * M.
        Total delay through interleaver + deinterleaver is (B - 1) * M * B bits.

        Args:
            interleaved_bits: 1D NumPy uint8 array of received bits.
            original_bit_length: Expected length of original input bitstream.
            block_boundaries: Telemetry metadata.

        Returns:
            Recovered 1D NumPy uint8 array of length `original_bit_length`.
        """
        # Complementary delay lines: branch i has size (B - 1 - i) * M
        fifos = [[0] * ((self.b - 1 - i) * self.m) for i in range(self.b)]
        out_chunks: list[int] = []

        for t, b in enumerate(interleaved_bits):
            branch_idx = t % self.b
            if len(fifos[branch_idx]) > 0:
                out_bit = fifos[branch_idx].pop(0)
                fifos[branch_idx].append(int(b) & 1)
                out_chunks.append(out_bit)
            else:
                out_chunks.append(int(b) & 1)

        # Discard pipeline latency of (B - 1) * M * B bits
        recovered = np.array(out_chunks[self.total_pipeline_delay : self.total_pipeline_delay + original_bit_length], dtype=np.uint8)
        return recovered


def interleave_convolutional(
    bits: np.ndarray,
    num_branches: int = 4,
    delay_step: int = 2,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Convenience functional interface for convolutional interleaving."""
    interleaver = ConvolutionalInterleaver(
        num_branches=num_branches,
        delay_step=delay_step,
    )
    interleaved_arr, flush_bits, flush_len, _, params = interleaver.interleave(bits)
    meta = {
        "flush_length": flush_len,
        "latency_bits": interleaver.total_pipeline_delay,
        "branch_delays": interleaver.branch_delays,
    }
    return interleaved_arr, flush_bits, meta


def deinterleave_convolutional(
    interleaved_bits: np.ndarray,
    num_branches: int = 4,
    delay_step: int = 2,
    original_bit_length: int | None = None,
) -> np.ndarray:
    """Convenience functional interface for convolutional reference deinterleaving."""
    interleaver = ConvolutionalInterleaver(
        num_branches=num_branches,
        delay_step=delay_step,
    )
    orig_len = original_bit_length if original_bit_length is not None else (len(interleaved_bits) - interleaver.flush_length)
    return interleaver.deinterleave(interleaved_bits, original_bit_length=orig_len)

