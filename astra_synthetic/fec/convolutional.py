"""
Convolutional Encoder and Viterbi Reference Decoder for ASTRA FEC Engine.
Supports arbitrary constraint length K, generator polynomial tap vectors, and zero-tail termination.
"""

from __future__ import annotations

from typing import Any, Sequence
import numpy as np


def octal_to_taps(octal_val: int, constraint_length: int) -> list[int]:
    """Convert an octal generator polynomial (e.g. 0o171, 0o133) to a binary tap list of length K.

    MSB of tap list corresponds to current input bit (t=0), LSB corresponds to oldest delay stage (t=K-1).
    """
    taps = [(octal_val >> (constraint_length - 1 - i)) & 1 for i in range(constraint_length)]
    return taps


class ConvolutionalCode:
    """Configurable shift-register convolutional encoder with Viterbi reference decoder."""

    def __init__(
        self,
        constraint_length: int = 7,
        generators: Sequence[int] | Sequence[str] = (0o171, 0o133),
        generator_format: str = "octal",
        rate: str = "1/2",
        termination_mode: str = "zero_tail",
    ):
        """Initialize ConvolutionalCode.

        Args:
            constraint_length: Constraint length K (e.g. 3, 5, 7).
            generators: Generator polynomials in specified format (e.g. [0o171, 0o133]).
            generator_format: 'octal', 'decimal', or 'binary_str'.
            rate: Code rate ('1/2', '1/3', etc.).
            termination_mode: 'zero_tail' or 'unterminated'.
        """
        if constraint_length < 2:
            raise ValueError(f"Constraint length K must be >= 2, got {constraint_length}")

        self.k = constraint_length
        self.num_generators = len(generators)
        self.rate_str = rate
        self.nominal_rate = 1.0 / self.num_generators
        self.termination_mode = termination_mode

        # Parse generators to binary tap arrays of length K
        self.octal_generators: list[int] = []
        self.tap_matrix: list[np.ndarray] = []

        for gen in generators:
            if isinstance(gen, str):
                if gen.startswith("0o") or gen.startswith("0O"):
                    oct_val = int(gen, 8)
                elif gen.startswith("0b") or gen.startswith("0B"):
                    oct_val = int(gen, 2)
                else:
                    oct_val = int(gen, 8) if generator_format == "octal" else int(gen)
            else:
                oct_val = int(gen)

            self.octal_generators.append(oct_val)
            taps = octal_to_taps(oct_val, self.k)
            self.tap_matrix.append(np.array(taps, dtype=np.uint8))

        self.num_states = 1 << (self.k - 1)
        self._precompute_trellis()

    def _precompute_trellis(self) -> None:
        """Precompute state transitions and expected outputs for fast Viterbi decoding."""
        self.next_state = np.zeros((self.num_states, 2), dtype=np.int32)
        self.outputs = np.zeros((self.num_states, 2, self.num_generators), dtype=np.uint8)

        for s in range(self.num_states):
            for in_bit in (0, 1):
                ns = (in_bit << (self.k - 2)) | (s >> 1)
                self.next_state[s, in_bit] = ns

                # Construct full register: [in_bit, s_msb, ..., s_lsb]
                reg = np.zeros(self.k, dtype=np.uint8)
                reg[0] = in_bit
                for bit_idx in range(self.k - 1):
                    reg[bit_idx + 1] = (s >> (self.k - 2 - bit_idx)) & 1

                for g_idx, taps in enumerate(self.tap_matrix):
                    out_bit = int(np.sum(reg * taps) % 2)
                    self.outputs[s, in_bit, g_idx] = out_bit

    def encode(
        self,
        bits: np.ndarray,
        termination: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray, int, dict[str, Any]]:
        """Encode input bitstream using shift-register convolutional encoding.

        Args:
            bits: 1D NumPy uint8 array of input bits (0 and 1).
            termination: 'zero_tail' (appends K-1 zeros) or 'unterminated'.

        Returns:
            tuple (encoded_bits, tail_bits, tail_length, parameters_dict)
        """
        if not isinstance(bits, np.ndarray):
            bits = np.asarray(bits, dtype=np.uint8)
        if bits.ndim != 1:
            raise ValueError(f"Input bits must be 1D, got shape {bits.shape}")

        term_mode = termination if termination is not None else self.termination_mode
        tail_len = (self.k - 1) if term_mode == "zero_tail" else 0
        tail_bits = np.zeros(tail_len, dtype=np.uint8)

        # Full input sequence to process
        stream = np.concatenate([bits, tail_bits]) if tail_len > 0 else bits

        encoded_chunks: list[int] = []
        state = 0

        for in_bit in stream:
            b = int(in_bit) & 1
            outs = self.outputs[state, b]
            for o in outs:
                encoded_chunks.append(int(o))
            state = self.next_state[state, b]

        encoded_arr = np.array(encoded_chunks, dtype=np.uint8)

        params = {
            "constraint_length": self.k,
            "generators_octal": self.octal_generators,
            "generators_octal_str": [f"0o{g:o}" for g in self.octal_generators],
            "generator_format": "octal",
            "code_rate": self.rate_str,
            "nominal_code_rate": round(self.nominal_rate, 6),
            "termination_mode": term_mode,
            "tail_length": tail_len,
            "num_generators": self.num_generators,
        }

        return encoded_arr, tail_bits, tail_len, params

    def decode_viterbi(
        self,
        encoded_bits: np.ndarray,
        original_bit_length: int,
        termination: str | None = None,
    ) -> np.ndarray:
        """Reference Hard-Decision Viterbi Decoder for validation and test assertions.

        Args:
            encoded_bits: Received/encoded 1D uint8 bit array.
            original_bit_length: Expected original info bit length.
            termination: Termination mode ('zero_tail' or 'unterminated').

        Returns:
            Decoded 1D uint8 bit array of length `original_bit_length`.
        """
        term_mode = termination if termination is not None else self.termination_mode
        n = self.num_generators
        num_symbols = len(encoded_bits) // n

        if num_symbols == 0:
            return np.empty(0, dtype=np.uint8)

        rx_symbols = encoded_bits[: num_symbols * n].reshape(num_symbols, n)

        INF = 10**8
        path_metrics = np.full(self.num_states, INF, dtype=np.int64)
        path_metrics[0] = 0

        traceback_prev = np.zeros((num_symbols, self.num_states), dtype=np.int32)
        traceback_bit = np.zeros((num_symbols, self.num_states), dtype=np.uint8)

        for t in range(num_symbols):
            rx_sym = rx_symbols[t]
            new_metrics = np.full(self.num_states, INF, dtype=np.int64)

            for s in range(self.num_states):
                if path_metrics[s] >= INF:
                    continue

                for in_bit in (0, 1):
                    ns = self.next_state[s, in_bit]
                    exp_sym = self.outputs[s, in_bit]
                    branch_metric = int(np.count_nonzero(rx_sym != exp_sym))
                    candidate_metric = path_metrics[s] + branch_metric

                    if candidate_metric < new_metrics[ns]:
                        new_metrics[ns] = candidate_metric
                        traceback_prev[t, ns] = s
                        traceback_bit[t, ns] = in_bit

            path_metrics = new_metrics

        if term_mode == "zero_tail":
            curr_state = 0 if path_metrics[0] < INF else int(np.argmin(path_metrics))
        else:
            curr_state = int(np.argmin(path_metrics))

        decoded_rev: list[int] = []
        for t in range(num_symbols - 1, -1, -1):
            b = int(traceback_bit[t, curr_state])
            decoded_rev.append(b)
            curr_state = int(traceback_prev[t, curr_state])

        decoded = decoded_rev[::-1]
        decoded_arr = np.array(decoded, dtype=np.uint8)
        return decoded_arr[:original_bit_length]
