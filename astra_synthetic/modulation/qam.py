"""
Quadrature Amplitude Modulation (QAM) Modulators and Reference Demodulators for ASTRA Engine 5.
Implements normalized 16-QAM, 64-QAM (and 256-QAM) with 2D Gray coding and RRC pulse shaping.
"""

from __future__ import annotations

from typing import Any
import numpy as np

from .bit_mapping import (
    QAM16_AXIS_GRAY,
    QAM16_AXIS_INVERSE,
    QAM64_AXIS_GRAY,
    QAM64_AXIS_INVERSE,
    binary_symbols_to_bits,
    pad_and_group_bits,
)
from .filters import apply_rrc_pulse_shaping
from .normalization import calculate_average_power, normalize_signal_power
from .pulse_shaping import extract_matched_symbols


class QAMModulator:
    """Configurable Square QAM Modulator (16-QAM, 64-QAM, 256-QAM)."""

    def __init__(
        self,
        modulation_type: str = "16qam",
        mapping_name: str = "gray_standard_v1",
    ):
        """Initialize QAM Modulator.

        Args:
            modulation_type: '16qam', '64qam', or '256qam'.
            mapping_name: Mapping name.
        """
        self.mod_type = modulation_type.lower()
        self.mapping_name = mapping_name

        if self.mod_type == "16qam":
            self.m = 16
            self.k = 4
            self.k_axis = 2
            self.axis_levels = np.array([-3.0, -1.0, 1.0, 3.0], dtype=np.float64)
            # Average power of 16-QAM with +/-1, +/-3 is (1^2 + 3^2)/2 = 10 -> factor = 1 / sqrt(10)
            self.norm_factor = 1.0 / np.sqrt(10.0)
            self.axis_gray_map = QAM16_AXIS_GRAY
            self.axis_gray_inv = QAM16_AXIS_INVERSE

        elif self.mod_type == "64qam":
            self.m = 64
            self.k = 6
            self.k_axis = 3
            self.axis_levels = np.array([-7.0, -5.0, -3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=np.float64)
            # Average power = (1^2 + 3^2 + 5^2 + 7^2)/4 = 42 -> factor = 1 / sqrt(42)
            self.norm_factor = 1.0 / np.sqrt(42.0)
            self.axis_gray_map = QAM64_AXIS_GRAY
            self.axis_gray_inv = QAM64_AXIS_INVERSE

        elif self.mod_type == "256qam":
            self.m = 256
            self.k = 8
            self.k_axis = 4
            self.axis_levels = np.arange(-15.0, 16.0, 2.0, dtype=np.float64)
            # Average power = 170 -> factor = 1 / sqrt(170)
            self.norm_factor = 1.0 / np.sqrt(170.0)
            # Generate 4-bit Gray map dynamically
            from .bit_mapping import binary_to_gray
            gray_order = [binary_to_gray(i) for i in range(16)]
            self.axis_gray_map = {gray_order[i]: self.axis_levels[i] for i in range(16)}
            self.axis_gray_inv = {self.axis_levels[i]: gray_order[i] for i in range(16)}

        else:
            raise ValueError(f"Unsupported QAM modulation type: {self.mod_type}")

        # Build complete 2D normalized constellation grid
        self.constellation_grid = {}
        for b_i, val_i in self.axis_gray_map.items():
            for b_q, val_q in self.axis_gray_map.items():
                sym_int = (b_i << self.k_axis) | b_q
                z = complex(val_i * self.norm_factor, val_q * self.norm_factor)
                self.constellation_grid[sym_int] = z

    def map_bits_to_symbols(
        self,
        bits: np.ndarray,
        pad_mode: str = "zeros",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
        """Map bits to normalized 2D QAM complex symbols.

        Returns:
            tuple (ideal_symbols, symbol_indices, padding_bits, pad_length, mapped_length)
        """
        binary_syms, pad_bits, pad_len, mapped_len = pad_and_group_bits(
            bits=bits,
            bits_per_symbol=self.k,
            pad_mode=pad_mode,
        )

        # Split into I-axis bits (top k_axis bits) and Q-axis bits (bottom k_axis bits)
        i_bits = binary_syms >> self.k_axis
        q_bits = binary_syms & ((1 << self.k_axis) - 1)

        i_vals = np.array([self.axis_gray_map[int(b)] for b in i_bits], dtype=np.float64)
        q_vals = np.array([self.axis_gray_map[int(b)] for b in q_bits], dtype=np.float64)

        ideal_symbols = ((i_vals + 1j * q_vals) * self.norm_factor).astype(np.complex64)
        return ideal_symbols, binary_syms, pad_bits, pad_len, mapped_len

    def modulate(
        self,
        bits: np.ndarray,
        sps: int = 8,
        pulse_shape: str = "rrc",
        rolloff: float = 0.35,
        span_symbols: int = 8,
        pad_mode: str = "zeros",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int, np.ndarray, int, dict[str, Any]]:
        """Modulate bits into clean pulse-shaped baseband IQ waveform."""
        ideal_symbols, symbol_indices, pad_bits, pad_len, mapped_len = self.map_bits_to_symbols(
            bits=bits, pad_mode=pad_mode
        )

        if pulse_shape == "rrc":
            clean_iq, filter_taps, group_delay = apply_rrc_pulse_shaping(
                symbols=ideal_symbols,
                sps=sps,
                beta=rolloff,
                span_symbols=span_symbols,
            )
        else:
            clean_iq = np.repeat(ideal_symbols, sps).astype(np.complex64)
            filter_taps = np.ones(sps, dtype=np.float64) / np.sqrt(sps)
            group_delay = 0

        clean_iq, scale_factor, pre_power = normalize_signal_power(clean_iq, target_power=1.0)

        params = {
            "modulation_type": self.mod_type,
            "modulation_family": "qam",
            "M": self.m,
            "bits_per_symbol": self.k,
            "mapping": self.mapping_name,
            "raw_axis_levels": list(self.axis_levels),
            "normalization_factor": float(self.norm_factor),
            "pulse_shape": pulse_shape,
            "rolloff": rolloff if pulse_shape == "rrc" else None,
            "span_symbols": span_symbols if pulse_shape == "rrc" else None,
            "sps": sps,
            "group_delay_samples": group_delay,
        }

        return (
            clean_iq,
            ideal_symbols,
            symbol_indices,
            pad_bits,
            pad_len,
            mapped_len,
            filter_taps,
            group_delay,
            params,
        )

    def reference_demodulate(
        self,
        clean_iq: np.ndarray,
        sps: int,
        group_delay: int,
        symbol_count: int,
        original_bit_length: int,
        filter_taps: np.ndarray | None = None,
        ideal_symbols: np.ndarray | None = None,
    ) -> np.ndarray:
        """Reference clean QAM demodulator to recover original transmitted bits."""
        if ideal_symbols is not None:
            sliced_syms = ideal_symbols
        else:
            sliced_syms = extract_matched_symbols(
                clean_iq=clean_iq,
                sps=sps,
                group_delay=group_delay,
                symbol_count=symbol_count,
                filter_taps=filter_taps,
            )

        # Scale back from normalized constellation to unnormalized integer grids
        unnorm_i = sliced_syms.real / self.norm_factor
        unnorm_q = sliced_syms.imag / self.norm_factor

        # Separable slicing to closest axis point
        # For axis levels [-3, -1, 1, 3], boundaries are -2, 0, 2
        dists_i = np.abs(unnorm_i[:, None] - self.axis_levels[None, :])
        closest_i_idx = np.argmin(dists_i, axis=1)
        sliced_i_val = self.axis_levels[closest_i_idx]

        dists_q = np.abs(unnorm_q[:, None] - self.axis_levels[None, :])
        closest_q_idx = np.argmin(dists_q, axis=1)
        sliced_q_val = self.axis_levels[closest_q_idx]

        # Map axis values to Gray-decoded bit groups
        i_bits = np.array([self.axis_gray_inv[v] for v in sliced_i_val], dtype=np.int64)
        q_bits = np.array([self.axis_gray_inv[v] for v in sliced_q_val], dtype=np.int64)

        binary_symbols = (i_bits << self.k_axis) | q_bits

        return binary_symbols_to_bits(
            symbols=binary_symbols,
            bits_per_symbol=self.k,
            original_bit_length=original_bit_length,
        )
