"""
Phase Shift Keying (PSK) Modulators and Reference Demodulators for ASTRA Engine 5.
Implements normalized BPSK, QPSK, 8PSK (and DQPSK) with Gray coding and RRC pulse shaping.
"""

from __future__ import annotations

from typing import Any
import numpy as np

from .bit_mapping import (
    PSK8_GRAY_INVERSE,
    PSK8_GRAY_MAP,
    QPSK_GRAY_INVERSE,
    QPSK_GRAY_MAP,
    binary_symbols_to_bits,
    pad_and_group_bits,
)
from .filters import apply_rrc_pulse_shaping, root_raised_cosine_filter
from .normalization import calculate_average_power, normalize_signal_power
from .pulse_shaping import extract_matched_symbols


class PSKModulator:
    """Configurable PSK Modulator (BPSK, QPSK, 8PSK, DQPSK)."""

    def __init__(
        self,
        modulation_type: str = "qpsk",
        phase_offset: float = 0.0,
        mapping_name: str = "gray_standard_v1",
    ):
        """Initialize PSK Modulator.

        Args:
            modulation_type: 'bpsk', 'qpsk', '8psk', 'dqpsk'.
            phase_offset: Fixed reference constellation rotation in radians.
            mapping_name: Mapping standard identifier.
        """
        self.mod_type = modulation_type.lower()
        self.phase_offset = phase_offset
        self.mapping_name = mapping_name

        if self.mod_type == "bpsk":
            self.m = 2
            self.k = 1
            # BPSK constellation: 0 -> +1, 1 -> -1
            self.constellation = np.array([1.0 + 0.0j, -1.0 + 0.0j], dtype=np.complex64)
        elif self.mod_type in ["qpsk", "dqpsk"]:
            self.m = 4
            self.k = 2
            # QPSK 4 points on unit circle: (+1+1j)/sqrt(2), (-1+1j)/sqrt(2), (-1-1j)/sqrt(2), (+1-1j)/sqrt(2)
            pts = np.array([1.0 + 1.0j, -1.0 + 1.0j, -1.0 - 1.0j, 1.0 - 1.0j], dtype=np.complex64) / np.sqrt(2.0)
            if abs(phase_offset) > 1e-6:
                pts = pts * np.exp(1j * phase_offset)
            self.constellation = pts.astype(np.complex64)
        elif self.mod_type == "8psk":
            self.m = 8
            self.k = 3
            # 8-PSK 8 equidistant points on unit circle
            angles = 2.0 * np.pi * np.arange(8) / 8.0 + phase_offset
            self.constellation = np.exp(1j * angles).astype(np.complex64)
        else:
            raise ValueError(f"Unsupported PSK modulation type: {self.mod_type}")

    def map_bits_to_symbols(
        self,
        bits: np.ndarray,
        pad_mode: str = "zeros",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
        """Map binary input bits to ideal complex constellation symbols.

        Returns:
            tuple (ideal_symbols, symbol_indices, padding_bits, pad_length, mapped_length)
        """
        binary_syms, pad_bits, pad_len, mapped_len = pad_and_group_bits(
            bits=bits,
            bits_per_symbol=self.k,
            pad_mode=pad_mode,
        )

        if self.mod_type == "bpsk":
            # 0 -> index 0 (+1), 1 -> index 1 (-1)
            mapped_indices = binary_syms
            ideal_symbols = self.constellation[mapped_indices]

        elif self.mod_type == "qpsk":
            # Apply Gray mapping
            mapped_indices = np.array([QPSK_GRAY_MAP[b] for b in binary_syms], dtype=np.int64)
            ideal_symbols = self.constellation[mapped_indices]

        elif self.mod_type == "dqpsk":
            # Differential QPSK
            mapped_indices = np.array([QPSK_GRAY_MAP[b] for b in binary_syms], dtype=np.int64)
            # Accumulate phase changes
            phase_steps = np.angle(self.constellation[mapped_indices])
            cum_phases = np.cumsum(phase_steps)
            ideal_symbols = np.exp(1j * cum_phases).astype(np.complex64)

        elif self.mod_type == "8psk":
            # Apply 8-PSK Gray mapping
            mapped_indices = np.array([PSK8_GRAY_MAP[b] for b in binary_syms], dtype=np.int64)
            ideal_symbols = self.constellation[mapped_indices]

        return ideal_symbols, mapped_indices, pad_bits, pad_len, mapped_len

    def modulate(
        self,
        bits: np.ndarray,
        sps: int = 8,
        pulse_shape: str = "rrc",
        rolloff: float = 0.35,
        span_symbols: int = 8,
        pad_mode: str = "zeros",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int, np.ndarray, int, dict[str, Any]]:
        """Modulate bits into clean pulse-shaped baseband IQ waveform.

        Returns:
            tuple (clean_iq, ideal_symbols, symbol_indices, padding_bits, pad_len, mapped_len, filter_taps, group_delay, params)
        """
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
            # Rectangular / flat pulse
            clean_iq = np.repeat(ideal_symbols, sps).astype(np.complex64)
            filter_taps = np.ones(sps, dtype=np.float64) / np.sqrt(sps)
            group_delay = 0

        # Ensure normalized average power ≈ 1.0
        clean_iq, scale_factor, pre_power = normalize_signal_power(clean_iq, target_power=1.0)

        params = {
            "modulation_type": self.mod_type,
            "modulation_family": "psk",
            "M": self.m,
            "bits_per_symbol": self.k,
            "mapping": self.mapping_name,
            "phase_offset": self.phase_offset,
            "pulse_shape": pulse_shape,
            "rolloff": rolloff if pulse_shape == "rrc" else None,
            "span_symbols": span_symbols if pulse_shape == "rrc" else None,
            "sps": sps,
            "group_delay_samples": group_delay,
            "constellation_points": [(float(z.real), float(z.imag)) for z in self.constellation],
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
        """Reference clean demodulator to recover original transmitted bits."""
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

        if self.mod_type == "dqpsk":
            # Differential demodulation: compute phase transitions relative to previous symbol
            prev_syms = np.concatenate([[1.0 + 0j], sliced_syms[:-1]])
            diff_syms = sliced_syms * np.conj(prev_syms)
            dists = np.abs(diff_syms[:, None] - self.constellation[None, :])
            closest_indices = np.argmin(dists, axis=1)
            binary_symbols = np.array([QPSK_GRAY_INVERSE[idx] for idx in closest_indices], dtype=np.int64)
        else:
            # Minimum Euclidean distance slicer against constellation points
            dists = np.abs(sliced_syms[:, None] - self.constellation[None, :])
            closest_indices = np.argmin(dists, axis=1)

            # Invert Gray mapping to standard binary symbol integers
            if self.mod_type == "bpsk":
                binary_symbols = closest_indices
            elif self.mod_type == "qpsk":
                binary_symbols = np.array([QPSK_GRAY_INVERSE[idx] for idx in closest_indices], dtype=np.int64)
            elif self.mod_type == "8psk":
                binary_symbols = np.array([PSK8_GRAY_INVERSE[idx] for idx in closest_indices], dtype=np.int64)
            else:
                binary_symbols = closest_indices

        return binary_symbols_to_bits(
            symbols=binary_symbols,
            bits_per_symbol=self.k,
            original_bit_length=original_bit_length,
        )
