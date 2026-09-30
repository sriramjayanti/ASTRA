"""
Frequency Shift Keying (FSK) Waveform Modulators and Reference Demodulators for ASTRA Engine 5.
Implements Continuous-Phase 2-FSK, 4-FSK (and MSK) with exact tone synthesis and non-coherent matched-filter demodulation.
"""

from __future__ import annotations

from typing import Any
import numpy as np

from .bit_mapping import binary_symbols_to_bits, pad_and_group_bits
from .normalization import calculate_average_power, normalize_signal_power


class FSKModulator:
    """Configurable FSK Waveform Modulator (2-FSK, 4-FSK, MSK)."""

    def __init__(
        self,
        modulation_type: str = "2fsk",
        continuous_phase: bool = True,
        tone_spacing_ratio: float = 1.0,
        initial_phase: float = 0.0,
    ):
        """Initialize FSK Modulator.

        Args:
            modulation_type: '2fsk', '4fsk', or 'msk'.
            continuous_phase: If True, phase is continuous across symbol transitions.
            tone_spacing_ratio: Tone spacing relative to symbol rate (Delta_f / R_s).
            initial_phase: Initial starting phase in radians.
        """
        self.mod_type = modulation_type.lower()
        self.continuous_phase = continuous_phase
        self.tone_spacing_ratio = tone_spacing_ratio
        self.initial_phase = initial_phase

        if self.mod_type == "2fsk":
            self.m = 2
            self.k = 1
            # 2 tones centered at +- Delta_f / 2
            # 0 -> -0.5 * tone_spacing, 1 -> +0.5 * tone_spacing
            self.tone_multipliers = np.array([-0.5, 0.5], dtype=np.float64)
            self.tone_map = {0: 0, 1: 1}
            self.tone_inv = {0: 0, 1: 1}

        elif self.mod_type == "4fsk":
            self.m = 4
            self.k = 2
            # 4 tones: -1.5, -0.5, +0.5, +1.5
            self.tone_multipliers = np.array([-1.5, -0.5, 0.5, 1.5], dtype=np.float64)
            # Gray-coded frequency ordering:
            # 00 (0) -> -1.5 (tone 0)
            # 01 (1) -> -0.5 (tone 1)
            # 11 (3) -> +0.5 (tone 2)
            # 10 (2) -> +1.5 (tone 3)
            self.tone_map = {0: 0, 1: 1, 3: 2, 2: 3}
            self.tone_inv = {0: 0, 1: 1, 2: 3, 3: 2}

        elif self.mod_type == "msk":
            self.m = 2
            self.k = 1
            # MSK is CPFSK with modulation index h = 0.5 (tone spacing = 0.5 * R_s)
            self.tone_spacing_ratio = 0.5
            self.continuous_phase = True
            self.tone_multipliers = np.array([-0.5, 0.5], dtype=np.float64)
            self.tone_map = {0: 0, 1: 1}
            self.tone_inv = {0: 0, 1: 1}

        else:
            raise ValueError(f"Unsupported FSK modulation type: {self.mod_type}")

    def modulate(
        self,
        bits: np.ndarray,
        symbol_rate: float = 9600.0,
        sample_rate: float = 192000.0,
        pad_mode: str = "zeros",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int, dict[str, Any]]:
        """Synthesize clean continuous/discontinuous phase FSK complex baseband waveform.

        Args:
            bits: Input bits array.
            symbol_rate: Symbol rate R_s in Baud (Hz).
            sample_rate: Sampling frequency F_s in Hz.
            pad_mode: Padding mode for symbol alignment.

        Returns:
            tuple (clean_iq, ideal_symbols, symbol_indices, pad_bits, pad_len, mapped_len, params)
        """
        sps_float = sample_rate / symbol_rate
        sps = int(round(sps_float))
        if abs(sps_float - sps) > 1e-4:
            raise ValueError(
                f"Non-integer SPS {sps_float:.4f} is not supported in Engine 5 v1 (sample_rate={sample_rate}, symbol_rate={symbol_rate})"
            )

        binary_syms, pad_bits, pad_len, mapped_len = pad_and_group_bits(
            bits=bits,
            bits_per_symbol=self.k,
            pad_mode=pad_mode,
        )

        num_symbols = len(binary_syms)
        tone_spacing_hz = symbol_rate * self.tone_spacing_ratio
        actual_tone_freqs = self.tone_multipliers * tone_spacing_hz

        # Map binary symbol values to selected tone index
        tone_indices = np.array([self.tone_map[int(b)] for b in binary_syms], dtype=np.int64)
        selected_freqs = actual_tone_freqs[tone_indices]

        # Vectorized phase synthesis
        freq_per_sample = np.repeat(selected_freqs, sps)
        dt = 1.0 / sample_rate

        if self.continuous_phase:
            # Phase increments per sample: Delta phi = 2 * pi * f * dt
            phase_increments = 2.0 * np.pi * freq_per_sample * dt
            # Cumulative phase integration
            phase = self.initial_phase + np.cumsum(phase_increments)
        else:
            # Phase resets at each symbol boundary
            m_indices = np.tile(np.arange(sps), num_symbols)
            phase = self.initial_phase + 2.0 * np.pi * freq_per_sample * m_indices * dt

        clean_iq = np.exp(1j * phase).astype(np.complex64)
        ideal_symbols = np.exp(1j * (2.0 * np.pi * selected_freqs / symbol_rate)).astype(np.complex64)

        clean_iq, scale_factor, pre_power = normalize_signal_power(clean_iq, target_power=1.0)

        params = {
            "modulation_type": self.mod_type,
            "modulation_family": "fsk",
            "M": self.m,
            "bits_per_symbol": self.k,
            "continuous_phase": self.continuous_phase,
            "tone_spacing_ratio": self.tone_spacing_ratio,
            "tone_spacing_hz": float(tone_spacing_hz),
            "actual_tone_frequencies_hz": [float(f) for f in actual_tone_freqs],
            "symbol_rate_hz": float(symbol_rate),
            "sample_rate_hz": float(sample_rate),
            "samples_per_symbol": sps,
            "pulse_shape": "fsk_direct",
            "initial_phase_rad": float(self.initial_phase),
        }

        return (
            clean_iq,
            ideal_symbols,
            tone_indices,
            pad_bits,
            pad_len,
            mapped_len,
            params,
        )

    def reference_demodulate(
        self,
        clean_iq: np.ndarray,
        symbol_rate: float,
        sample_rate: float,
        symbol_count: int,
        original_bit_length: int,
    ) -> np.ndarray:
        """Reference non-coherent matched-filter FSK demodulator."""
        sps = int(round(sample_rate / symbol_rate))
        tone_spacing_hz = symbol_rate * self.tone_spacing_ratio
        actual_tone_freqs = self.tone_multipliers * tone_spacing_hz
        dt = 1.0 / sample_rate
        m_vec = np.arange(sps)

        # Precompute candidate tone basis waveforms of length SPS
        tone_bases = np.array([
            np.exp(-1j * 2.0 * np.pi * f * m_vec * dt) for f in actual_tone_freqs
        ], dtype=np.complex128)  # shape: (M, sps)

        detected_tones = []
        for sym_idx in range(symbol_count):
            chunk = clean_iq[sym_idx * sps : (sym_idx + 1) * sps]
            if len(chunk) < sps:
                break
            # Correlate chunk against each tone basis
            correlations = np.dot(tone_bases, chunk)
            energies = np.abs(correlations) ** 2
            best_tone = int(np.argmax(energies))
            detected_tones.append(best_tone)

        # Invert tone index to binary symbol integer
        binary_symbols = np.array([self.tone_inv[t] for t in detected_tones], dtype=np.int64)

        return binary_symbols_to_bits(
            symbols=binary_symbols,
            bits_per_symbol=self.k,
            original_bit_length=original_bit_length,
        )
