"""
Data models for ASTRA Modulation / Clean IQ Waveform Generator (Engine 5).
Encapsulates mapped symbols, ideal constellation points, pulse-shaped clean baseband IQ,
timing/sample metadata, and SHA-256 integrity verification.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any
import numpy as np


@dataclass
class ModulationRecord:
    """Represents a clean baseband IQ modulated waveform record with full ground truth.

    Attributes:
        modulation_record_id: Unique record identifier (e.g. 'mod_000001').
        interleaver_record_id: Upstream InterleaverRecord ID.
        fec_record_id: Source FECRecord ID.
        frame_id: Source FrameRecord ID.
        modulation_type: Canonical modulation name ('2fsk', '4fsk', 'bpsk', 'qpsk', '8psk', '16qam', '64qam').
        modulation_family: Family category ('fsk', 'psk', 'qam').
        modulation_order: Constellation or tone count M (2, 4, 8, 16, 64).
        bits_per_symbol: Number of bits per symbol k = log2(M).
        input_bits: Exact transmitted coding-layer bits from InterleaverRecord.
        mapped_bit_length: Total bits processed after any symbol-alignment padding.
        mapping_padding_bits: Zero-padding bits appended to align to bits_per_symbol.
        mapping_padding_length: Count of padding bits added.
        symbol_indices: 1D array of mapped symbol integers [0, M-1].
        ideal_symbols: 1D array of ideal complex constellation points (complex64).
        symbol_count: Total symbol count N_sym.
        symbol_rate: Symbol rate R_s in symbols/second.
        sample_rate: Sampling frequency F_s in Hz.
        samples_per_symbol: SPS = F_s / R_s.
        pulse_shape: Pulse shaping filter name ('rrc', 'rect', 'none', 'fsk_direct').
        rolloff: Excess bandwidth factor beta for RRC filter (e.g. 0.35), or None.
        filter_span_symbols: Truncation span in symbols for FIR filter, or None.
        filter_group_delay_samples: Filter group delay in samples.
        clean_iq: 1D complex baseband IQ samples (complex64).
        clean_iq_sample_count: Length of clean_iq array.
        duration_seconds: Signal duration = clean_iq_sample_count / sample_rate.
        average_iq_power: Normalized RMS/average power |IQ|^2.
        parameters: Full modulation, mapping, and filter parameters.
        metadata: Pipeline provenance and audit metadata.
        input_sha256: SHA-256 of packed input_bits.
        iq_sha256: SHA-256 of raw complex64 clean_iq bytes.
    """
    modulation_record_id: str
    interleaver_record_id: str
    fec_record_id: str
    frame_id: str
    modulation_type: str
    modulation_family: str
    modulation_order: int
    bits_per_symbol: int
    input_bits: np.ndarray
    mapped_bit_length: int
    mapping_padding_bits: np.ndarray
    mapping_padding_length: int
    symbol_indices: np.ndarray
    ideal_symbols: np.ndarray
    symbol_count: int
    symbol_rate: float
    sample_rate: float
    samples_per_symbol: float
    pulse_shape: str
    rolloff: float | None
    filter_span_symbols: int | None
    filter_group_delay_samples: int
    clean_iq: np.ndarray
    clean_iq_sample_count: int
    duration_seconds: float
    average_iq_power: float
    parameters: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    input_sha256: str = ""
    iq_sha256: str = ""

    def __post_init__(self):
        # Enforce numpy arrays and proper dtypes
        if not isinstance(self.input_bits, np.ndarray):
            self.input_bits = np.asarray(self.input_bits, dtype=np.uint8)
        elif self.input_bits.dtype != np.uint8:
            self.input_bits = self.input_bits.astype(np.uint8)

        if not isinstance(self.mapping_padding_bits, np.ndarray):
            self.mapping_padding_bits = np.asarray(self.mapping_padding_bits, dtype=np.uint8)
        elif self.mapping_padding_bits.dtype != np.uint8:
            self.mapping_padding_bits = self.mapping_padding_bits.astype(np.uint8)

        if not isinstance(self.symbol_indices, np.ndarray):
            self.symbol_indices = np.asarray(self.symbol_indices, dtype=np.int64)

        if not isinstance(self.ideal_symbols, np.ndarray):
            self.ideal_symbols = np.asarray(self.ideal_symbols, dtype=np.complex64)
        elif self.ideal_symbols.dtype != np.complex64:
            self.ideal_symbols = self.ideal_symbols.astype(np.complex64)

        if not isinstance(self.clean_iq, np.ndarray):
            self.clean_iq = np.asarray(self.clean_iq, dtype=np.complex64)
        elif self.clean_iq.dtype != np.complex64:
            self.clean_iq = self.clean_iq.astype(np.complex64)

        # Compute SHA-256 for input bits
        if not self.input_sha256:
            if len(self.input_bits) > 0:
                self.input_sha256 = hashlib.sha256(np.packbits(self.input_bits).tobytes()).hexdigest()
            else:
                self.input_sha256 = hashlib.sha256(b"").hexdigest()

        # Compute SHA-256 for complex64 IQ bytes
        if not self.iq_sha256:
            if len(self.clean_iq) > 0:
                self.iq_sha256 = hashlib.sha256(self.clean_iq.tobytes()).hexdigest()
            else:
                self.iq_sha256 = hashlib.sha256(b"").hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """Convert record telemetry to JSON-serializable dictionary."""
        return {
            "modulation_record_id": self.modulation_record_id,
            "interleaver_record_id": self.interleaver_record_id,
            "fec_record_id": self.fec_record_id,
            "frame_id": self.frame_id,
            "modulation_type": self.modulation_type,
            "modulation_family": self.modulation_family,
            "modulation_order": self.modulation_order,
            "bits_per_symbol": self.bits_per_symbol,
            "input_bit_length": len(self.input_bits),
            "mapped_bit_length": self.mapped_bit_length,
            "mapping_padding_length": self.mapping_padding_length,
            "symbol_count": self.symbol_count,
            "symbol_rate_hz": self.symbol_rate,
            "sample_rate_hz": self.sample_rate,
            "samples_per_symbol": self.samples_per_symbol,
            "pulse_shape": self.pulse_shape,
            "rolloff": self.rolloff,
            "filter_span_symbols": self.filter_span_symbols,
            "filter_group_delay_samples": self.filter_group_delay_samples,
            "clean_iq_sample_count": self.clean_iq_sample_count,
            "duration_seconds": round(self.duration_seconds, 6),
            "average_iq_power": round(float(self.average_iq_power), 6),
            "parameters": self.parameters,
            "input_sha256": self.input_sha256,
            "iq_sha256": self.iq_sha256,
            "metadata": self.metadata,
        }
