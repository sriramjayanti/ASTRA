"""
Data models for ASTRA RF / Channel Impairment Generator (Engine 6).
Encapsulates clean vs impaired IQ waveforms, exact channel ground truth,
stage power telemetry, and SHA-256 integrity verification.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any
import numpy as np


@dataclass
class ChannelRecord:
    """Represents an RF channel-impaired baseband IQ record with complete ground truth.

    Attributes:
        channel_record_id: Unique identifier (e.g. 'chan_000001').
        modulation_record_id: Source ModulationRecord ID.
        interleaver_record_id: Source InterleaverRecord ID.
        fec_record_id: Source FECRecord ID.
        frame_id: Source FrameRecord ID.
        clean_iq: Pristine input complex baseband waveform (immutable ground truth, complex64).
        impaired_iq: Impaired output complex baseband waveform (complex64).
        sample_rate: Sampling frequency F_s in Hz.
        snr_db_target: Target Additive White Gaussian Noise SNR in dB (or None).
        snr_db_measured: Measured actual SNR in dB based on true signal & noise power.
        cfo_hz: Carrier Frequency Offset in Hz.
        normalized_cfo: CFO normalized to sample rate (CFO / F_s).
        phase_offset_rad: Carrier phase offset in radians [-pi, pi].
        timing_offset_samples: Fractional/integer sample timing delay.
        timing_offset_symbols: Timing offset expressed in symbol periods.
        gain_db: Signal gain/attenuation in dB.
        gain_linear: Linear amplitude multiplier.
        frequency_drift_hz_per_sec: Linear carrier frequency drift rate (Hz/s).
        fading_type: 'none', 'rayleigh', 'rician'.
        fading_parameters: Detailed fading parameters (e.g. K_factor_db).
        multipath_enabled: Boolean flag indicating if multipath FIR was applied.
        multipath_taps: Complex tap coefficients (or None).
        multipath_delays_samples: Integer/fractional delay offsets per tap.
        interference_enabled: Boolean flag indicating if interference was added.
        interference_parameters: Detailed interference parameters.
        sample_clock_offset_ppm: Sample clock frequency offset in parts-per-million (ppm).
        power_stages: Measured signal power at each intermediate impairment stage.
        impairment_order: Explicit list of applied impairment steps in execution order.
        parameters: Complete configuration parameters dictionary.
        metadata: Lineage tracking and version metadata.
        clean_iq_sha256: SHA-256 hash of clean_iq bytes.
        impaired_iq_sha256: SHA-256 hash of impaired_iq bytes.
    """
    channel_record_id: str
    modulation_record_id: str
    interleaver_record_id: str
    fec_record_id: str
    frame_id: str
    clean_iq: np.ndarray
    impaired_iq: np.ndarray
    sample_rate: float
    snr_db_target: float | None = None
    snr_db_measured: float | None = None
    cfo_hz: float = 0.0
    normalized_cfo: float = 0.0
    phase_offset_rad: float = 0.0
    timing_offset_samples: float = 0.0
    timing_offset_symbols: float = 0.0
    gain_db: float = 0.0
    gain_linear: float = 1.0
    frequency_drift_hz_per_sec: float = 0.0
    fading_type: str = "none"
    fading_parameters: dict[str, Any] = field(default_factory=dict)
    multipath_enabled: bool = False
    multipath_taps: np.ndarray | None = None
    multipath_delays_samples: np.ndarray | None = None
    interference_enabled: bool = False
    interference_parameters: dict[str, Any] = field(default_factory=dict)
    sample_clock_offset_ppm: float = 0.0
    power_stages: dict[str, float] = field(default_factory=dict)
    impairment_order: list[str] = field(default_factory=list)
    parameters: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    clean_iq_sha256: str = ""
    impaired_iq_sha256: str = ""

    def __post_init__(self):
        # Enforce np.complex64 for IQ arrays
        if not isinstance(self.clean_iq, np.ndarray):
            self.clean_iq = np.asarray(self.clean_iq, dtype=np.complex64)
        elif self.clean_iq.dtype != np.complex64:
            self.clean_iq = self.clean_iq.astype(np.complex64)

        if not isinstance(self.impaired_iq, np.ndarray):
            self.impaired_iq = np.asarray(self.impaired_iq, dtype=np.complex64)
        elif self.impaired_iq.dtype != np.complex64:
            self.impaired_iq = self.impaired_iq.astype(np.complex64)

        if self.sample_rate > 0:
            self.normalized_cfo = self.cfo_hz / self.sample_rate

        # Hashes
        if not self.clean_iq_sha256:
            self.clean_iq_sha256 = (
                hashlib.sha256(self.clean_iq.tobytes()).hexdigest()
                if len(self.clean_iq) > 0
                else hashlib.sha256(b"").hexdigest()
            )

        if not self.impaired_iq_sha256:
            self.impaired_iq_sha256 = (
                hashlib.sha256(self.impaired_iq.tobytes()).hexdigest()
                if len(self.impaired_iq) > 0
                else hashlib.sha256(b"").hexdigest()
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert channel record telemetry to JSON-serializable dictionary."""
        return {
            "channel_record_id": self.channel_record_id,
            "modulation_record_id": self.modulation_record_id,
            "interleaver_record_id": self.interleaver_record_id,
            "fec_record_id": self.fec_record_id,
            "frame_id": self.frame_id,
            "sample_rate_hz": float(self.sample_rate),
            "snr_db_target": float(self.snr_db_target) if self.snr_db_target is not None else None,
            "snr_db_measured": round(float(self.snr_db_measured), 4) if self.snr_db_measured is not None else None,
            "cfo_hz": round(float(self.cfo_hz), 4),
            "normalized_cfo": round(float(self.normalized_cfo), 8),
            "phase_offset_rad": round(float(self.phase_offset_rad), 6),
            "timing_offset_samples": round(float(self.timing_offset_samples), 6),
            "timing_offset_symbols": round(float(self.timing_offset_symbols), 6),
            "gain_db": round(float(self.gain_db), 4),
            "gain_linear": round(float(self.gain_linear), 6),
            "frequency_drift_hz_per_sec": round(float(self.frequency_drift_hz_per_sec), 6),
            "fading_type": self.fading_type,
            "fading_parameters": self.fading_parameters,
            "multipath_enabled": bool(self.multipath_enabled),
            "multipath_taps": [
                (float(t.real), float(t.imag)) for t in self.multipath_taps
            ] if self.multipath_taps is not None else None,
            "multipath_delays_samples": [
                int(d) for d in self.multipath_delays_samples
            ] if self.multipath_delays_samples is not None else None,
            "interference_enabled": bool(self.interference_enabled),
            "interference_parameters": self.interference_parameters,
            "sample_clock_offset_ppm": round(float(self.sample_clock_offset_ppm), 4),
            "power_stages": {k: round(float(v), 6) for k, v in self.power_stages.items()},
            "impairment_order": list(self.impairment_order),
            "clean_iq_sample_count": len(self.clean_iq),
            "impaired_iq_sample_count": len(self.impaired_iq),
            "clean_iq_sha256": self.clean_iq_sha256,
            "impaired_iq_sha256": self.impaired_iq_sha256,
            "parameters": self.parameters,
            "metadata": self.metadata,
        }
