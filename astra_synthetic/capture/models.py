"""
Data models for ASTRA Sampling / Capture / IQ-WAV File Generator (Engine 7).
Tracks exact storage representations, physical file metrics, visible vs hidden truth metadata,
and SHA-256 integrity hashes.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import numpy as np


@dataclass
class CaptureRecord:
    """Represents a serialized SDR capture file with exact format ground truth and dual metadata tracking.

    Attributes:
        capture_record_id: Unique capture identifier (e.g. 'cap_000001').
        channel_record_id: Source ChannelRecord ID.
        modulation_record_id: Source ModulationRecord ID.
        interleaver_record_id: Source InterleaverRecord ID.
        fec_record_id: Source FECRecord ID.
        frame_id: Source FrameRecord ID.
        file_path: Absolute or relative path to the generated binary/WAV file.
        file_format: File container format ('raw_iq', 'wav', 'sigmf').
        sample_rate_hz: Nominal sample rate F_s in Hz.
        center_frequency_hz: Optional RF center frequency in Hz.
        sample_count_complex: Number of complex baseband IQ samples (N).
        scalar_sample_count: Number of scalar stored samples (2N for I/Q interleaved).
        duration_seconds: Capture duration in seconds (N / F_s).
        storage_dtype: Storage data type ('float32', 'int16', 'int8', 'float64').
        endianness: Storage byte order ('little', 'big').
        iq_order: Interleaving order ('IQ', 'QI').
        channel_count: Number of audio/IQ channels (typically 2).
        quantization_enabled: Whether quantization was applied during capture.
        quantization_bits: Integer bit width if quantized (e.g. 16, 8) or None.
        quantization_scale: Linear amplitude scaling factor applied for quantization.
        quantization_mse: Mean squared quantization error relative to normalized input.
        quantization_snr_db: Signal-to-quantization-noise ratio in dB.
        clipping_enabled: Whether front-end amplitude clipping was applied.
        clipped_sample_count: Number of scalar samples saturated by clipping.
        source_iq_sha256: SHA-256 hash of original ChannelRecord.impaired_iq raw bytes.
        file_sha256: SHA-256 hash of the generated file on disk.
        file_size_bytes: Size of the generated file on disk in bytes.
        visible_metadata_path: Path to visible capture metadata file (accessible to blind receivers).
        truth_metadata_path: Path to hidden truth metadata file (evaluation & scoring).
        sample_rate_visible: Whether sample rate is exposed in visible metadata (or None / masked).
        parameters: Complete configuration parameters used during capture generation.
        metadata: Generator versioning and lineage tracking dictionary.
    """
    capture_record_id: str
    channel_record_id: str
    modulation_record_id: str
    interleaver_record_id: str
    fec_record_id: str
    frame_id: str
    file_path: str
    file_format: str
    sample_rate_hz: float
    center_frequency_hz: float | None = None
    sample_count_complex: int = 0
    scalar_sample_count: int = 0
    duration_seconds: float = 0.0
    storage_dtype: str = "float32"
    endianness: str = "little"
    iq_order: str = "IQ"
    channel_count: int = 2
    quantization_enabled: bool = False
    quantization_bits: int | None = None
    quantization_scale: float | None = None
    quantization_mse: float | None = None
    quantization_snr_db: float | None = None
    clipping_enabled: bool = False
    clipped_sample_count: int = 0
    source_iq_sha256: str = ""
    file_sha256: str = ""
    file_size_bytes: int = 0
    visible_metadata_path: str | None = None
    truth_metadata_path: str | None = None
    sample_rate_visible: bool = True
    parameters: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_visible_dict(self) -> dict[str, Any]:
        """Convert to observable capture metadata visible to a blind receiver/ingestion system.

        DOES NOT contain hidden modulation, coding, SNR, or payload ground truth.
        """
        return {
            "capture_record_id": self.capture_record_id,
            "file_name": Path(self.file_path).name if self.file_path else "",
            "file_format": self.file_format,
            "sample_rate_hz": self.sample_rate_hz if self.sample_rate_visible else None,
            "center_frequency_hz": self.center_frequency_hz,
            "sample_count_complex": self.sample_count_complex,
            "scalar_sample_count": self.scalar_sample_count,
            "duration_seconds": round(float(self.duration_seconds), 6),
            "storage_dtype": self.storage_dtype,
            "endianness": self.endianness,
            "iq_order": self.iq_order,
            "channel_count": self.channel_count,
            "file_size_bytes": self.file_size_bytes,
            "file_sha256": self.file_sha256,
        }

    def to_truth_dict(self) -> dict[str, Any]:
        """Convert to complete synthetic ground truth metadata for evaluation, grading, and ML training."""
        return {
            "capture_record_id": self.capture_record_id,
            "channel_record_id": self.channel_record_id,
            "modulation_record_id": self.modulation_record_id,
            "interleaver_record_id": self.interleaver_record_id,
            "fec_record_id": self.fec_record_id,
            "frame_id": self.frame_id,
            "file_path": self.file_path,
            "file_format": self.file_format,
            "sample_rate_hz_truth": float(self.sample_rate_hz),
            "sample_rate_visible": self.sample_rate_visible,
            "center_frequency_hz": self.center_frequency_hz,
            "sample_count_complex": self.sample_count_complex,
            "scalar_sample_count": self.scalar_sample_count,
            "duration_seconds": round(float(self.duration_seconds), 6),
            "storage_dtype": self.storage_dtype,
            "endianness": self.endianness,
            "iq_order": self.iq_order,
            "channel_count": self.channel_count,
            "quantization": {
                "enabled": self.quantization_enabled,
                "bits": self.quantization_bits,
                "scale": float(self.quantization_scale) if self.quantization_scale is not None else None,
                "mse": float(self.quantization_mse) if self.quantization_mse is not None else None,
                "snr_db": float(self.quantization_snr_db) if self.quantization_snr_db is not None else None,
            },
            "clipping": {
                "enabled": self.clipping_enabled,
                "clipped_sample_count": self.clipped_sample_count,
            },
            "source_iq_sha256": self.source_iq_sha256,
            "file_sha256": self.file_sha256,
            "file_size_bytes": self.file_size_bytes,
            "visible_metadata_path": self.visible_metadata_path,
            "truth_metadata_path": self.truth_metadata_path,
            "parameters": self.parameters,
            "metadata": self.metadata,
        }
