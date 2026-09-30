"""
Complete Ground-Truth Builder for ASTRA Dataset Orchestration (Engine 8).
Assembles comprehensive, multi-stage hidden truth documents linking all 7 upstream engines.
"""

from __future__ import annotations

from typing import Any
from astra_synthetic.payload.models import PayloadRecord
from astra_synthetic.framing.models import FrameRecord
from astra_synthetic.fec.models import FECRecord
from astra_synthetic.interleaving.models import InterleaverRecord
from astra_synthetic.modulation.models import ModulationRecord
from astra_synthetic.channel.models import ChannelRecord
from astra_synthetic.capture.models import CaptureRecord


def build_full_truth_document(
    dataset_record_id: str,
    source_chain_id: str,
    split: str | None,
    payload_rec: PayloadRecord,
    frame_rec: FrameRecord,
    fec_rec: FECRecord,
    int_rec: InterleaverRecord,
    mod_rec: ModulationRecord,
    chan_rec: ChannelRecord,
    cap_rec: CaptureRecord,
    generator_versions: dict[str, str],
) -> dict[str, Any]:
    """Construct complete hidden synthetic ground-truth metadata for evaluation and supervised training."""
    entropy_val = getattr(payload_rec, "entropy_estimate", getattr(payload_rec, "entropy", 1.0))
    frame_sha = getattr(frame_rec, "sha256", getattr(frame_rec, "frame_sha256", ""))
    crc_val_str = str(getattr(frame_rec, "crc_value", ""))

    return {
        "identity": {
            "dataset_record_id": dataset_record_id,
            "source_chain_id": source_chain_id,
            "split": split,
            "generator_versions": generator_versions,
        },
        "payload": {
            "payload_id": payload_rec.payload_id,
            "payload_type": payload_rec.payload_type,
            "bit_length": payload_rec.bit_length,
            "byte_length": payload_rec.byte_length,
            "payload_sha256": payload_rec.sha256,
            "entropy": round(float(entropy_val), 4),
        },
        "framing": {
            "frame_id": frame_rec.frame_id,
            "sync_start": frame_rec.sync_start,
            "sync_length": frame_rec.sync_length,
            "header_start": frame_rec.header_start,
            "header_length": frame_rec.header_length,
            "payload_start": frame_rec.payload_start,
            "payload_length": frame_rec.payload_length,
            "crc_start": frame_rec.crc_start,
            "crc_length": frame_rec.crc_length,
            "frame_bit_length": frame_rec.frame_bit_length,
            "crc_type": frame_rec.crc_type,
            "crc_value": crc_val_str,
            "frame_sha256": frame_sha,
        },
        "fec": {
            "fec_record_id": fec_rec.fec_record_id,
            "scheme": fec_rec.fec_profile or fec_rec.fec_type,
            "code_family": fec_rec.fec_type,
            "code_rate": round(float(fec_rec.effective_code_rate), 4),
            "input_bit_length": fec_rec.input_bit_length,
            "encoded_bit_length": fec_rec.encoded_bit_length,
            "padding_bits_count": fec_rec.padding_length,
            "fec_parameters": fec_rec.parameters,
        },
        "interleaving": {
            "interleaver_record_id": int_rec.interleaver_record_id,
            "scheme": int_rec.profile_name or int_rec.interleaver_type,
            "family": int_rec.interleaver_type,
            "input_bit_length": int_rec.input_bit_length,
            "interleaved_bit_length": int_rec.output_bit_length,
            "padding_bits_count": int_rec.padding_length,
            "interleaver_parameters": int_rec.parameters,
        },
        "modulation": {
            "modulation_record_id": mod_rec.modulation_record_id,
            "modulation_type": mod_rec.modulation_type,
            "modulation_family": mod_rec.modulation_family,
            "modulation_order": mod_rec.modulation_order,
            "bits_per_symbol": mod_rec.bits_per_symbol,
            "symbol_count": mod_rec.symbol_count,
            "symbol_rate_baud": mod_rec.symbol_rate,
            "sample_rate_hz": mod_rec.sample_rate,
            "samples_per_symbol": mod_rec.samples_per_symbol,
            "pulse_shape": mod_rec.pulse_shape,
            "rolloff": mod_rec.rolloff,
            "average_iq_power": round(float(mod_rec.average_iq_power), 4),
            "clean_iq_sample_count": mod_rec.clean_iq_sample_count,
            "clean_iq_sha256": mod_rec.iq_sha256,
        },
        "channel": {
            "channel_record_id": chan_rec.channel_record_id,
            "snr_db_target": chan_rec.snr_db_target,
            "snr_db_measured": round(float(chan_rec.snr_db_measured), 4) if chan_rec.snr_db_measured is not None else None,
            "cfo_hz": round(float(chan_rec.cfo_hz), 4),
            "normalized_cfo": round(float(chan_rec.normalized_cfo), 8),
            "phase_offset_rad": round(float(chan_rec.phase_offset_rad), 6),
            "timing_offset_samples": round(float(chan_rec.timing_offset_samples), 6),
            "timing_offset_symbols": round(float(chan_rec.timing_offset_symbols), 6),
            "gain_db": round(float(chan_rec.gain_db), 4),
            "frequency_drift_hz_per_sec": round(float(chan_rec.frequency_drift_hz_per_sec), 6),
            "fading_type": chan_rec.fading_type,
            "fading_parameters": chan_rec.fading_parameters,
            "multipath_enabled": chan_rec.multipath_enabled,
            "interference_enabled": chan_rec.interference_enabled,
            "sample_clock_offset_ppm": round(float(chan_rec.sample_clock_offset_ppm), 4),
            "power_stages": {k: round(float(v), 6) for k, v in chan_rec.power_stages.items()},
            "impaired_iq_sha256": chan_rec.impaired_iq_sha256,
        },
        "capture": {
            "capture_record_id": cap_rec.capture_record_id,
            "file_path": cap_rec.file_path,
            "file_format": cap_rec.file_format,
            "storage_dtype": cap_rec.storage_dtype,
            "endianness": cap_rec.endianness,
            "iq_order": cap_rec.iq_order,
            "channel_count": cap_rec.channel_count,
            "sample_rate_hz": float(cap_rec.sample_rate_hz),
            "sample_count_complex": cap_rec.sample_count_complex,
            "duration_seconds": round(float(cap_rec.duration_seconds), 6),
            "quantization_enabled": cap_rec.quantization_enabled,
            "quantization_scale": float(cap_rec.quantization_scale) if cap_rec.quantization_scale is not None else None,
            "quantization_snr_db": float(cap_rec.quantization_snr_db) if cap_rec.quantization_snr_db is not None else None,
            "file_size_bytes": cap_rec.file_size_bytes,
            "file_sha256": cap_rec.file_sha256,
        },
    }
