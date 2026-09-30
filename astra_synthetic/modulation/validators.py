"""
Validation and reference clean demodulator utilities for ASTRA Engine 5.
Validates IQ finite properties, power normalization, metadata, and executes round-trip recovery.
"""

from __future__ import annotations

import json
import hashlib
from typing import Any
import numpy as np

from ..interleaving.models import InterleaverRecord
from ..payload.validators import ValidationError, validate_payload_bits
from .models import ModulationRecord
from .psk import PSKModulator
from .qam import QAMModulator
from .fsk import FSKModulator


def validate_modulation_record(
    record: ModulationRecord,
    original_interleaver: InterleaverRecord | None = None,
) -> None:
    """Validate completeness, mathematical properties, and SHA-256 integrity of a ModulationRecord.

    Args:
        record: ModulationRecord instance.
        original_interleaver: Optional source InterleaverRecord to verify bit preservation.

    Raises:
        ValidationError: If any consistency or integrity check fails.
    """
    if not isinstance(record, ModulationRecord):
        raise ValidationError(f"Expected ModulationRecord, got {type(record).__name__}")

    # 1. Validate bit arrays
    validate_payload_bits(record.input_bits)
    if len(record.input_bits) != record.mapped_bit_length - record.mapping_padding_length:
        raise ValidationError(
            f"Bit length equation mismatch: len(input_bits)={len(record.input_bits)}, "
            f"mapped={record.mapped_bit_length}, pad={record.mapping_padding_length}"
        )

    # 2. Check source Interleaver bits
    if original_interleaver is not None:
        if record.interleaver_record_id != original_interleaver.interleaver_record_id:
            raise ValidationError(
                f"interleaver_record_id mismatch: {record.interleaver_record_id} vs {original_interleaver.interleaver_record_id}"
            )
        if not np.array_equal(record.input_bits, original_interleaver.interleaved_bits):
            raise ValidationError("ModulationRecord.input_bits do not match source InterleaverRecord.interleaved_bits!")

    # 3. Validate clean IQ waveform
    if record.clean_iq.dtype != np.complex64:
        raise ValidationError(f"clean_iq must have dtype complex64, got {record.clean_iq.dtype}")
    if len(record.clean_iq) != record.clean_iq_sample_count:
        raise ValidationError(
            f"IQ sample count mismatch: declared {record.clean_iq_sample_count} vs actual {len(record.clean_iq)}"
        )
    if not np.all(np.isfinite(record.clean_iq)):
        raise ValidationError("clean_iq contains non-finite values (NaN or Inf)!")

    # 4. Power validation
    avg_pwr = np.mean(np.abs(record.clean_iq) ** 2) if len(record.clean_iq) > 0 else 0.0
    if abs(avg_pwr - record.average_iq_power) > 0.05 and avg_pwr > 0:
        raise ValidationError(
            f"Recorded average power ({record.average_iq_power}) deviates from measured ({avg_pwr:.4f})"
        )

    # 5. Check SHA-256 hashes
    expected_in_sha = (
        hashlib.sha256(np.packbits(record.input_bits).tobytes()).hexdigest()
        if len(record.input_bits) > 0
        else hashlib.sha256(b"").hexdigest()
    )
    if record.input_sha256 != expected_in_sha:
        raise ValidationError(
            f"Input SHA-256 mismatch: recorded {record.input_sha256} vs computed {expected_in_sha}"
        )

    expected_iq_sha = (
        hashlib.sha256(record.clean_iq.tobytes()).hexdigest()
        if len(record.clean_iq) > 0
        else hashlib.sha256(b"").hexdigest()
    )
    if record.iq_sha256 != expected_iq_sha:
        raise ValidationError(
            f"IQ SHA-256 mismatch: recorded {record.iq_sha256} vs computed {expected_iq_sha}"
        )

    # 6. Metadata JSON serializability
    try:
        json.dumps(record.to_dict())
    except Exception as e:
        raise ValidationError(f"Modulation metadata is not JSON serializable: {e}") from e


def reference_demodulate(record: ModulationRecord) -> np.ndarray:
    """Execute the reference clean demodulator for a ModulationRecord to verify exact bit recovery.

    Args:
        record: ModulationRecord instance.

    Returns:
        1D NumPy uint8 array of recovered bits.
    """
    mtype = record.modulation_type.lower()
    fam = record.modulation_family.lower()
    params = record.parameters

    if fam == "psk":
        modulator = PSKModulator(
            modulation_type=mtype,
            phase_offset=params.get("phase_offset", 0.0),
            mapping_name=params.get("mapping", "gray_standard_v1"),
        )
        return modulator.reference_demodulate(
            clean_iq=record.clean_iq,
            sps=int(record.samples_per_symbol),
            group_delay=record.filter_group_delay_samples,
            symbol_count=record.symbol_count,
            original_bit_length=len(record.input_bits),
            ideal_symbols=record.ideal_symbols,
        )

    elif fam == "qam":
        modulator = QAMModulator(
            modulation_type=mtype,
            mapping_name=params.get("mapping", "gray_standard_v1"),
        )
        return modulator.reference_demodulate(
            clean_iq=record.clean_iq,
            sps=int(record.samples_per_symbol),
            group_delay=record.filter_group_delay_samples,
            symbol_count=record.symbol_count,
            original_bit_length=len(record.input_bits),
            ideal_symbols=record.ideal_symbols,
        )

    elif fam == "fsk":
        modulator = FSKModulator(
            modulation_type=mtype,
            continuous_phase=params.get("continuous_phase", True),
            tone_spacing_ratio=params.get("tone_spacing_ratio", 1.0),
            initial_phase=params.get("initial_phase_rad", 0.0),
        )
        return modulator.reference_demodulate(
            clean_iq=record.clean_iq,
            symbol_rate=record.symbol_rate,
            sample_rate=record.sample_rate,
            symbol_count=record.symbol_count,
            original_bit_length=len(record.input_bits),
        )

    else:
        raise ValueError(f"Unknown modulation family '{fam}' for reference demodulation.")
