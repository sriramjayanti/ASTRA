"""
Full Pipeline Validation Suite for ASTRA Dataset Orchestration (Engine 8).
Verifies end-to-end chain integrity, source-linkage consistency, reference decoding,
and visible metadata leak prevention.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any
import numpy as np

from astra_synthetic.payload.validators import ValidationError
from astra_synthetic.payload.models import PayloadRecord
from astra_synthetic.framing.models import FrameRecord
from astra_synthetic.fec.models import FECRecord
from astra_synthetic.interleaving.models import InterleaverRecord
from astra_synthetic.modulation.models import ModulationRecord
from astra_synthetic.channel.models import ChannelRecord
from astra_synthetic.capture.models import CaptureRecord
from astra_synthetic.modulation import reference_demodulate


FORBIDDEN_VISIBLE_KEYS = {
    "modulation",
    "modulation_type",
    "modulation_family",
    "fec",
    "fec_type",
    "interleaver",
    "interleaver_type",
    "payload_bits",
    "payload_type",
    "snr_db",
    "cfo_hz",
}


def validate_full_synthetic_chain(
    payload_rec: PayloadRecord,
    frame_rec: FrameRecord,
    fec_rec: FECRecord,
    int_rec: InterleaverRecord,
    mod_rec: ModulationRecord,
    chan_rec: ChannelRecord,
    cap_rec: CaptureRecord,
    is_clean_control: bool = False,
) -> tuple[bool, dict[str, Any]]:
    """Validate completeness and mathematical integrity across the entire 7-engine pipeline.

    Args:
        payload_rec: Engine 1 output.
        frame_rec: Engine 2 output.
        fec_rec: Engine 3 output.
        int_rec: Engine 4 output.
        mod_rec: Engine 5 output.
        chan_rec: Engine 6 output.
        cap_rec: Engine 7 output.
        is_clean_control: If True, executes full bit-exact reference demodulation and validation.

    Returns:
        tuple (is_valid, validation_report_dict)

    Raises:
        ValidationError: If any critical chain inconsistency is detected.
    """
    results: dict[str, Any] = {}

    # 1. Payload & Framing Checks
    if len(payload_rec.payload_bits) == 0:
        raise ValidationError("PayloadRecord has 0 payload_bits")
    if frame_rec.payload_length != len(payload_rec.payload_bits):
        raise ValidationError(f"Frame payload length {frame_rec.payload_length} != payload length {len(payload_rec.payload_bits)}")

    p_extracted = frame_rec.frame_bits[frame_rec.payload_start : frame_rec.payload_start + frame_rec.payload_length]
    if not np.array_equal(p_extracted, payload_rec.payload_bits):
        raise ValidationError("FrameRecord does not accurately contain the original PayloadRecord bits!")
    results["framing_payload_intact"] = True

    # 2. Source Lineage Verification
    if fec_rec.frame_id != frame_rec.frame_id:
        raise ValidationError("FECRecord.frame_id does not match FrameRecord.frame_id")
    if int_rec.fec_record_id != fec_rec.fec_record_id:
        raise ValidationError("InterleaverRecord.fec_record_id does not match FECRecord.fec_record_id")
    if mod_rec.interleaver_record_id != int_rec.interleaver_record_id:
        raise ValidationError("ModulationRecord.interleaver_record_id does not match InterleaverRecord.interleaver_record_id")
    if chan_rec.modulation_record_id != mod_rec.modulation_record_id:
        raise ValidationError("ChannelRecord.modulation_record_id does not match ModulationRecord.modulation_record_id")
    if cap_rec.channel_record_id != chan_rec.channel_record_id:
        raise ValidationError("CaptureRecord.channel_record_id does not match ChannelRecord.channel_record_id")
    results["source_lineage_coherent"] = True

    # 3. Clean Waveform Immutability
    expected_clean_sha = hashlib.sha256(mod_rec.clean_iq.tobytes()).hexdigest()
    if chan_rec.clean_iq_sha256 != expected_clean_sha:
        raise ValidationError("ChannelRecord clean_iq SHA-256 altered from source ModulationRecord clean_iq!")
    results["clean_iq_immutable"] = True

    # 4. Anti-Leakage Check on Visible Metadata
    vis_dict = cap_rec.to_visible_dict()
    for k in FORBIDDEN_VISIBLE_KEYS:
        if k in vis_dict:
            raise ValidationError(f"CRITICAL LEAK: Forbidden label '{k}' found in visible capture metadata!")
    results["visible_metadata_clean"] = True

    # 5. End-to-End Reference Demodulation for Clean Controls
    if is_clean_control:
        recovered_int_bits = reference_demodulate(mod_rec)
        if not np.array_equal(recovered_int_bits[: len(int_rec.interleaved_bits)], int_rec.interleaved_bits):
            raise ValidationError("Clean control reference demodulation failed to recover interleaved bits!")
        results["clean_demodulation_exact"] = True

    return True, results
