"""
Unit tests for source-chain linkage and multi-stage ID traceability (Engine 8).
"""

from pathlib import Path
import pytest
from astra_synthetic.orchestration.generator import ASTRASyntheticDatasetGenerator


def test_source_chain_linkage(tmp_path):
    gen = ASTRASyntheticDatasetGenerator()
    rec = gen.generate_one(index=5, output_dir=tmp_path)

    assert rec.source_chain_id == "chain_00000005"
    assert rec.payload_id == "payload_00000001" or "payload_" in rec.payload_id
    assert "frame_" in rec.frame_id
    assert "fec_" in rec.fec_record_id
    assert "int_" in rec.interleaver_record_id
    assert "mod_" in rec.modulation_record_id
    assert "chan_" in rec.channel_record_id
    assert "cap_" in rec.capture_record_id
