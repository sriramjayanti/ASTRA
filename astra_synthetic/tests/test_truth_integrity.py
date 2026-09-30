"""
Unit tests for complete multi-stage ground truth document integrity (Engine 8).
"""

import json
from pathlib import Path
import pytest
from astra_synthetic.orchestration.generator import ASTRASyntheticDatasetGenerator


def test_truth_document_completeness(tmp_path):
    gen = ASTRASyntheticDatasetGenerator()
    rec = gen.generate_one(index=10, output_dir=tmp_path)

    truth_p = Path(rec.truth_metadata_path)
    assert truth_p.is_file()

    with open(truth_p, "r", encoding="utf-8") as f:
        doc = json.load(f)

    # Verify all 8 core sections exist
    required_sections = [
        "identity",
        "payload",
        "framing",
        "fec",
        "interleaving",
        "modulation",
        "channel",
        "capture",
    ]
    for sec in required_sections:
        assert sec in doc, f"Missing section: {sec}"

    # Verify specific key parameters
    assert "payload_sha256" in doc["payload"]
    assert "crc_type" in doc["framing"]
    assert "code_family" in doc["fec"]
    assert "modulation_type" in doc["modulation"]
    assert "snr_db_measured" in doc["channel"]
    assert "file_sha256" in doc["capture"]
