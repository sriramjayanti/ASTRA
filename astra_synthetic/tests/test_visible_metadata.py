"""
Unit tests for anti-leakage in visible metadata documents (Engine 8).
"""

import json
from pathlib import Path
import pytest
from astra_synthetic.orchestration.generator import ASTRASyntheticDatasetGenerator


def test_visible_metadata_no_hidden_labels(tmp_path):
    gen = ASTRASyntheticDatasetGenerator()
    rec = gen.generate_one(index=7, output_dir=tmp_path)

    vis_p = Path(rec.visible_metadata_path)
    assert vis_p.is_file()

    with open(vis_p, "r", encoding="utf-8") as f:
        doc = json.load(f)

    # Strictly forbidden keys in visible metadata
    forbidden_keys = [
        "modulation",
        "modulation_type",
        "modulation_family",
        "fec",
        "fec_family",
        "interleaver",
        "interleaver_family",
        "payload",
        "payload_bits",
        "snr_db",
        "cfo_hz",
    ]

    for k in forbidden_keys:
        assert k not in doc, f"Forbidden label leak detected in visible metadata: '{k}'"

    # Only container-level properties should be visible
    assert "file_name" in doc
    assert "file_format" in doc
    assert "storage_dtype" in doc
    assert "file_size_bytes" in doc
