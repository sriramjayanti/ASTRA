"""
Integration tests for ASTRASyntheticDatasetGenerator (Engine 8).
"""

from pathlib import Path
import pytest
from astra_synthetic.orchestration.generator import ASTRASyntheticDatasetGenerator


def test_orchestrator_generate_one(tmp_path):
    gen = ASTRASyntheticDatasetGenerator()
    rec = gen.generate_one(index=1, output_dir=tmp_path)

    assert rec.dataset_record_id == "ds_rec_00000001"
    assert rec.source_chain_id == "chain_00000001"
    assert rec.valid is True
    assert Path(rec.capture_path).is_file()
    assert Path(rec.visible_metadata_path).is_file()
    assert Path(rec.truth_metadata_path).is_file()

    # Check that labels contain key parameters
    assert "modulation_type" in rec.labels
    assert "fec_family" in rec.labels
    assert "interleaver_family" in rec.labels
    assert "difficulty" in rec.labels
    assert rec.split in ("train", "validation", "test")


def test_orchestrator_clean_control_exact_demodulation(tmp_path):
    gen = ASTRASyntheticDatasetGenerator()
    # Force clean channel and QPSK
    overrides = {
        "channel": {"profile": "clean", "difficulty": "clean"},
        "modulation": {"profile": "qpsk_rrc", "modulation_type": "qpsk"},
    }
    rec = gen.generate_one(index=2, output_dir=tmp_path, overrides=overrides)
    assert rec.valid is True
    assert rec.validation_results.get("clean_demodulation_exact") is True
