"""
Integration tests for dataset generation, manifests, reports, and blind test package exports (Engine 8).
"""

from pathlib import Path
import pandas as pd
import pytest
from astra_synthetic.orchestration.generator import ASTRASyntheticDatasetGenerator


def test_dataset_generation_and_manifests(tmp_path):
    gen = ASTRASyntheticDatasetGenerator()
    records = gen.generate_dataset(count=15, output_root=tmp_path, show_progress=False)

    assert len(records) == 15
    assert all(r.valid for r in records)

    # 1. Manifests Verification
    manifests_dir = tmp_path / "manifests"
    assert (manifests_dir / "all.csv").is_file()
    assert (manifests_dir / "train.csv").is_file()
    assert (manifests_dir / "validation.csv").is_file()
    assert (manifests_dir / "test.csv").is_file()
    assert (manifests_dir / "modulation.csv").is_file()
    assert (manifests_dir / "fec.csv").is_file()
    assert (manifests_dir / "interleaver.csv").is_file()
    assert (manifests_dir / "bitstream.csv").is_file()

    df_all = pd.read_csv(manifests_dir / "all.csv")
    assert len(df_all) == 15
    assert "source_chain_id" in df_all.columns
    assert "modulation" in df_all.columns

    # 2. Reports Verification
    reports_dir = tmp_path / "reports"
    assert (reports_dir / "data_quality.json").is_file()
    assert (reports_dir / "leakage_check.json").is_file()
    assert (reports_dir / "distribution_report.json").is_file()

    # 3. Global Metadata
    assert (tmp_path / "dataset_metadata.json").is_file()

    # 4. Blind Test Export
    blind_dir = tmp_path / "blind_test_export"
    gen.export_blind_test_set(output_dir=blind_dir, split_name="test")
    assert (blind_dir / "manifest.csv").is_file()
    assert (blind_dir / "captures").is_dir()
    assert (blind_dir / "visible_metadata").is_dir()

    # Verify blind manifest does NOT have modulation labels
    df_blind = pd.read_csv(blind_dir / "manifest.csv")
    assert "modulation" not in df_blind.columns
    assert "snr_db" not in df_blind.columns
