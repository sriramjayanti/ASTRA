"""
ASTRA Synthetic Engine 8 — Full End-to-End Dataset Orchestration Demonstration.

Demonstrates:
1. Orchestrated execution of Engines 1 through 7 into reproducible datasets.
2. Immutable source_chain_id assignment across all 7 stages.
3. Hierarchical seed derivation with np.random.SeedSequence.
4. Independent visible metadata (.capture.json) vs complete hidden truth (.truth.json).
5. Deterministic, zero-leakage source-level train/validation/test splitting.
6. Task-specific dataset manifests (Modulation, FEC, Interleaver, Bitstream Structure).
7. Comprehensive data quality, leakage verification, and distribution reporting.
8. Blind test package export and secret evaluation truth package export.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from astra_synthetic.orchestration.generator import ASTRASyntheticDatasetGenerator
from astra_synthetic.orchestration.splits import verify_no_split_leakage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("astra_orchestration_demo")


def run_full_dataset_demo():
    workspace_root = Path(__file__).parent.parent.parent
    config_path = workspace_root / "astra_synthetic" / "configs" / "dataset_config.yaml"
    output_dir = workspace_root / "datasets" / "astra_synthetic_v1"
    blind_test_dir = workspace_root / "datasets" / "blind_test"
    eval_truth_dir = workspace_root / "datasets" / "evaluation_truth"

    print("=" * 80)
    print("ASTRA SYNTHETIC ENGINE 8 — DATASET ORCHESTRATION DEMONSTRATION")
    print("=" * 80)

    # 1. Initialize Generator from Configuration
    logger.info(f"Loading master configuration from: {config_path}")
    generator = ASTRASyntheticDatasetGenerator(config=config_path)

    # 2. Generate 100-record demonstration dataset
    demo_count = 100
    print(f"\n[STEP 1] Generating {demo_count} synthetic records across full transmitter chain...")
    records = generator.generate_dataset(
        count=demo_count,
        output_root=output_dir,
        show_progress=True,
    )
    print(f"[OK] Generated {len(records)} records. Valid count: {sum(1 for r in records if r.valid)}")

    # 3. Verify Leakage
    print("\n[STEP 2] Verifying Source Chain ID Split Leakage (Train / Val / Test)...")
    is_disjoint, overlap_report = verify_no_split_leakage(records)
    if is_disjoint:
        print("[OK] PASSED: ZERO split leakage detected! Train, validation, and test source IDs are strictly disjoint.")
    else:
        print(f"[FAIL] FAILED: Split leakage detected! Overlaps: {overlap_report}")
        raise RuntimeError("Split leakage detected!")

    # 4. Check Generated Manifests
    print("\n[STEP 3] Verifying Generated Manifests:")
    manifests_dir = output_dir / "manifests"
    for mf in sorted(manifests_dir.glob("*.csv")):
        lines = len(mf.read_text(encoding="utf-8").strip().splitlines()) - 1
        print(f"  - {mf.name:20s}: {lines:4d} entries ({mf.stat().st_size / 1024:.1f} KB)")

    # 5. Check Reports
    print("\n[STEP 4] Verifying Diagnostic Reports:")
    reports_dir = output_dir / "reports"
    dq_file = reports_dir / "data_quality.json"
    if dq_file.exists():
        with open(dq_file, "r", encoding="utf-8") as f:
            dq = json.load(f)
        print(f"  - Data Quality: {dq['valid_records']}/{dq['total_records']} valid ({dq['valid_percentage']}%), "
              f"Failed: {dq['failed_records']}, "
              f"Shortcut Warnings: {len(dq.get('shortcut_warnings', []))}")

    # 6. Export Blind Test Package
    print("\n[STEP 5] Exporting Blind Evaluation Package (No Truth Labels)...")
    blind_pkg = generator.export_blind_test_set(output_dir=blind_test_dir, split_name="test")
    print(f"[OK] Exported Blind Test Package to: {blind_pkg}")

    # 7. Export Evaluation Truth Package
    print("\n[STEP 6] Exporting Secret Evaluation Truth Package (For Scoring)...")
    eval_pkg = generator.export_evaluation_truth(output_dir=eval_truth_dir, split_name="test")
    print(f"[OK] Exported Evaluation Truth Package to: {eval_pkg}")

    print("\n" + "=" * 80)
    print("DEMONSTRATION COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    run_full_dataset_demo()
