"""
Dataset Statistics and Quality Reporting Engine for ASTRA (Engine 8).
Computes class distributions, SNR histograms, joint correlation checks,
and generates data_quality.json, leakage_check.json, and distribution_report.json.
"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any, Sequence
import numpy as np

from .models import DatasetRecord
from .splits import verify_no_split_leakage


def compute_dataset_statistics(records: Sequence[DatasetRecord]) -> dict[str, Any]:
    """Compute comprehensive summary statistics across the generated dataset."""
    total = len(records)
    if total == 0:
        return {"total_records": 0}

    valid_count = sum(1 for r in records if r.valid)
    failed_count = total - valid_count

    # 1. Marginal Class Distributions
    mod_counts = Counter(r.labels.get("modulation_type", "unknown") for r in records)
    fec_counts = Counter(r.labels.get("fec_family", "unknown") for r in records)
    int_counts = Counter(r.labels.get("interleaver_family", "unknown") for r in records)
    diff_counts = Counter(r.labels.get("difficulty", "unknown") for r in records)
    fmt_counts = Counter(r.labels.get("file_format", "unknown") for r in records)
    split_counts = Counter(r.split or "unknown" for r in records)

    # 2. SNR & Channel Metrics
    snrs = [float(r.labels["snr_db"]) for r in records if r.labels.get("snr_db") is not None]
    cfos = [float(r.labels.get("cfo_hz", 0.0)) for r in records]

    snr_stats = {
        "count": len(snrs),
        "mean_db": round(float(np.mean(snrs)), 2) if snrs else None,
        "std_db": round(float(np.std(snrs)), 2) if snrs else None,
        "min_db": round(float(np.min(snrs)), 2) if snrs else None,
        "max_db": round(float(np.max(snrs)), 2) if snrs else None,
    }

    cfo_stats = {
        "mean_hz": round(float(np.mean(cfos)), 2) if cfos else 0.0,
        "std_hz": round(float(np.std(cfos)), 2) if cfos else 0.0,
        "max_abs_hz": round(float(np.max(np.abs(cfos))), 2) if cfos else 0.0,
    }

    # 3. Joint Correlation Checks (Shortcut prevention)
    warnings: list[str] = []
    # Check if any modulation class is correlated with a single FEC
    for mod_name in mod_counts:
        mod_recs = [r for r in records if r.labels.get("modulation_type") == mod_name]
        mod_fecs = Counter(r.labels.get("fec_family") for r in mod_recs)
        if len(mod_recs) >= 10 and len(mod_fecs) == 1:
            warnings.append(f"Potential shortcut: Modulation '{mod_name}' is 100% correlated with FEC '{list(mod_fecs.keys())[0]}'")

    return {
        "total_records": total,
        "valid_records": valid_count,
        "failed_records": failed_count,
        "valid_percentage": round((valid_count / total) * 100.0, 2),
        "splits": dict(split_counts),
        "distributions": {
            "modulation": dict(mod_counts),
            "fec": dict(fec_counts),
            "interleaver": dict(int_counts),
            "difficulty": dict(diff_counts),
            "format": dict(fmt_counts),
        },
        "snr_metrics": snr_stats,
        "cfo_metrics": cfo_stats,
        "shortcut_warnings": warnings,
    }


def generate_dataset_reports(
    records: Sequence[DatasetRecord],
    reports_dir: Path | str,
) -> dict[str, Path]:
    """Generate official JSON data quality, leakage, and distribution reports."""
    r_dir = Path(reports_dir)
    r_dir.mkdir(parents=True, exist_ok=True)

    stats = compute_dataset_statistics(records)

    # 1. Data Quality Report
    p_qual = r_dir / "data_quality.json"
    with open(p_qual, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    # 2. Split Leakage Report
    _, leak_info = verify_no_split_leakage(records)
    p_leak = r_dir / "leakage_check.json"
    with open(p_leak, "w", encoding="utf-8") as f:
        json.dump(leak_info, f, indent=2)

    # 3. Distribution Report
    p_dist = r_dir / "distribution_report.json"
    with open(p_dist, "w", encoding="utf-8") as f:
        json.dump(stats.get("distributions", {}), f, indent=2)

    return {
        "data_quality": p_qual,
        "leakage_check": p_leak,
        "distribution_report": p_dist,
    }
