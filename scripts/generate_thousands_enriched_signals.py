"""
ASTRA Synthetic Signal Generator: Thousands of Enriched Multi-Parametric Signals.

Generates thousands of signals targeting parameters missing from CSPB:
- 2-FSK and 4-FSK waveforms (with continuous phase, variable deviations)
- Varied Carrier Frequency Offsets (CFO: -5 kHz to +5 kHz)
- Multi-tier SNR distributions (-6 dB to +26 dB)
- Diverse symbol rates (1.2k, 2.4k, 4.8k, 9.6k, 19.2k, 38.4k Baud)
- Multi-rate sampling (96 kHz, 192 kHz)
- Diverse channel conditions: AWGN, Rayleigh/Rician fading, Doppler drift, clock offsets
- Diverse framing and FEC schemes
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import time
import numpy as np
import yaml

import sys
workspace_root = Path(__file__).resolve().parent.parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

from astra_synthetic.orchestration.generator import ASTRASyntheticDatasetGenerator
from astra_synthetic.orchestration.splits import verify_no_split_leakage

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ENRICHED_GENERATOR")


def build_enriched_config(output_root: Path, count: int = 2000) -> dict:
    """Config with heavy weighting on FSK and missing impairment distributions."""
    return {
        "dataset": {
            "name": "astra_synthetic_enriched_v2",
            "version": "2.0.0",
            "master_seed": 2026,
            "total_records": count,
            "output_root": str(output_root),
            "split": {
                "train": 0.70,
                "validation": 0.15,
                "test": 0.15,
            },
            "storage": {
                "save_intermediates": False,
                "use_hdf5": False,
            },
            "validation": {
                "end_to_end_clean_control": True,
                "leakage_check": True,
                "distribution_check": True,
            },
        },
        "distributions": {
            # Emphasize FSK (missing from CSPB) while also providing balanced PSK/QAM
            "modulation": {
                "2fsk": 0.25,      # 500 signals
                "4fsk": 0.25,      # 500 signals
                "bpsk": 0.10,      # 200 signals
                "qpsk": 0.10,      # 200 signals
                "8psk": 0.08,      # 160 signals
                "msk": 0.08,       # 160 signals
                "16qam": 0.06,     # 120 signals
                "64qam": 0.04,     # 80 signals
                "256qam": 0.04,    # 80 signals
            },
            "fec": {
                "none": 0.25,
                "convolutional": 0.30,
                "reed_solomon": 0.20,
                "concatenated": 0.15,
                "ldpc": 0.10,
            },
            "interleaver": {
                "none": 0.25,
                "block": 0.30,
                "convolutional": 0.20,
                "diagonal": 0.15,
                "pseudo_random": 0.10,
            },
            # Channel difficulty distribution
            "channel_difficulty": {
                "clean": 0.10,
                "easy": 0.30,
                "medium": 0.35,
                "hard": 0.20,
                "satellite_like": 0.05,
            },
            "capture_profiles": {
                "raw_f32_le_iq": 0.50,
                "raw_i16_le_iq": 0.30,
                "wav_i16_iq": 0.20,
            },
        },
    }


def generate_thousands(count: int = 2000, output_dir: Path | None = None):
    workspace_root = Path(__file__).resolve().parent.parent
    if output_dir is None:
        output_dir = workspace_root / "datasets" / "astra_synthetic_enriched_v2"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    cfg = build_enriched_config(output_dir, count=count)
    
    print("=" * 80)
    print(f"ASTRA: GENERATING {count:,} ENRICHED SYNTHETIC SIGNALS")
    print(f"Output Directory: {output_dir}")
    print("Target Missing Parameters: 2-FSK, 4-FSK, Multi-SNR, CFO, Fading Channels")
    print("=" * 80)
    
    t0 = time.time()
    generator = ASTRASyntheticDatasetGenerator(config=cfg)
    records = generator.generate_dataset(
        count=count,
        output_root=output_dir,
        show_progress=True,
    )
    duration = time.time() - t0
    
    valid_count = sum(1 for r in records if r.valid)
    print("\n" + "-" * 80)
    print(f"[GENERATION COMPLETE] Generated {len(records):,} records in {duration:.1f}s ({len(records)/duration:.1f} rec/s)")
    print(f"  • Valid Records: {valid_count:,} / {len(records):,} ({valid_count/len(records)*100:.1f}%)")
    
    # Verify split leakage
    is_disjoint, overlap_report = verify_no_split_leakage(records)
    print(f"  • Split Leakage Status: {'PASSED (Zero Leakage)' if is_disjoint else 'FAILED'}")
    
    # Manifest counts
    manifest_all = output_dir / "manifests" / "all.csv"
    if manifest_all.exists():
        import pandas as pd
        df = pd.read_csv(manifest_all)
        print("\n[MODULATION BREAKDOWN]:")
        for mod, num in df["modulation"].value_counts().items():
            print(f"  - {mod.upper():10s}: {num:4d} signals ({num/len(df)*100:.1f}%)")
            
        print("\n[CHANNEL DIFFICULTY BREAKDOWN]:")
        if "channel_profile" in df.columns:
            for prof, num in df["channel_profile"].value_counts().items():
                print(f"  - {prof:20s}: {num:4d} signals")
                
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=2000, help="Number of signals to generate")
    args = parser.parse_args()
    generate_thousands(count=args.count)
