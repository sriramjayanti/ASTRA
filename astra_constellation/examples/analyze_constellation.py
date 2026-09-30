"""
ASTRA Constellation Analysis Demonstration Script.
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from astra_constellation.src.inference import ConstellationAnalyzer
from astra_constellation.src.geometry import get_reference_constellation


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 80)
    print("ASTRA Constellation Analysis Engine - Demonstration")
    print("=" * 80)

    analyzer = ConstellationAnalyzer()

    # 1. Simulate a clean QPSK burst
    print("\n[Test 1] Analyzing QPSK Constellation...")
    ref_qpsk = get_reference_constellation("QPSK")
    syms = np.random.choice(ref_qpsk, size=1500)
    noise = (np.random.normal(0, 0.06, 1500) + 1j * np.random.normal(0, 0.06, 1500)).astype(np.complex64)
    qpsk_iq = syms + noise

    pred_qpsk = analyzer.analyze(qpsk_iq)
    print(f"  Best Constellation : {pred_qpsk.best_constellation} (Order M = {pred_qpsk.m_ary_order})")
    print(f"  Confidence         : {pred_qpsk.confidence:.4f}")
    print(f"  Status             : {pred_qpsk.status}")
    print(f"  EVM                : {pred_qpsk.evidence.get('estimated_evm_percent', 0):.2f}%")
    print(f"  Radial Rings       : {pred_qpsk.evidence.get('radial_ring_count', 0)}")
    print(f"  Top-3 Candidates   : {[c['constellation_type'] for c in pred_qpsk.top_k]}")

    # 2. Simulate a 16-QAM burst
    print("\n[Test 2] Analyzing 16-QAM Constellation...")
    ref_16qam = get_reference_constellation("16QAM")
    syms_qam = np.random.choice(ref_16qam, size=2000)
    noise_qam = (np.random.normal(0, 0.04, 2000) + 1j * np.random.normal(0, 0.04, 2000)).astype(np.complex64)
    qam_iq = syms_qam + noise_qam

    pred_qam = analyzer.analyze(qam_iq)
    print(f"  Best Constellation : {pred_qam.best_constellation} (Order M = {pred_qam.m_ary_order})")
    print(f"  Confidence         : {pred_qam.confidence:.4f}")
    print(f"  Status             : {pred_qam.status}")
    print(f"  EVM                : {pred_qam.evidence.get('estimated_evm_percent', 0):.2f}%")
    print(f"  Radial Rings       : {pred_qam.evidence.get('radial_ring_count', 0)} (Expected 3 for 16-QAM)")
    print(f"  Grid Compactness   : {pred_qam.evidence.get('square_grid_compactness', 0):.4f}")
    print(f"  Top-3 Candidates   : {[c['constellation_type'] for c in pred_qam.top_k]}")

    print("\n" + "=" * 80)
    print("Demonstration successfully completed.")
    print("=" * 80)


if __name__ == "__main__":
    main()
