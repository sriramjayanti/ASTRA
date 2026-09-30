"""
synchronize_candidate.py
Demonstration script for ASTRA Stage 6 — Synchronization Engine.
Simulates competing candidate hypotheses against an impaired IQ waveform and compares lock metrics.
"""

import sys
import os
import json
import numpy as np

# Ensure workspace and module root are in sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
sync_root = os.path.dirname(script_dir)
workspace_root = os.path.dirname(sync_root)

for p in [workspace_root, sync_root]:
    if p not in sys.path:
        sys.path.insert(0, p)

from astra_synchronization.src.inference import SynchronizationEngine
from astra_synchronization.src.utils import generate_synthetic_test_signal


def run_demo():
    print("=" * 80)
    print("ASTRA STAGE 6 — SYNCHRONIZATION ENGINE DEMONSTRATION")
    print("=" * 80)

    # 1. Synthesize an impaired transmission (Ground Truth: QPSK @ 9600 Baud, CFO = +1250 Hz, SNR = 20 dB)
    sample_rate_hz = 192000.0
    true_mod = "QPSK"
    true_baud = 9600.0
    true_cfo = 1250.0
    true_phase = 0.45
    true_timing_offset = 0.35
    true_snr = 20.0

    print("\n[1] TRANSMITTED SIGNAL ENVIRONMENT:")
    print(f"   Ground Truth Modulation: {true_mod}")
    print(f"   Ground Truth Symbol Rate: {true_baud:g} Baud (SPS = {sample_rate_hz/true_baud:.1f})")
    print(f"   Injected CFO           : {true_cfo:+.1f} Hz ({(true_cfo/sample_rate_hz)*100:.2f}% of Fs)")
    print(f"   Carrier Phase Offset   : {true_phase:.2f} rad")
    print(f"   Timing Offset          : {true_timing_offset:.2f} samples")
    print(f"   Channel SNR            : {true_snr:.1f} dB")
    print(f"   Sampling Rate          : {sample_rate_hz:g} Hz")

    rx_iq, gt = generate_synthetic_test_signal(
        modulation=true_mod,
        num_symbols=600,
        symbol_rate_hz=true_baud,
        sample_rate_hz=sample_rate_hz,
        cfo_hz=true_cfo,
        phase_offset_rad=true_phase,
        timing_offset_samples=true_timing_offset,
        snr_db=true_snr,
        rolloff=0.35
    )

    engine = SynchronizationEngine()

    # 2. Stage 5 Competing Candidate Hypotheses
    candidates = [
        {
            "candidate_id": "cand_QPSK_9600_001",
            "modulation": "QPSK",
            "modulation_family": "PSK",
            "symbol_rate_hz": 9600.0,
            "sample_rate_hz": sample_rate_hz,
            "initial_score": 0.68
        },
        {
            "candidate_id": "cand_8PSK_9600_002",
            "modulation": "8PSK",
            "modulation_family": "PSK",
            "symbol_rate_hz": 9600.0,
            "sample_rate_hz": sample_rate_hz,
            "initial_score": 0.10
        },
        {
            "candidate_id": "cand_QPSK_4800_003",
            "modulation": "QPSK",
            "modulation_family": "PSK",
            "symbol_rate_hz": 4800.0,
            "sample_rate_hz": sample_rate_hz,
            "initial_score": 0.08
        },
        {
            "candidate_id": "cand_16QAM_9600_004",
            "modulation": "16-QAM",
            "modulation_family": "QAM",
            "symbol_rate_hz": 9600.0,
            "sample_rate_hz": sample_rate_hz,
            "initial_score": 0.04
        }
    ]

    print("\n[2] TESTING CANDIDATE HYPOTHESES FROM STAGE 5:")
    results = engine.synchronize_batch(rx_iq, candidates)

    # 3. Formatted Comparison Table
    print("\n" + "=" * 92)
    print(f"{'Candidate ID':<24} | {'Mod':<8} | {'Baud':<6} | {'Status':<11} | {'CFO Est (Hz)':<12} | {'Timing':<7} | {'Carrier':<7} | {'Sync Score':<10}")
    print("-" * 92)
    for r in results:
        t_score = r.lock_metrics.get("timing_lock_score", 0.0)
        c_score = r.lock_metrics.get("carrier_lock_score", 0.0)
        overall = r.lock_metrics.get("overall_sync_score", 0.0)
        print(f"{r.candidate_id:<24} | {r.modulation:<8} | {r.symbol_rate_hz:<6.0f} | {r.status:<11} | {r.estimated_cfo_hz:<+12.1f} | {t_score:<7.4f} | {c_score:<7.4f} | {overall:<10.4f}")
    print("=" * 92)

    # 4. Deep Inspection of Winning Candidate
    winning = results[0]
    print(f"\n[3] WINNING CANDIDATE DETAILED SYNCHRONIZATION TELEMETRY ({winning.candidate_id}):")
    print(winning.to_json(indent=2))

    print("\n[4] NEXT STAGE READY:")
    print(f"   Recovered Symbol Stream Length: {winning.symbol_count} symbols (1 sample/symbol)")
    print(f"   Carrier Ambiguity States       : {[round(a, 3) for a in winning.phase_ambiguity_states]}")
    print(f"   Matched Filter Rolloff (alpha) : {winning.matched_filter_rolloff}")
    print(f"   Directly consumable by Stage 7 Demodulation Engine.")
    print("=" * 80)


if __name__ == "__main__":
    run_demo()
