"""
demodulate_candidate.py
Demonstration script for ASTRA Stage 7 — Demodulation Engine.
Demonstrates end-to-end flow from Stage 6 synchronization through Stage 7 hard/soft demodulation
and phase ambiguity expansion.
"""

import sys
import os
import json
import numpy as np

# Ensure workspace and module roots are in sys.path
script_dir = os.path.dirname(os.path.abspath(__file__))
demod_root = os.path.dirname(script_dir)
workspace_root = os.path.dirname(demod_root)

for p in [workspace_root, demod_root]:
    if p not in sys.path:
        sys.path.insert(0, p)

from astra_synchronization.src.inference import SynchronizationEngine
from astra_synchronization.src.utils import generate_synthetic_test_signal
from astra_demodulation.src.inference import DemodulationEngine


def run_demo():
    print("=" * 80)
    print("ASTRA STAGE 7 — DEMODULATION ENGINE DEMONSTRATION")
    print("=" * 80)

    # 1. Synthesize signal with known impairments
    sample_rate_hz = 192000.0
    symbol_rate_hz = 9600.0
    cfo_hz = 1150.0
    snr_db = 22.0
    num_symbols = 400

    print("\n[1] TRANSMISSION PARAMETERS:")
    print(f"   Modulation    : QPSK (2 bits/symbol)")
    print(f"   Symbol Rate   : {symbol_rate_hz:g} Baud")
    print(f"   Injected CFO  : {cfo_hz:+.1f} Hz")
    print(f"   Channel SNR   : {snr_db:.1f} dB")
    print(f"   Symbol Count  : {num_symbols} symbols (Expected bits = {num_symbols * 2})")

    rx_iq, gt = generate_synthetic_test_signal(
        modulation="QPSK",
        num_symbols=num_symbols,
        symbol_rate_hz=symbol_rate_hz,
        sample_rate_hz=sample_rate_hz,
        cfo_hz=cfo_hz,
        phase_offset_rad=0.35,
        timing_offset_samples=0.25,
        snr_db=snr_db
    )

    # 2. Execute Stage 6 Synchronization
    print("\n[2] EXECUTING STAGE 6 SYNCHRONIZATION ENGINE...")
    sync_engine = SynchronizationEngine()
    hypothesis = {
        "candidate_id": "cand_QPSK_9600_001",
        "modulation": "QPSK",
        "modulation_family": "PSK",
        "symbol_rate_hz": symbol_rate_hz,
        "sample_rate_hz": sample_rate_hz
    }
    sync_res = sync_engine.synchronize(rx_iq, hypothesis)
    print(f"   Sync Status     : {sync_res.status}")
    print(f"   CFO Recovered   : {sync_res.estimated_cfo_hz:+.1f} Hz")
    print(f"   Symbol Samples  : {sync_res.symbol_count} constellation points")
    print(f"   Phase Ambiguities: {[round(np.rad2deg(a)) for a in sync_res.phase_ambiguity_states]} deg")

    # 3. Execute Stage 7 Demodulation
    print("\n[3] EXECUTING STAGE 7 DEMODULATION ENGINE...")
    demod_engine = DemodulationEngine()
    demod_res = demod_engine.demodulate(sync_res, hypothesis)

    print(f"   Demod Status    : {demod_res.status}")
    print(f"   Hard Bits Output: {demod_res.bit_count} bits (uint8)")
    print(f"   Soft LLRs Output: {len(demod_res.soft_llrs)} LLRs (float32)")
    print(f"   Noise Var Est   : {demod_res.noise_variance:.6f} ({demod_res.noise_source})")

    # 4. Quality Telemetry
    q = demod_res.quality
    print("\n[4] DEMODULATION QUALITY & EVM TELEMETRY:")
    print(f"   EVM RMS         : {q.evm_rms:.4f}")
    print(f"   EVM Percent     : {q.evm_percent:.2f}%")
    print(f"   EVM dB          : {q.evm_db:.2f} dB")
    print(f"   Estimated SNR   : {q.snr_estimate_db:.2f} dB")
    print(f"   Mean |LLR|      : {q.mean_abs_llr:.2f} (Confidence magnitude)")
    print(f"   Low-Conf Bits   : {q.low_confidence_bit_fraction * 100:.1f}%")
    print(f"   Quality Score   : {q.demodulation_quality_score:.4f} [0.0 - 1.0]")

    # 5. Phase Ambiguity Variants
    print(f"\n[5] GENERATED PHASE AMBIGUITY BITSTREAM VARIANTS ({len(demod_res.phase_variants)} VARIANTS):")
    print("-" * 75)
    print(f"{'Variant ID':<26} | {'Rotation':<10} | {'Bits':<8} | {'EVM (%)':<10} | {'Quality':<8}")
    print("-" * 75)
    for v in demod_res.phase_variants:
        print(f"{v.variant_id:<26} | {v.rotation_deg:<10.1f} | {v.bit_count:<8} | {v.quality.evm_percent:<10.2f} | {v.quality.demodulation_quality_score:<8.4f}")
    print("-" * 75)

    # 6. Preview of Recovered Data
    print("\n[6] BITSTREAM PREVIEW (First 32 bits and LLRs):")
    preview_bits = demod_res.hard_bits[:32]
    preview_llrs = demod_res.soft_llrs[:32]
    print(f"   Hard Bits: {''.join(str(b) for b in preview_bits)}")
    print(f"   Soft LLRs: {[round(float(l), 2) for l in preview_llrs[:8]]} ...")
    print(f"   Note: Positive LLR -> Bit 0, Negative LLR -> Bit 1.")

    print("\n[7] NEXT STAGE INTERACTION:")
    print("   Bitstream variants and LLRs are directly ready for Stage 8 Interleaver Testing")
    print("   and Stage 9 FEC Decoders (Viterbi, Reed-Solomon, LDPC).")
    print("=" * 80)


if __name__ == "__main__":
    run_demo()
