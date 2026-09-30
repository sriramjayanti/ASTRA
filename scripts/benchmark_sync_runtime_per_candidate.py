"""
Benchmarks Stage 6 Synchronization Runtime per Modulation Candidate Hypothesis.
"""

from __future__ import annotations
import sys
import time
from pathlib import Path
import numpy as np

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

from astra_synchronization.src.inference import SynchronizationEngine


def benchmark_sync_runtime():
    # Synthesize 4096 samples of QPSK with CFO and timing offset
    fs = 192000.0
    baud = 24000.0
    sps = fs / baud
    n_syms = int(4096 / sps)
    
    bits = np.random.randint(0, 4, n_syms)
    syms = np.exp(1j * (bits * np.pi / 2 + np.pi / 4))
    upsampled = np.zeros(int(n_syms * sps), dtype=np.complex64)
    upsampled[::int(sps)] = syms
    
    # RRC pulse
    t = np.arange(-8 * sps, 8 * sps + 1)
    # Simple Gaussian/sinc approximation
    pulse = np.sinc(t / sps) * np.cos(np.pi * 0.35 * t / sps)
    iq = np.convolve(upsampled, pulse, mode="same")
    
    # Add CFO and AWGN
    time_vec = np.arange(len(iq)) / fs
    iq = iq * np.exp(1j * 2 * np.pi * 1500.0 * time_vec)
    iq = iq + (np.random.randn(len(iq)) + 1j * np.random.randn(len(iq))) * 0.05
    
    engine = SynchronizationEngine()
    
    test_mods = ["BPSK", "QPSK", "8PSK", "16QAM", "64QAM", "2-FSK", "4-FSK"]
    timings = {}
    
    # Warmup
    for mod in ["QPSK", "2-FSK"]:
        _ = engine.synchronize(
            iq=iq,
            sample_rate_hz=fs,
            symbol_rate_hz=baud,
            modulation=mod,
        )
        
    for mod in test_mods:
        t0 = time.perf_counter()
        n_iters = 30
        for _ in range(n_iters):
            res = engine.synchronize(
                iq=iq,
                sample_rate_hz=fs,
                symbol_rate_hz=baud,
                modulation=mod,
            )
        dt = (time.perf_counter() - t0) / n_iters * 1000.0 # ms
        timings[mod] = dt
        print(f"Modulation Hypothesis [{mod:<7}]: {dt:.2f} ms per candidate")
        
    avg_sync_ms = np.mean(list(timings.values()))
    print(f"\nAverage Single Hypothesis Sync Time: {avg_sync_ms:.2f} ms")
    
    print("\n--- Downstream Candidate Beam Latency Projection ---")
    print(f"{'Beam Width (K)':<18} | {'Hypotheses (K x 3 Bauds)':<26} | {'Total Stage 6 Latency (ms)':<28}")
    print("-" * 76)
    for k in [3, 4, 5, 6, 8, 10]:
        total_hyps = k * 3
        total_time_ms = total_hyps * avg_sync_ms
        print(f"{k:<18} | {total_hyps:<26} | {total_time_ms:.2f} ms")


if __name__ == "__main__":
    benchmark_sync_runtime()
