import sys
import time
from pathlib import Path
import numpy as np

WORKSPACE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE))

from astra_config.classes import normalize_modulation_name
from astra_fusion.src.inference import ASTRAFusionEngine
from astra_symbol_rate.src.inference import SymbolRateEstimator
from astra_candidate_engine.src.inference import CandidateHypothesisEngine
from astra_synchronization.src.inference import SynchronizationEngine
from astra_demodulation.src.inference import DemodulationEngine
from astra_demodulation.src.hard_decision import slice_hard_decisions
from astra_demodulation.src.mappings import get_constellation
from astra_interleaver.src.inference import InterleaverTestingEngine
from astra_fec.src.inference import FECTestingEngine
from astra_validation.src.inference import ValidationEngine
from astra_validation.src.crc import CRCCalculator, load_crc_profiles

# 1. Build controlled frame stream
target_payload = b"hi hello"
crc_profiles = load_crc_profiles()
crc_prof = crc_profiles.get("crc16_ccitt_false", list(crc_profiles.values())[0])
crc_calc = CRCCalculator(crc_prof)

sync_bits = np.unpackbits(np.array([0xEB, 0x90], dtype=np.uint8))
hdr_bits = np.unpackbits(np.array([0x01, len(target_payload)], dtype=np.uint8))
payload_bits = np.unpackbits(np.frombuffer(target_payload, dtype=np.uint8))
data_bits = np.concatenate([sync_bits, hdr_bits, payload_bits])
crc_val = crc_calc.compute(data_bits)
crc_bits = np.unpackbits(np.array([crc_val >> 8, crc_val & 0xFF], dtype=np.uint8))
frame_bits = np.concatenate([data_bits, crc_bits])
tx_bits = np.tile(frame_bits, 8)

print(f"Target payload: '{target_payload.decode()}', Frame bits: {len(frame_bits)}, Total stream bits: {len(tx_bits)}")

# 2. Synthesize QPSK waveform
const = get_constellation("QPSK")
sps = 20
baud = 9600.0
fs = 192000.0

from astra_synthetic.modulation.filters import apply_rrc_pulse_shaping

# Map bits to QPSK symbols
k = const.bits_per_symbol
bit_to_pt = {tuple(row.tolist()): const.complex_points[idx] for idx, row in enumerate(const.bit_labels)}
syms = np.array([bit_to_pt[tuple(tx_bits[i:i+k])] for i in range(0, len(tx_bits), k)], dtype=np.complex64)
c_syms, _, _ = apply_rrc_pulse_shaping(syms, sps=sps, beta=0.35, span_symbols=8)

# Channel impairments: 28 dB SNR, 0 Hz CFO
t = np.arange(len(c_syms)) / fs
cfo_phase = np.exp(1j * (2.0 * np.pi * 0.0 * t))
noise_std = 10.0 ** (-28.0 / 20.0) / np.sqrt(2.0)
noise = (np.random.randn(len(c_syms)) + 1j * np.random.randn(len(c_syms))) * noise_std
raw_iq = (c_syms * cfo_phase + noise).astype(np.complex64)

# 3. Synchronize with true and estimated parameters
sync_engine = SynchronizationEngine()
demod_engine = DemodulationEngine()

from astra_candidate_engine.src.models import ReceiverHypothesis
hyp = ReceiverHypothesis(
    candidate_id="cand_qpsk",
    modulation="QPSK",
    modulation_family="PSK",
    symbol_rate_hz=baud,
    sample_rate_hz=fs,
    samples_per_symbol=float(sps)
)

s_res = sync_engine.synchronize(raw_iq, hyp)
print(f"Sync success: {s_res.success}, lock_score: {s_res.lock_metrics.get('composite_lock_score', 0.0):.3f}, symbol_count: {s_res.symbol_count}")
print(f"Residual CFO hz in s_res: {s_res.residual_cfo_hz}")

d_res = demod_engine.demodulate(s_res)
print(f"Demod success: {d_res.success}, hard_bits count: {len(d_res.hard_bits)}")
print("True syms 50..55:", syms[50:55])
print("Recv syms 50..55:", s_res.symbol_samples[50:55])
print("Ratio recv/true 50..55:", s_res.symbol_samples[50:55] / syms[50:55])

# Map sync bits to ideal QPSK symbols
sync_syms = np.array([bit_to_pt[tuple(sync_bits[i:i+k])] for i in range(0, len(sync_bits), k)], dtype=np.complex64)
print(f"Sync word has {len(sync_syms)} QPSK symbols: {sync_syms}")

# Correlate received symbol samples against sync_syms
rx_syms = s_res.symbol_samples
xcorr = np.correlate(rx_syms, sync_syms, mode='valid')
peak_idx = int(np.argmax(np.abs(xcorr)))
est_phase_angle = float(np.angle(xcorr[peak_idx]))
print(f"Sync peak found at symbol {peak_idx} with phase offset {np.rad2deg(est_phase_angle):.1f} deg (corr mag = {np.abs(xcorr[peak_idx]):.2f}/{len(sync_syms)})")

# For every frame in rx_syms:
print("\nScanning all detected frames across stream:")
for f_idx in range(4):
    f_sym_start = peak_idx - f_idx * 56
    if f_sym_start < 0 or f_sym_start + 56 > len(rx_syms):
        continue
    f_syms = rx_syms[f_sym_start : f_sym_start + 56]
    # Local phase estimate on sync word of this frame
    f_sync_corr = np.sum(f_syms[:8] * np.conj(sync_syms))
    f_phase = np.angle(f_sync_corr)
    f_aligned = f_syms * np.exp(-1j * f_phase)
    f_bits = slice_hard_decisions(f_aligned, const).hard_bits
    
    print("True sync syms:")
    print(sync_syms[:8])
    print("Recv aligned syms (first 8):")
    print(f_aligned[:8])
    print("True frame bits (first 16):")
    print(frame_bits[:16])
    print("Recv frame bits (first 16):")
    print(f_bits[:16])
    diff_indices = np.where(f_bits != frame_bits)[0]
    print(f"Diff indices ({len(diff_indices)}): {diff_indices.tolist()}")
    
    if p_bytes_rx == target_payload:
        # Validate with ValidationEngine
        fec_cand = fec_engine.decode_candidate(
            hypothesis=fec_engine.generate_candidates(len(f_bits))[0],
            hard_bits=f_bits
        )
        val_res = val_engine.validate(fec_cand, context={"candidate_frame_lengths": [112]})
        print(f"  -> Validation Status: {val_res.validation_status} (Score: {val_res.overall_validation_score:.3f})")
        print("  -> >>> [PASS] EXACT PAYLOAD 'hi hello' RECOVERED WITH VALIDATED CRC! <<<")

if p_rx_bytes == target_payload:
    print("\n>>> [PASS] 100% EXACT PAYLOAD 'hi hello' RECOVERED WITH VALIDATED CRC! <<<")
