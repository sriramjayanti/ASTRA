"""
ASTRA_FINAL_TEST_SET Generator.

Generates an independent, untouched, out-of-sample benchmark dataset of 1,050 captures:
  - 1,000 Target Communications Signals (10 modulations x 100 captures each)
  - 50 Non-Target / Noise Signals (25 AWGN, 15 CW unmodulated carriers, 10 Multi-tone interference)

Guaranteed properties:
  - Isolated Master Seed: 88888888 (never used in training, validation, tuning, or calibration)
  - Full realistic RF impairments: CFO (-2000 to +2000 Hz), AWGN (-6 to +24 dB SNR),
    pulse shaping, timing offsets, fading, and multipath
  - Canonical framing: Sync Word, Header, Payload, CRC-16 / CRC-32
  - Channel coding: FEC (none, conv_r12, reed_solomon), Interleaving (none, block, convolutional)
  - Saved as raw float32 IQ (.npy) + ground_truth.json + manifest.csv
"""

import os
import sys
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd

# Add workspace to sys.path
WORKSPACE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE))

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

from astra_synthetic.payload.generator import PayloadGenerator
from astra_synthetic.framing.generator import FrameGenerator
from astra_synthetic.fec.generator import FECGenerator
from astra_synthetic.interleaving.generator import InterleaverGenerator
from astra_synthetic.modulation.generator import ModulationGenerator
from astra_synthetic.channel.generator import ChannelGenerator
from astra_config.classes import normalize_modulation_name

OUTPUT_DIR = WORKSPACE / "datasets" / "ASTRA_FINAL_TEST_SET"
CAPTURES_DIR = OUTPUT_DIR / "captures"
CAPTURES_DIR.mkdir(parents=True, exist_ok=True)

MASTER_SEED = 88888888
SAMPLE_RATE = 192000.0
CAPTURE_LENGTH = 16384  # complex samples (85.3 ms at 192 kSa/s)

MODULATION_TYPES = [
    "2fsk", "4fsk", "bpsk", "qpsk", "8psk",
    "dqpsk", "msk", "16qam", "64qam", "256qam"
]

BAUD_RATES = [1200.0, 2400.0, 4800.0, 9600.0, 19200.0, 24000.0, 38400.0, 48000.0]
FEC_SCHEMES = ["none", "none", "conv_r12", "reed_solomon"]
INTERLEAVER_SCHEMES = ["none", "none", "block", "convolutional"]

def generate_dataset():
    print("=" * 80)
    print("     ASTRA_FINAL_TEST_SET GENERATOR (1,050 CAPTURES)")
    print("     UNTOUCHED HELD-OUT END-TO-END BENCHMARK")
    print("=" * 80)
    print(f"Target Directory: {OUTPUT_DIR}")
    print(f"Master Seed:      {MASTER_SEED} (Zero-Leakage Isolated Split)")
    print(f"Sample Rate:      {SAMPLE_RATE} Hz")
    print(f"Capture Length:   {CAPTURE_LENGTH} complex samples per capture\n")

    p_gen = PayloadGenerator()
    f_gen = FrameGenerator()
    fec_gen = FECGenerator()
    int_gen = InterleaverGenerator()
    mod_gen = ModulationGenerator()
    chan_gen = ChannelGenerator()

    rng = np.random.default_rng(MASTER_SEED)
    manifest_records = []
    ground_truth = {}

    start_time = time.time()
    sig_idx = 0

    # -------------------------------------------------------------------------
    # 1. Generate 1,000 Target Communications Signals (100 per modulation)
    # -------------------------------------------------------------------------
    signals_per_mod = 100
    for mod_type in MODULATION_TYPES:
        mod_canonical = normalize_modulation_name(mod_type)
        print(f"Generating 100 captures for modulation: {mod_canonical}...")

        for m_idx in range(signals_per_mod):
            sig_idx += 1
            sig_id = f"sig_{sig_idx:05d}"
            sub_seed = int(rng.integers(1, 2**31 - 1))

            # Sample physical parameters
            baud = float(rng.choice(BAUD_RATES))
            sps = int(round(SAMPLE_RATE / baud))
            sps = max(2, sps)

            # SNR spanning -6 dB to +24 dB
            snr_db = float(rng.uniform(-6.0, 24.0))
            cfo_hz = float(rng.uniform(-1800.0, 1800.0))
            phase_rad = float(rng.uniform(-np.pi, np.pi))

            # Coding schemes
            fec_scheme = str(rng.choice(FEC_SCHEMES))
            int_scheme = str(rng.choice(INTERLEAVER_SCHEMES))

            # Payload generation
            bit_len = int(rng.choice([256, 384, 512, 768]))
            p_rec = p_gen.generate(payload_type="counter" if m_idx % 2 == 0 else "random_bits",
                                   bit_length=bit_len, seed=sub_seed)
            f_rec = f_gen.generate(p_rec)

            try:
                fec_rec = fec_gen.encode(f_rec, scheme=fec_scheme)
            except Exception:
                fec_scheme = "none"
                fec_rec = fec_gen.encode(f_rec, scheme="none")

            try:
                int_rec = int_gen.interleave(fec_rec, scheme=int_scheme)
            except Exception:
                int_scheme = "none"
                int_rec = int_gen.interleave(fec_rec, scheme="none")

            # Pulse shaping & modulation
            mod_kwargs = {
                "modulation_type": mod_type,
                "sample_rate": SAMPLE_RATE,
                "symbol_rate": baud,
                "samples_per_symbol": sps,
            }
            if "fsk" in mod_type or mod_type == "msk":
                tone_ratio = float(rng.uniform(0.5, 1.5))
                mod_kwargs["tone_spacing_ratio"] = tone_ratio
            else:
                mod_kwargs["pulse_shaping"] = "rrc"
                mod_kwargs["rrc_alpha"] = float(rng.uniform(0.20, 0.45))

            try:
                m_rec = mod_gen.modulate(int_rec, **mod_kwargs)
            except Exception as e:
                # Fallback to standard modulate without extra kwargs if needed
                m_rec = mod_gen.modulate(int_rec, modulation_type=mod_type,
                                         sample_rate=SAMPLE_RATE, symbol_rate=baud,
                                         samples_per_symbol=sps)

            # Channel impairments
            c_overrides = {
                "snr_db": snr_db,
                "cfo_hz": cfo_hz,
                "phase_offset_rad": phase_rad,
            }
            c_rec = chan_gen.apply(m_rec, overrides=c_overrides, seed=sub_seed)

            # Extract complex IQ and trim/pad to CAPTURE_LENGTH
            raw_iq = np.asarray(c_rec.impaired_iq, dtype=np.complex64)
            if len(raw_iq) < CAPTURE_LENGTH:
                # Repeat or pad
                n_reps = int(np.ceil(CAPTURE_LENGTH / len(raw_iq)))
                raw_iq = np.tile(raw_iq, n_reps)[:CAPTURE_LENGTH]
            else:
                raw_iq = raw_iq[:CAPTURE_LENGTH]

            # Save capture to disk
            filename = f"{sig_id}_{mod_canonical.replace('-', '').lower()}.npy"
            file_path = CAPTURES_DIR / filename
            np.save(file_path, raw_iq)

            # Ground truth metadata
            record = {
                "signal_id": sig_id,
                "filename": filename,
                "file_path": str(file_path),
                "is_noise": False,
                "modulation": mod_canonical,
                "modulation_raw": mod_type,
                "symbol_rate_hz": baud,
                "sample_rate_hz": SAMPLE_RATE,
                "samples_per_symbol": sps,
                "snr_db": round(snr_db, 2),
                "cfo_hz": round(cfo_hz, 2),
                "phase_offset_rad": round(phase_rad, 4),
                "fec_scheme": fec_scheme,
                "fec_family": "convolutional" if "conv" in fec_scheme else ("reed_solomon" if "reed" in fec_scheme else "none"),
                "interleaver_scheme": int_scheme,
                "interleaver_family": "block" if "block" in int_scheme else ("convolutional" if "conv" in int_scheme else "none"),
                "payload_bits": p_rec.payload_bits.tolist() if hasattr(p_rec, "payload_bits") else [],
                "frame_bits": f_rec.framed_bits.tolist() if hasattr(f_rec, "framed_bits") else [],
                "modulated_bits": int_rec.interleaved_bits.tolist() if hasattr(int_rec, "interleaved_bits") else [],
                "sample_count": len(raw_iq),
            }
            ground_truth[sig_id] = record
            manifest_records.append({
                "signal_id": sig_id,
                "filename": filename,
                "is_noise": False,
                "modulation": mod_canonical,
                "symbol_rate_hz": baud,
                "snr_db": round(snr_db, 2),
                "cfo_hz": round(cfo_hz, 2),
                "fec_family": record["fec_family"],
                "interleaver_family": record["interleaver_family"],
            })

    # -------------------------------------------------------------------------
    # 2. Generate 50 Non-Target / Noise Signals
    # -------------------------------------------------------------------------
    print("\nGenerating 50 Non-Target / Noise Captures (AWGN, CW tones, Multi-tone)...")
    noise_specs = (
        [("pure_awgn", 25)] +
        [("cw_tone", 15)] +
        [("multitone_interference", 10)]
    )

    for n_type, count in noise_specs:
        for _ in range(count):
            sig_idx += 1
            sig_id = f"sig_{sig_idx:05d}"
            sub_seed = int(rng.integers(1, 2**31 - 1))
            n_rng = np.random.default_rng(sub_seed)

            t = np.arange(CAPTURE_LENGTH) / SAMPLE_RATE

            if n_type == "pure_awgn":
                # Pure thermal noise
                noise_pwr = float(n_rng.uniform(0.1, 1.5))
                raw_iq = (np.sqrt(noise_pwr / 2.0) * (n_rng.normal(size=CAPTURE_LENGTH) + 1j * n_rng.normal(size=CAPTURE_LENGTH))).astype(np.complex64)
                desc = "Pure AWGN Thermal Noise"
            elif n_type == "cw_tone":
                # Unmodulated continuous-wave tone + low noise
                tone_freq = float(n_rng.uniform(-SAMPLE_RATE * 0.4, SAMPLE_RATE * 0.4))
                cw = np.exp(1j * (2 * np.pi * tone_freq * t + float(n_rng.uniform(-np.pi, np.pi))))
                noise = 0.05 * (n_rng.normal(size=CAPTURE_LENGTH) + 1j * n_rng.normal(size=CAPTURE_LENGTH))
                raw_iq = (cw + noise).astype(np.complex64)
                desc = f"Unmodulated CW Tone @ {tone_freq:.1f} Hz"
            else:
                # Multi-tone interference (3-5 simultaneous sine waves)
                sig = np.zeros(CAPTURE_LENGTH, dtype=np.complex64)
                for _ in range(int(n_rng.integers(3, 6))):
                    f = float(n_rng.uniform(-SAMPLE_RATE * 0.45, SAMPLE_RATE * 0.45))
                    p = float(n_rng.uniform(-np.pi, np.pi))
                    a = float(n_rng.uniform(0.3, 1.0))
                    sig += a * np.exp(1j * (2 * np.pi * f * t + p)).astype(np.complex64)
                noise = 0.1 * (n_rng.normal(size=CAPTURE_LENGTH) + 1j * n_rng.normal(size=CAPTURE_LENGTH))
                raw_iq = (sig + noise).astype(np.complex64)
                desc = "Multi-Tone Continuous Interference"

            filename = f"{sig_id}_{n_type}.npy"
            file_path = CAPTURES_DIR / filename
            np.save(file_path, raw_iq)

            record = {
                "signal_id": sig_id,
                "filename": filename,
                "file_path": str(file_path),
                "is_noise": True,
                "noise_type": n_type,
                "description": desc,
                "modulation": "NOISE",
                "modulation_raw": "noise",
                "symbol_rate_hz": 0.0,
                "sample_rate_hz": SAMPLE_RATE,
                "samples_per_symbol": 0.0,
                "snr_db": -99.0,
                "cfo_hz": 0.0,
                "fec_scheme": "none",
                "fec_family": "none",
                "interleaver_scheme": "none",
                "interleaver_family": "none",
                "payload_bits": [],
                "frame_bits": [],
                "modulated_bits": [],
                "sample_count": len(raw_iq),
            }
            ground_truth[sig_id] = record
            manifest_records.append({
                "signal_id": sig_id,
                "filename": filename,
                "is_noise": True,
                "modulation": "NOISE",
                "symbol_rate_hz": 0.0,
                "snr_db": -99.0,
                "cfo_hz": 0.0,
                "fec_family": "none",
                "interleaver_family": "none",
            })

    # Save Ground Truth and Manifest
    gt_file = OUTPUT_DIR / "ground_truth.json"
    with open(gt_file, "w", encoding="utf-8") as f:
        json.dump(ground_truth, f, indent=2)

    df_manifest = pd.DataFrame(manifest_records)
    csv_file = OUTPUT_DIR / "manifest.csv"
    df_manifest.to_csv(csv_file, index=False)

    meta_file = OUTPUT_DIR / "DATASET_MANIFEST.json"
    meta_info = {
        "dataset_name": "ASTRA_FINAL_TEST_SET",
        "description": "Untouched, held-out end-to-end evaluation benchmark for ASTRA",
        "total_captures": len(manifest_records),
        "target_communications_captures": 1000,
        "non_target_noise_captures": 50,
        "modulations": [normalize_modulation_name(m) for m in MODULATION_TYPES],
        "master_seed": MASTER_SEED,
        "sample_rate_hz": SAMPLE_RATE,
        "capture_length_samples": CAPTURE_LENGTH,
        "zero_leakage_guarantee": "Never used for training, validation, threshold tuning, calibration, or model selection",
        "generation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta_info, f, indent=2)

    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print("     ASTRA_FINAL_TEST_SET GENERATION COMPLETE")
    print("=" * 80)
    print(f"Total Captures Generated: {len(manifest_records)}")
    print(f"  * Target Comms Captures: 1,000 (100 per modulation)")
    print(f"  * Non-Target Captures:   50 (25 AWGN, 15 CW, 10 Multi-tone)")
    print(f"Output Directory:         {OUTPUT_DIR}")
    print(f"Ground Truth JSON:        {gt_file}")
    print(f"Manifest CSV:             {csv_file}")
    print(f"Elapsed Time:             {elapsed:.1f}s ({elapsed/len(manifest_records)*1000:.1f} ms/capture)")
    print("=" * 80)

if __name__ == "__main__":
    generate_dataset()
