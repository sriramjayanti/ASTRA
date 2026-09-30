"""
Demonstration and Fixture Generation Script for ASTRA Synthetic Engine 7.
Runs the entire 7-stage synthetic pipeline, produces 8 standard capture format variations,
generates ingestion test fixtures, verifies round-trip reading, and outputs dataset manifests.
"""

from pathlib import Path
import numpy as np

from astra_synthetic.payload.generator import PayloadGenerator
from astra_synthetic.framing.generator import FrameGenerator
from astra_synthetic.fec.generator import FECGenerator
from astra_synthetic.interleaving.generator import InterleaverGenerator
from astra_synthetic.modulation.generator import ModulationGenerator
from astra_synthetic.channel.generator import ChannelGenerator
from astra_synthetic.capture.generator import CaptureGenerator
from astra_synthetic.capture.raw_iq import read_raw_iq_file
from astra_synthetic.capture.wav_iq import read_wav_iq_file
from astra_synthetic.capture.sigmf_writer import read_sigmf_iq_file
from astra_synthetic.capture.serializers import write_dataset_manifest


def run_demonstration():
    print("=" * 85)
    print("ASTRA SYNTHETIC ENGINE 7 — SAMPLING / CAPTURE / IQ-WAV GENERATOR DEMO")
    print("=" * 85)

    # 1. Execute Upstream Chain (Engines 1 through 6)
    p_gen = PayloadGenerator()
    f_gen = FrameGenerator()
    fec_gen = FECGenerator()
    int_gen = InterleaverGenerator()
    mod_gen = ModulationGenerator()
    chan_gen = ChannelGenerator()
    cap_gen = CaptureGenerator()

    # Create source data chain
    payload = p_gen.generate(payload_type="counter", bit_length=512, seed=42)
    frame = f_gen.generate(payload)
    fec_rec = fec_gen.encode(frame, scheme="none")
    int_rec = int_gen.interleave(fec_rec, scheme="none")
    clean_mod = mod_gen.modulate(
        int_rec,
        modulation_type="qpsk",
        sample_rate=192000.0,
        symbol_rate=24000.0,
        samples_per_symbol=8,
        pulse_shaping="rrc",
        rrc_alpha=0.35,
    )
    # Apply combined realistic channel impairments
    chan_rec = chan_gen.apply(
        clean_mod,
        profile_name="combined_medium",
        seed=1001,
        overrides={"snr_db": 15.0, "cfo_hz": 1500.0},
    )

    print(f"Source Chain:")
    print(f"  Payload ID:     {payload.payload_id}")
    print(f"  Frame ID:       {frame.frame_id}")
    print(f"  Modulation ID:  {clean_mod.modulation_record_id} ({clean_mod.modulation_type.upper()})")
    print(f"  Channel ID:     {chan_rec.channel_record_id} (SNR: {chan_rec.snr_db_measured:.1f} dB, CFO: {chan_rec.cfo_hz:+.0f} Hz)")
    print(f"  Impaired IQ:    {len(chan_rec.impaired_iq)} complex samples ({chan_rec.sample_rate:,.0f} Hz)")
    print("-" * 85)

    # 2. Generate 8 Demonstration Capture Formats
    scenarios = [
        ("A. float32 LE IQ", "raw_f32_le_iq", {}),
        ("B. float32 BE IQ", "raw_f32_be_iq", {}),
        ("C. int16 LE IQ", "raw_i16_le_iq", {}),
        ("D. int16 LE QI", "raw_i16_le_qi", {}),
        ("E. int8 LE IQ", "raw_i8_iq", {}),
        ("F. stereo int16 WAV (IQ)", "wav_i16_iq", {}),
        ("G. stereo int16 WAV (QI)", "wav_i16_qi", {}),
        ("H. SigMF float32 LE", "sigmf_cf32_le", {"center_frequency_hz": 433920000.0}),
    ]

    out_base = Path("output/demo_captures")
    out_base.mkdir(parents=True, exist_ok=True)

    records = []

    print(f"{'#':<3} | {'SCENARIO':<26} | {'FORMAT':<8} | {'DTYPE':<8} | {'ENDIAN':<7} | {'ORDER':<6} | {'BYTES':<8} | {'QSNR (dB)':<10} | {'ROUND-TRIP'}")
    print("-" * 115)

    for idx, (label, prof_name, overrides) in enumerate(scenarios, start=1):
        cap_rec = cap_gen.write(
            chan_rec,
            profile_name=prof_name,
            output_dir=out_base / "files",
            seed=42 + idx * 79,
            overrides=overrides,
        )
        records.append(cap_rec)

        # 3. Perform Reference Reader Round-trip Validation
        fp = Path(cap_rec.file_path)
        if cap_rec.file_format == "raw_iq":
            recovered = read_raw_iq_file(
                fp,
                storage_dtype=cap_rec.storage_dtype,
                endianness=cap_rec.endianness,
                iq_order=cap_rec.iq_order,
                quantization_scale=cap_rec.quantization_scale,
            )
        elif cap_rec.file_format == "wav":
            recovered, _ = read_wav_iq_file(
                fp,
                iq_order=cap_rec.iq_order,
                quantization_scale=cap_rec.quantization_scale,
            )
        elif cap_rec.file_format == "sigmf":
            meta_p = fp.parent / f"{fp.stem}.sigmf-meta"
            recovered, _ = read_sigmf_iq_file(
                meta_p,
                quantization_scale=cap_rec.quantization_scale,
            )

        # Evaluate roundtrip match
        if cap_rec.storage_dtype == "float32":
            rt_valid = np.allclose(chan_rec.impaired_iq, recovered, atol=1e-5)
        elif cap_rec.storage_dtype == "int16":
            rt_valid = np.allclose(chan_rec.impaired_iq, recovered, atol=1e-3)
        else:  # int8
            rt_valid = np.allclose(chan_rec.impaired_iq, recovered, atol=0.1)

        rt_status = "PASS (Bit-exact)" if cap_rec.storage_dtype == "float32" and rt_valid else ("PASS (Quantized)" if rt_valid else "FAIL")
        qsnr_str = f"{cap_rec.quantization_snr_db:.1f}" if cap_rec.quantization_snr_db is not None else "inf"

        print(f"{idx:<3} | {label:<26} | {cap_rec.file_format:<8} | {cap_rec.storage_dtype:<8} | {cap_rec.endianness:<7} | {cap_rec.iq_order:<6} | {cap_rec.file_size_bytes:<8} | {qsnr_str:<10} | {rt_status}")

    print("-" * 115)

    # 4. Generate Master Manifest CSV
    manifest_p = out_base / "manifest.csv"
    write_dataset_manifest(records, manifest_p)
    print(f"\n[OK] Generated master manifest CSV: {manifest_p}")

    # 5. Generate Dedicated Ingestion Test Fixtures
    fixtures_dir = Path("fixtures")
    fixtures_dir.mkdir(parents=True, exist_ok=True)

    fixture_configs = [
        ("f32_le_iq", "raw_f32_le_iq", {}),
        ("f32_be_iq", "raw_f32_be_iq", {}),
        ("i16_le_iq", "raw_i16_le_iq", {}),
        ("i16_le_qi", "raw_i16_le_qi", {}),
        ("wav_i16_iq", "wav_i16_iq", {}),
        ("missing_samplerate", "missing_samplerate_test", {}),
    ]

    print("\nGenerating Ingestion Test Fixtures:")
    for fix_name, prof, ovr in fixture_configs:
        fix_path = fixtures_dir / fix_name
        fix_path.mkdir(parents=True, exist_ok=True)
        rec = cap_gen.write(
            chan_rec,
            profile_name=prof,
            output_dir=fix_path,
            filename=f"fixture_{fix_name}.iq" if "wav" not in fix_name else f"fixture_{fix_name}.wav",
            overrides=ovr,
        )
        print(f"  + [{fix_name:<20}] -> {rec.file_path}")

    print(f"\n[OK] All Ingestion Test Fixtures successfully generated in: {fixtures_dir.resolve()}")
    print("=" * 85)


if __name__ == "__main__":
    run_demonstration()
