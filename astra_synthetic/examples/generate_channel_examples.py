"""
Demonstration script for ASTRA Synthetic Engine 6 — RF / Channel Impairment Generator.
Generates 10 representative channel scenarios from a pristine QPSK waveform,
creates debug visualization plots, prints a comparative parameter report, and serializes records.
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from astra_synthetic.payload.generator import PayloadGenerator
from astra_synthetic.framing.generator import FrameGenerator
from astra_synthetic.fec.generator import FECGenerator
from astra_synthetic.interleaving.generator import InterleaverGenerator
from astra_synthetic.modulation.generator import ModulationGenerator
from astra_synthetic.channel.generator import ChannelGenerator
from astra_synthetic.channel.serializers import save_channel_batch


def run_demonstration():
    print("=" * 80)
    print("ASTRA SYNTHETIC ENGINE 6 — RF / CHANNEL IMPAIRMENT GENERATION DEMO")
    print("=" * 80)

    # 1. Generate clean source QPSK modulation record from Engines 1-5
    p_gen = PayloadGenerator()
    f_gen = FrameGenerator()
    fec_gen = FECGenerator()
    int_gen = InterleaverGenerator()
    mod_gen = ModulationGenerator()

    payload_rec = p_gen.generate(payload_type="counter", bit_length=512, seed=42)
    frame_rec = f_gen.generate(payload_rec)
    fec_rec = fec_gen.encode(frame_rec, scheme="none")
    int_rec = int_gen.interleave(fec_rec, scheme="none")
    clean_mod_record = mod_gen.modulate(
        int_rec,
        modulation_type="qpsk",
        sample_rate=192000.0,
        symbol_rate=24000.0,
        samples_per_symbol=8,
        pulse_shaping="rrc",
        rrc_alpha=0.35,
    )

    print(f"Base Modulation Record: {clean_mod_record.modulation_record_id}")
    print(f"Modulation Type:        {clean_mod_record.modulation_type.upper()}")
    print(f"Sample Rate:            {clean_mod_record.sample_rate:,.1f} Hz")
    print(f"Clean IQ Sample Count:  {len(clean_mod_record.clean_iq)}")
    print(f"Clean IQ SHA-256:       {clean_mod_record.iq_sha256[:16]}...")
    print("-" * 80)

    # 2. Configure 10 demonstration scenarios
    scenarios = [
        ("A. Clean Control", "clean", {}),
        ("B. AWGN Only (0 dB)", "awgn_0db", {}),
        ("C. CFO Only (+2.5 kHz)", "cfo_only", {"cfo_hz": 2500.0}),
        ("D. Phase Offset Only (+1.57 rad)", "phase_only", {"phase_offset_rad": np.pi / 2.0}),
        ("E. Timing Offset Only (+0.35 samples)", "timing_only", {"timing_offset_samples": 0.35}),
        ("F. Rayleigh Fading", "rayleigh_flat", {}),
        ("G. Rician Fading (K=8 dB)", "rician_mild", {"fading": {"type": "rician", "k_factor_db": 8.0}}),
        ("H. Multipath (Mild)", "multipath_mild", {}),
        ("I. Combined Medium", "combined_medium", {"snr_db": 10.0, "cfo_hz": 1200.0}),
        ("J. Satellite-Like (LEO)", "satellite_like_v1", {}),
    ]

    chan_gen = ChannelGenerator()
    channel_records = []

    print(f"{'#':<3} | {'SCENARIO':<30} | {'REC ID':<12} | {'SNR (dB)':<9} | {'CFO (Hz)':<9} | {'PHASE':<7} | {'TIMING':<7} | {'FADING':<9} | {'MULTIPATH':<9} | {'OUT POWER':<10}")
    print("-" * 115)

    for idx, (label, prof_name, overrides) in enumerate(scenarios, start=1):
        rec = chan_gen.apply(
            clean_mod_record,
            profile_name=prof_name,
            seed=42 + idx * 101,
            overrides=overrides,
        )
        channel_records.append((label, rec))

        snr_str = f"{rec.snr_db_measured:.1f}" if rec.snr_db_measured is not None else "inf (clean)"
        cfo_str = f"{rec.cfo_hz:+.0f}" if abs(rec.cfo_hz) > 1e-3 else "0"
        phase_str = f"{rec.phase_offset_rad:+.2f}" if abs(rec.phase_offset_rad) > 1e-3 else "0"
        timing_str = f"{rec.timing_offset_samples:+.2f}" if abs(rec.timing_offset_samples) > 1e-3 else "0"
        fading_str = rec.fading_type
        mp_str = "YES" if rec.multipath_enabled else "NO"
        p_out = float(np.mean(np.abs(rec.impaired_iq) ** 2))

        print(f"{idx:<3} | {label:<30} | {rec.channel_record_id:<12} | {snr_str:<9} | {cfo_str:<9} | {phase_str:<7} | {timing_str:<7} | {fading_str:<9} | {mp_str:<9} | {p_out:<10.4f}")

    print("-" * 115)

    # 3. Generate Multi-panel Visualization Plot
    output_dir = Path("output/demo_channel")
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(4, 3, figsize=(18, 16))
    fig.suptitle("ASTRA Synthetic Engine 6 — Baseband RF Impairment Showcase", fontsize=16, fontweight="bold")

    plot_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    for i, idx in enumerate(plot_indices):
        ax = axes[i // 3, i % 3]
        label, rec = channel_records[idx]

        iq = rec.impaired_iq
        # Plot constellation (subsample for clarity)
        iq_sub = iq[len(iq)//4 : len(iq)//4 + 600]
        ax.scatter(iq_sub.real, iq_sub.imag, alpha=0.6, s=12, edgecolors="none", color="crimson" if "Combined" in label or "Satellite" in label else "navy")
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.6)
        ax.axvline(0, color="gray", linestyle="--", linewidth=0.6)
        ax.set_title(f"{label}\n(ID: {rec.channel_record_id})", fontsize=10, fontweight="bold")
        ax.set_xlabel("In-Phase (I)")
        ax.set_ylabel("Quadrature (Q)")
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-2.5, 2.5)
        ax.grid(True, linestyle=":", alpha=0.5)

    # 11th Panel: Power Spectral Density comparison (Clean vs Impaired)
    ax_psd = axes[3, 1]
    Fs = clean_mod_record.sample_rate
    clean_pxx, clean_f = plt.psd(clean_mod_record.clean_iq, NFFT=512, Fs=Fs/1000)
    sat_rec = channel_records[9][1]
    sat_pxx, sat_f = plt.psd(sat_rec.impaired_iq, NFFT=512, Fs=Fs/1000)
    ax_psd.clear()
    ax_psd.plot(clean_f, 10 * np.log10(np.maximum(clean_pxx, 1e-12)), color="blue", label="Clean QPSK", alpha=0.7)
    ax_psd.plot(sat_f, 10 * np.log10(np.maximum(sat_pxx, 1e-12)), color="red", label="Satellite Impaired", alpha=0.7)
    ax_psd.set_title("PSD Comparison: Clean vs Satellite (kHz)", fontsize=10, fontweight="bold")
    ax_psd.set_xlabel("Frequency (kHz)")
    ax_psd.set_ylabel("Power / Frequency (dB/Hz)")
    ax_psd.legend(loc="upper right", fontsize=8)
    ax_psd.grid(True, linestyle=":", alpha=0.5)

    # 12th Panel: Impulse Response for Multipath Channel
    ax_cir = axes[3, 2]
    mp_rec = channel_records[7][1]
    if mp_rec.multipath_taps is not None and mp_rec.multipath_delays_samples is not None:
        delays = mp_rec.multipath_delays_samples
        amps = np.abs(mp_rec.multipath_taps)
        ax_cir.stem(delays, amps, linefmt="teal", markerfmt="o", basefmt="gray")
        ax_cir.set_title("Mild Multipath Impulse Response |h[n]|", fontsize=10, fontweight="bold")
        ax_cir.set_xlabel("Delay (samples)")
        ax_cir.set_ylabel("Magnitude")
        ax_cir.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    plot_path = output_dir / "channel_impairments_demo.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"\n[OK] Demonstration multi-panel plot saved to: {plot_path}")

    # 4. Serialize channel records to disk
    just_records = [r for _, r in channel_records]
    saved_paths = save_channel_batch(just_records, output_dir=output_dir / "records", compact=False)
    print(f"[OK] Successfully saved {len(saved_paths)} ChannelRecords to: {output_dir / 'records'}")
    print("=" * 80)


if __name__ == "__main__":
    run_demonstration()
