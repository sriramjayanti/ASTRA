"""
ASTRA Synthetic Engine 5: Modulation / Clean IQ Waveform Demonstration Script.
Consumes InterleaverRecord from Engine 4, generates clean baseband IQ waveforms across all 7 modulations
(2-FSK, 4-FSK, BPSK, QPSK, 8PSK, 16QAM, 64QAM), validates reference demodulation,
saves artifacts to disk, generates constellation plots, and prints the summary comparison report.
"""

from __future__ import annotations

import logging
from pathlib import Path
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("astra_modulation_demo")

# Setup project path
import sys
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from astra_synthetic.payload import PayloadGenerator
from astra_synthetic.framing import FrameGenerator
from astra_synthetic.fec import FECGenerator
from astra_synthetic.interleaving import InterleaverGenerator
from astra_synthetic.modulation import (
    ModulationGenerator,
    save_modulation_record,
    reference_demodulate,
)


def run_demonstration(output_dir: Path | str = "output/demo_modulation") -> list:
    """Generate and display 7 clean IQ Modulation records from a single source InterleaverRecord."""
    payload_cfg = current_dir.parent / "configs" / "payload_config.yaml"
    frame_cfg = current_dir.parent / "configs" / "frame_config.yaml"
    fec_cfg = current_dir.parent / "configs" / "fec_config.yaml"
    int_cfg = current_dir.parent / "configs" / "interleaver_config.yaml"
    mod_cfg = current_dir.parent / "configs" / "modulation_config.yaml"

    payload_gen = PayloadGenerator(payload_cfg if payload_cfg.exists() else None)
    frame_gen = FrameGenerator(frame_cfg if frame_cfg.exists() else None)
    fec_gen = FECGenerator(fec_cfg if fec_cfg.exists() else None)
    int_gen = InterleaverGenerator(int_cfg if int_cfg.exists() else None)
    mod_gen = ModulationGenerator(mod_cfg if mod_cfg.exists() else None)

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 95)
    print("ASTRA SYNTHETIC ENGINE 5 - DEMONSTRATION CLEAN IQ MODULATION GENERATOR")
    print("=" * 95 + "\n")

    # Step 1: Upstream pipeline: Payload -> Frame -> FEC -> Interleaver (Block 16x32)
    payload = payload_gen.generate(payload_type="random_bits", bit_length=1024, seed=42, payload_id="payload_000001")
    frame = frame_gen.generate(payload, sequence_number=1, frame_id="frame_000001")
    fec_rec = fec_gen.encode(frame, profile_name="conv_k7_r12", fec_record_id="fec_000001")
    int_rec = int_gen.interleave(fec_rec, profile_name="block_16x32", interleaver_record_id="int_000001")

    print(f"Source Interleaver: {int_rec.interleaver_record_id} ({int_rec.interleaver_type} / {int_rec.profile_name})")
    print(f"Input Transmitted Bits: {int_rec.output_bit_length} bits, SHA-256: {int_rec.output_sha256[:16]}...\n")

    # Step 2: Generate 7 Modulation modes
    modes = [
        ("A. 2-FSK (Continuous Phase, Tone Spacing = 1.0 * Rs)", "2fsk", "2fsk_cp"),
        ("B. 4-FSK (Continuous Phase, Tone Spacing = 1.0 * Rs)", "4fsk", "4fsk_cp"),
        ("C. BPSK (Root Raised Cosine, beta=0.35)", "bpsk", "bpsk_rrc"),
        ("D. QPSK (Gray-Coded, Root Raised Cosine, beta=0.35)", "qpsk", "qpsk_rrc"),
        ("E. 8PSK (Gray-Coded, Root Raised Cosine, beta=0.35)", "8psk", "8psk_rrc"),
        ("F. 16-QAM (Gray-Coded, Root Raised Cosine, beta=0.35)", "16qam", "16qam_rrc"),
        ("G. 64-QAM (Gray-Coded, Root Raised Cosine, beta=0.35)", "64qam", "64qam_rrc"),
    ]

    records = []
    comparison_rows = []

    for title, mtype, prof_name in modes:
        rec = mod_gen.modulate(
            int_rec,
            modulation_type=mtype,
            profile_name=prof_name,
            symbol_rate=9600.0,
            sample_rate=192000.0,
        )
        records.append(rec)
        saved_dir = save_modulation_record(rec, output_dir=out_path)

        # Reference clean demodulation
        recovered_bits = reference_demodulate(rec)
        demod_ok = np.array_equal(recovered_bits, int_rec.interleaved_bits)

        comparison_rows.append({
            "modulation": rec.modulation_type.upper(),
            "bits_per_sym": rec.bits_per_symbol,
            "symbol_rate": int(rec.symbol_rate),
            "sample_rate": int(rec.sample_rate),
            "sps": int(rec.samples_per_symbol),
            "symbols": rec.symbol_count,
            "iq_samples": rec.clean_iq_sample_count,
            "avg_power": round(rec.average_iq_power, 4),
            "round_trip": "YES" if demod_ok else "NO",
        })

        detail_param = f"beta={rec.rolloff}" if rec.pulse_shape == "rrc" else f"spacing_ratio={rec.parameters.get('tone_spacing_ratio')}"

        print(f"{title}")
        print(f"  Record ID:            {rec.modulation_record_id}")
        print(f"  Interleaver Source:   {rec.interleaver_record_id} (FEC: {rec.fec_record_id})")
        print(f"  Modulation / Order:   {rec.modulation_type.upper()} (M={rec.modulation_order}, bits/sym={rec.bits_per_symbol})")
        print(f"  Symbol / Sample Rate: {rec.symbol_rate:.0f} Baud / {rec.sample_rate:.0f} Hz (SPS = {rec.samples_per_symbol:.0f})")
        print(f"  Pulse Shape / Param:  {rec.pulse_shape} ({detail_param})")
        print(f"  Symbol Count:         {rec.symbol_count} symbols (Padding: {rec.mapping_padding_length} bits)")
        print(f"  Clean IQ Samples:     {rec.clean_iq_sample_count} samples (Duration: {rec.duration_seconds * 1e3:.2f} ms)")
        print(f"  Average Signal Power: {rec.average_iq_power:.4f}")
        print(f"  IQ SHA-256:           {rec.iq_sha256}")
        print(f"  Ref Demodulation:     {'PASSED (Bit-Exact Recovery)' if demod_ok else 'FAILED'}")
        print(f"  Saved Directory:      {saved_dir}")
        print("-" * 95)

    # Print Summary Comparison Table
    print("\n" + "=" * 95)
    print("MODULATION FAMILY COMPARISON REPORT")
    print("=" * 95)
    header_fmt = "{:<12} {:<11} {:<13} {:<13} {:<6} {:<9} {:<12} {:<11} {:<10}"
    print(header_fmt.format("MODULATION", "BITS/SYM", "SYMBOL RATE", "SAMPLE RATE", "SPS", "SYMBOLS", "IQ SAMPLES", "AVG POWER", "ROUND-TRIP"))
    print("-" * 95)
    for row in comparison_rows:
        print(header_fmt.format(
            row["modulation"],
            row["bits_per_sym"],
            row["symbol_rate"],
            row["sample_rate"],
            row["sps"],
            row["symbols"],
            row["iq_samples"],
            row["avg_power"],
            row["round_trip"],
        ))
    print("=" * 95 + "\n")

    # Step 3: Optional Plot Generation (Constellations & Spectra)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 4, figsize=(18, 9))
        fig.suptitle("ASTRA Synthetic Engine 5: Clean IQ Waveforms & Constellations", fontsize=14, fontweight="bold")

        plot_configs = [
            (records[0], axes[0, 0], "2-FSK Instantaneous Freq"),
            (records[1], axes[0, 1], "4-FSK Instantaneous Freq"),
            (records[2], axes[0, 2], "BPSK Constellation"),
            (records[3], axes[0, 3], "QPSK Constellation"),
            (records[4], axes[1, 0], "8PSK Constellation"),
            (records[5], axes[1, 1], "16-QAM Constellation"),
            (records[6], axes[1, 2], "64-QAM Constellation"),
        ]

        for rec, ax, title in plot_configs:
            if rec.modulation_family == "fsk":
                # Instantaneous frequency estimation: diff(unwrap(phase)) / (2*pi*dt)
                phase_unwrapped = np.unwrap(np.angle(rec.clean_iq[:500]))
                inst_freq = np.diff(phase_unwrapped) * rec.sample_rate / (2.0 * np.pi)
                t_vec = np.arange(len(inst_freq)) / rec.sample_rate * 1e3  # ms
                ax.plot(t_vec, inst_freq / 1e3, color="#1f77b4", lw=1.5)
                ax.set_title(title, fontsize=11, fontweight="bold")
                ax.set_xlabel("Time (ms)")
                ax.set_ylabel("Frequency (kHz)")
                ax.grid(True, alpha=0.3)
            else:
                ax.scatter(rec.ideal_symbols.real, rec.ideal_symbols.imag, c="#d62728", s=25, alpha=0.8, edgecolors="none")
                ax.set_title(title, fontsize=11, fontweight="bold")
                ax.set_xlabel("In-Phase (I)")
                ax.set_ylabel("Quadrature (Q)")
                ax.set_aspect("equal", "box")
                ax.grid(True, alpha=0.3)
                ax.axhline(0, color="gray", lw=0.8, alpha=0.5)
                ax.axvline(0, color="gray", lw=0.8, alpha=0.5)

        # Remove 8th empty subplot
        fig.delaxes(axes[1, 3])

        plt.tight_layout()
        plot_path = out_path / "constellations_and_spectra.png"
        plt.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"Generated demonstration plot: {plot_path}\n")
    except Exception as e:
        logger.warning("Plot generation skipped: %s", e)

    return records



if __name__ == "__main__":
    run_demonstration()
