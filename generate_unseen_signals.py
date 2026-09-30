"""
ASTRA Synthetic Signal Generator for Blind Testing.

Generates a realistic suite of unseen RF signal captures in standard SDR formats:
  - SigMF (.sigmf-data / .sigmf-meta)
  - Stereo WAV (.wav)
  - Raw Float32 Complex IQ (.iq)

Each signal incorporates physical baseband RF impairments:
  - Additive White Gaussian Noise (AWGN)
  - Carrier Frequency Offset (CFO)
  - Timing Jitter / Clock Offsets
  - Root-Raised Cosine (RRC) pulse shaping
  - Channel coding (Convolutional FEC, Block Interleaving)
  - Canonical sync patterns, headers, payloads, and CRC validation words.
"""

from pathlib import Path
import json
import numpy as np

from astra_synthetic.payload.generator import PayloadGenerator
from astra_synthetic.framing.generator import FrameGenerator
from astra_synthetic.fec.generator import FECGenerator
from astra_synthetic.interleaving.generator import InterleaverGenerator
from astra_synthetic.modulation.generator import ModulationGenerator
from astra_synthetic.channel.generator import ChannelGenerator
from astra_synthetic.capture.generator import CaptureGenerator

OUTPUT_DIR = Path("test_signals")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("     ASTRA SYNTHETIC ENGINE: UNSEEN TEST SIGNAL GENERATOR")
print("=" * 80)

p_gen = PayloadGenerator()
f_gen = FrameGenerator()
fec_gen = FECGenerator()
int_gen = InterleaverGenerator()
mod_gen = ModulationGenerator()
chan_gen = ChannelGenerator()
cap_gen = CaptureGenerator()

manifest = []

# -----------------------------------------------------------------------------
# Signal 1: QPSK Telemetry (SigMF format)
# -----------------------------------------------------------------------------
print("\n[1/7] Generating Signal 1: QPSK Telemetry (SigMF format)...")
p1 = p_gen.generate(payload_type="counter", bit_length=512, seed=101)
f1 = f_gen.generate(p1)
fec1 = fec_gen.encode(f1, scheme="none")
int1 = int_gen.interleave(fec1, scheme="none")
m1 = mod_gen.modulate(
    int1,
    modulation_type="qpsk",
    sample_rate=192000.0,
    symbol_rate=9600.0,
    samples_per_symbol=20,
    pulse_shaping="rrc",
    rrc_alpha=0.35,
)
c1 = chan_gen.apply(m1, profile_name="combined_medium", overrides={"snr_db": 22.0, "cfo_hz": 450.0})
cap1 = cap_gen.write(c1, profile_name="sigmf_cf32_le", output_dir=OUTPUT_DIR, filename="signal_01_qpsk_telemetry.sigmf-data")
print(f"  -> File: {cap1.file_path} ({cap1.sample_count_complex} samples, {cap1.duration_seconds:.2f}s)")
manifest.append({
    "signal_id": "signal_01",
    "filename": "signal_01_qpsk_telemetry.sigmf-data",
    "format": "SigMF (cf32_le)",
    "modulation": "QPSK",
    "symbol_rate": 9600.0,
    "sample_rate": 192000.0,
    "sps": 20,
    "snr_db": 22.0,
    "cfo_hz": 450.0,
    "fec": "none",
    "interleaver": "none",
    "description": "Clean QPSK telemetry stream in standard SigMF format with companion metadata."
})

# -----------------------------------------------------------------------------
# Signal 2: 8PSK Scrambled Burst (Stereo WAV format)
# -----------------------------------------------------------------------------
print("\n[2/7] Generating Signal 2: 8PSK Scrambled Burst (Stereo WAV format)...")
p2 = p_gen.generate(payload_type="random_bits", bit_length=768, seed=102)
f2 = f_gen.generate(p2)
fec2 = fec_gen.encode(f2, scheme="none")
int2 = int_gen.interleave(fec2, scheme="none")
m2 = mod_gen.modulate(
    int2,
    modulation_type="8psk",
    sample_rate=192000.0,
    symbol_rate=19200.0,
    samples_per_symbol=10,
    pulse_shaping="rrc",
    rrc_alpha=0.35,
)
c2 = chan_gen.apply(m2, profile_name="combined_medium", overrides={"snr_db": 18.0, "cfo_hz": -680.0})
cap2 = cap_gen.write(c2, profile_name="wav_i16_iq", output_dir=OUTPUT_DIR, filename="signal_02_8psk_burst.wav")
print(f"  -> File: {cap2.file_path} ({cap2.sample_count_complex} samples, {cap2.duration_seconds:.2f}s)")
manifest.append({
    "signal_id": "signal_02",
    "filename": "signal_02_8psk_burst.wav",
    "format": "Stereo WAV (int16 IQ)",
    "modulation": "8PSK",
    "symbol_rate": 19200.0,
    "sample_rate": 192000.0,
    "sps": 10,
    "snr_db": 18.0,
    "cfo_hz": -680.0,
    "fec": "none",
    "interleaver": "none",
    "description": "High-baud 8PSK audio-encapsulated IQ capture with negative carrier frequency offset."
})

# -----------------------------------------------------------------------------
# Signal 3: 16-QAM High Speed (Raw Float32 IQ format)
# -----------------------------------------------------------------------------
print("\n[3/7] Generating Signal 3: 16-QAM High Speed (Raw Float32 IQ format)...")
p3 = p_gen.generate(payload_type="counter", bit_length=1024, seed=103)
f3 = f_gen.generate(p3)
fec3 = fec_gen.encode(f3, scheme="none")
int3 = int_gen.interleave(fec3, scheme="none")
m3 = mod_gen.modulate(
    int3,
    modulation_type="16qam",
    sample_rate=192000.0,
    symbol_rate=24000.0,
    samples_per_symbol=8,
    pulse_shaping="rrc",
    rrc_alpha=0.25,
)
c3 = chan_gen.apply(m3, profile_name="combined_medium", overrides={"snr_db": 25.0, "cfo_hz": 210.0})
cap3 = cap_gen.write(c3, profile_name="raw_f32_le_iq", output_dir=OUTPUT_DIR, filename="signal_03_16qam_highspeed.iq")
print(f"  -> File: {cap3.file_path} ({cap3.sample_count_complex} samples, {cap3.duration_seconds:.2f}s)")
manifest.append({
    "signal_id": "signal_03",
    "filename": "signal_03_16qam_highspeed.iq",
    "format": "Raw Binary IQ (f32_le)",
    "modulation": "16-QAM",
    "symbol_rate": 24000.0,
    "sample_rate": 192000.0,
    "sps": 8,
    "snr_db": 25.0,
    "cfo_hz": 210.0,
    "fec": "none",
    "interleaver": "none",
    "description": "High spectral-efficiency 16-state quadrature amplitude modulation grid."
})

# -----------------------------------------------------------------------------
# Signal 4: 2-FSK Industrial Beacon (Stereo WAV format)
# -----------------------------------------------------------------------------
print("\n[4/7] Generating Signal 4: 2-FSK Industrial Beacon (Stereo WAV format)...")
p4 = p_gen.generate(payload_type="counter", bit_length=256, seed=104)
f4 = f_gen.generate(p4)
fec4 = fec_gen.encode(f4, scheme="none")
int4 = int_gen.interleave(fec4, scheme="none")
m4 = mod_gen.modulate(
    int4,
    modulation_type="2fsk",
    sample_rate=96000.0,
    symbol_rate=4800.0,
    samples_per_symbol=20,
    pulse_shaping="none",
)
c4 = chan_gen.apply(m4, profile_name="combined_medium", overrides={"snr_db": 15.0, "cfo_hz": 820.0})
cap4 = cap_gen.write(c4, profile_name="wav_i16_iq", output_dir=OUTPUT_DIR, filename="signal_04_2fsk_beacon.wav")
print(f"  -> File: {cap4.file_path} ({cap4.sample_count_complex} samples, {cap4.duration_seconds:.2f}s)")
manifest.append({
    "signal_id": "signal_04",
    "filename": "signal_04_2fsk_beacon.wav",
    "format": "Stereo WAV (int16 IQ)",
    "modulation": "2-FSK",
    "symbol_rate": 4800.0,
    "sample_rate": 96000.0,
    "sps": 20,
    "snr_db": 15.0,
    "cfo_hz": 820.0,
    "fec": "none",
    "interleaver": "none",
    "description": "Continuous-Phase 2-FSK supervisory beacon with 820 Hz frequency shift."
})

# -----------------------------------------------------------------------------
# Signal 5: BPSK Deep Space Probe (SigMF format)
# -----------------------------------------------------------------------------
print("\n[5/7] Generating Signal 5: BPSK Deep Space Probe (SigMF format)...")
p5 = p_gen.generate(payload_type="counter", bit_length=256, seed=105)
f5 = f_gen.generate(p5)
fec5 = fec_gen.encode(f5, scheme="convolutional")
int5 = int_gen.interleave(fec5, scheme="none")
m5 = mod_gen.modulate(
    int5,
    modulation_type="bpsk",
    sample_rate=96000.0,
    symbol_rate=2400.0,
    samples_per_symbol=40,
    pulse_shaping="rrc",
    rrc_alpha=0.5,
)
c5 = chan_gen.apply(m5, profile_name="combined_medium", overrides={"snr_db": 10.0, "cfo_hz": -350.0})
cap5 = cap_gen.write(c5, profile_name="sigmf_cf32_le", output_dir=OUTPUT_DIR, filename="signal_05_bpsk_deepspace.sigmf-data")
print(f"  -> File: {cap5.file_path} ({cap5.sample_count_complex} samples, {cap5.duration_seconds:.2f}s)")
manifest.append({
    "signal_id": "signal_05",
    "filename": "signal_05_bpsk_deepspace.sigmf-data",
    "format": "SigMF (cf32_le)",
    "modulation": "BPSK",
    "symbol_rate": 2400.0,
    "sample_rate": 96000.0,
    "sps": 40,
    "snr_db": 10.0,
    "cfo_hz": -350.0,
    "fec": "Convolutional K=7 R=1/2",
    "interleaver": "none",
    "description": "Low-SNR (10 dB) deep-space probe transmission encoded with rate 1/2 convolutional code."
})

# -----------------------------------------------------------------------------
# Signal 6: 4-FSK Telemetry (Raw Float32 IQ format)
# -----------------------------------------------------------------------------
print("\n[6/7] Generating Signal 6: 4-FSK Telemetry (Raw Float32 IQ format)...")
p6 = p_gen.generate(payload_type="random_bytes", bit_length=512, seed=106)
f6 = f_gen.generate(p6)
fec6 = fec_gen.encode(f6, scheme="none")
int6 = int_gen.interleave(fec6, scheme="none")
m6 = mod_gen.modulate(
    int6,
    modulation_type="4fsk",
    sample_rate=192000.0,
    symbol_rate=9600.0,
    samples_per_symbol=20,
    pulse_shaping="none",
)
c6 = chan_gen.apply(m6, profile_name="combined_medium", overrides={"snr_db": 16.0, "cfo_hz": -410.0})
cap6 = cap_gen.write(c6, profile_name="raw_f32_le_iq", output_dir=OUTPUT_DIR, filename="signal_06_4fsk_telemetry.iq")
print(f"  -> File: {cap6.file_path} ({cap6.sample_count_complex} samples, {cap6.duration_seconds:.2f}s)")
manifest.append({
    "signal_id": "signal_06",
    "filename": "signal_06_4fsk_telemetry.iq",
    "format": "Raw Binary IQ (f32_le)",
    "modulation": "4-FSK",
    "symbol_rate": 9600.0,
    "sample_rate": 192000.0,
    "sps": 20,
    "snr_db": 16.0,
    "cfo_hz": -410.0,
    "fec": "none",
    "interleaver": "none",
    "description": "4-ary Frequency Shift Keyed telemetry with multi-tone transitions."
})

# -----------------------------------------------------------------------------
# Signal 7: Interleaved & FEC QPSK with Mission Payload (Raw Float32 IQ)
# -----------------------------------------------------------------------------
print("\n[7/7] Generating Signal 7: Interleaved & FEC QPSK with Mission Payload (Raw IQ)...")
# Secret payload text:
secret_msg = "ASTRA RECOVERY MISSION: DATA VERIFIED"
msg_bytes = secret_msg.encode("utf-8")
msg_bits = np.unpackbits(np.frombuffer(msg_bytes, dtype=np.uint8))
p7 = p_gen.generate(payload_type="random_bits", bit_length=len(msg_bits), seed=107)
p7.bits = msg_bits
p7.bit_length = len(msg_bits)

f7 = f_gen.generate(p7)
fec7 = fec_gen.encode(f7, scheme="convolutional")
int7 = int_gen.interleave(fec7, scheme="block")
m7 = mod_gen.modulate(
    int7,
    modulation_type="qpsk",
    sample_rate=192000.0,
    symbol_rate=9600.0,
    samples_per_symbol=20,
    pulse_shaping="rrc",
    rrc_alpha=0.35,
)
c7 = chan_gen.apply(m7, profile_name="combined_medium", overrides={"snr_db": 18.0, "cfo_hz": 1186.4})
cap7 = cap_gen.write(c7, profile_name="raw_f32_le_iq", output_dir=OUTPUT_DIR, filename="signal_07_qpsk_mission_data.iq")
print(f"  -> File: {cap7.file_path} ({cap7.sample_count_complex} samples, {cap7.duration_seconds:.2f}s)")
manifest.append({
    "signal_id": "signal_07",
    "filename": "signal_07_qpsk_mission_data.iq",
    "format": "Raw Binary IQ (f32_le)",
    "modulation": "QPSK",
    "symbol_rate": 9600.0,
    "sample_rate": 192000.0,
    "sps": 20,
    "snr_db": 18.0,
    "cfo_hz": 1186.4,
    "fec": "Convolutional K=7 R=1/2",
    "interleaver": "Block Matrix 16x16",
    "expected_payload": secret_msg,
    "description": "Full-chain QPSK with Block 16x16 interleaver, Conv K7 R1/2 FEC, and secret payload text."
})

# Save Manifest
manifest_path = OUTPUT_DIR / "MANIFEST.json"
with open(manifest_path, "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)

print("\n" + "=" * 80)
print(f"SUCCESS: Generated {len(manifest)} unseen test signals in '{OUTPUT_DIR.resolve()}'")
print(f"Manifest written to: '{manifest_path.resolve()}'")
print("=" * 80)
