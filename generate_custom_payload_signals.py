"""
Generates custom test signal captures with specific user payloads:
- sriram
- subhani
- varun
- vivek
- chaitanys
- xxxx
- vvvv
- sakfjlasds;lkjgaso;tiuewoasjdlkads;lfkjdffj'af
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

p_gen = PayloadGenerator()
f_gen = FrameGenerator()
fec_gen = FECGenerator()
int_gen = InterleaverGenerator()
mod_gen = ModulationGenerator()
chan_gen = ChannelGenerator()
cap_gen = CaptureGenerator()

CUSTOM_SIGNALS = [
    {
        "id": "signal_08_sriram",
        "payload_text": "sriram",
        "filename": "signal_08_sriram_qpsk.sigmf-data",
        "profile": "sigmf_cf32_le",
        "format": "SigMF (cf32_le)",
        "mod": "qpsk",
        "symbol_rate": 9600.0,
        "sample_rate": 192000.0,
        "sps": 20,
        "snr_db": 22.0,
        "cfo_hz": 350.0,
        "desc": "QPSK signal with payload 'sriram' in standard SigMF format."
    },
    {
        "id": "signal_09_subhani",
        "payload_text": "subhani",
        "filename": "signal_09_subhani_8psk.wav",
        "profile": "wav_i16_iq",
        "format": "Stereo WAV (int16 IQ)",
        "mod": "8psk",
        "symbol_rate": 19200.0,
        "sample_rate": 192000.0,
        "sps": 10,
        "snr_db": 20.0,
        "cfo_hz": -420.0,
        "desc": "8PSK audio-encapsulated IQ capture with payload 'subhani'."
    },
    {
        "id": "signal_10_varun",
        "payload_text": "varun",
        "filename": "signal_10_varun_16qam.iq",
        "profile": "raw_f32_le_iq",
        "format": "Raw Binary IQ (f32_le)",
        "mod": "16qam",
        "symbol_rate": 24000.0,
        "sample_rate": 192000.0,
        "sps": 8,
        "snr_db": 25.0,
        "cfo_hz": 180.0,
        "desc": "16-QAM high-speed carrier carrying payload 'varun'."
    },
    {
        "id": "signal_11_vivek",
        "payload_text": "vivek",
        "filename": "signal_11_vivek_bpsk.sigmf-data",
        "profile": "sigmf_cf32_le",
        "format": "SigMF (cf32_le)",
        "mod": "bpsk",
        "symbol_rate": 4800.0,
        "sample_rate": 192000.0,
        "sps": 40,
        "snr_db": 18.0,
        "cfo_hz": 520.0,
        "desc": "Robust BPSK carrier transmitting payload 'vivek'."
    },
    {
        "id": "signal_12_chaitanys",
        "payload_text": "chaitanys",
        "filename": "signal_12_chaitanys_2fsk.wav",
        "profile": "wav_i16_iq",
        "format": "Stereo WAV (int16 IQ)",
        "mod": "2fsk",
        "symbol_rate": 4800.0,
        "sample_rate": 96000.0,
        "sps": 20,
        "snr_db": 19.0,
        "cfo_hz": -250.0,
        "desc": "2-FSK frequency-shift keyed telemetry carrying 'chaitanys'."
    },
    {
        "id": "signal_13_xxxx",
        "payload_text": "xxxx",
        "filename": "signal_13_xxxx_qpsk.iq",
        "profile": "raw_f32_le_iq",
        "format": "Raw Binary IQ (f32_le)",
        "mod": "qpsk",
        "symbol_rate": 9600.0,
        "sample_rate": 192000.0,
        "sps": 20,
        "snr_db": 21.0,
        "cfo_hz": 600.0,
        "desc": "QPSK signal with pattern payload 'xxxx'."
    },
    {
        "id": "signal_14_vvvv",
        "payload_text": "vvvv",
        "filename": "signal_14_vvvv_8psk.iq",
        "profile": "raw_f32_le_iq",
        "format": "Raw Binary IQ (f32_le)",
        "mod": "8psk",
        "symbol_rate": 9600.0,
        "sample_rate": 192000.0,
        "sps": 20,
        "snr_db": 20.0,
        "cfo_hz": -300.0,
        "desc": "8PSK phase-keyed burst carrying payload 'vvvv'."
    },
    {
        "id": "signal_15_sakfjlasds",
        "payload_text": "sakfjlasds;lkjgaso;tiuewoasjdlkads;lfkjdffj'af",
        "filename": "signal_15_sakfjlasds_burst.sigmf-data",
        "profile": "sigmf_cf32_le",
        "format": "SigMF (cf32_le)",
        "mod": "qpsk",
        "symbol_rate": 19200.0,
        "sample_rate": 192000.0,
        "sps": 10,
        "snr_db": 24.0,
        "cfo_hz": 850.0,
        "desc": "Wideband QPSK burst containing complex payload 'sakfjlasds;lkjgaso;tiuewoasjdlkads;lfkjdffj\\'af'."
    }
]

# Read existing manifest
manifest_path = OUTPUT_DIR / "MANIFEST.json"
manifest = []
if manifest_path.exists():
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except Exception:
        manifest = []

existing_ids = {m["signal_id"] for m in manifest}

print("=" * 80)
print("     ASTRA: GENERATING USER-REQUESTED PAYLOAD SIGNALS")
print("=" * 80)

for idx, sig in enumerate(CUSTOM_SIGNALS, 1):
    print(f"\n[{idx}/{len(CUSTOM_SIGNALS)}] Generating {sig['id']}: payload='{sig['payload_text'][:20]}...'")
    msg_bytes = sig["payload_text"].encode("utf-8")
    # Pad to at least 16 bytes for framing consistency
    if len(msg_bytes) < 16:
        msg_bytes = msg_bytes + b" " * (16 - len(msg_bytes))

    msg_bits = np.unpackbits(np.frombuffer(msg_bytes, dtype=np.uint8))
    # Repeat to fill at least 512 bits
    target_len = max(512, len(msg_bits) * 4)
    repeated_bits = np.tile(msg_bits, (target_len // len(msg_bits)) + 1)[:target_len]

    p = p_gen.generate(payload_type="random_bits", bit_length=len(repeated_bits), seed=200 + idx)
    p.bits = repeated_bits
    p.bit_length = len(repeated_bits)

    f = f_gen.generate(p)
    fec = fec_gen.encode(f, scheme="none")
    intl = int_gen.interleave(fec, scheme="none")

    pulse = "none" if sig["mod"] == "2fsk" else "rrc"
    m = mod_gen.modulate(
        intl,
        modulation_type=sig["mod"],
        sample_rate=sig["sample_rate"],
        symbol_rate=sig["symbol_rate"],
        samples_per_symbol=sig["sps"],
        pulse_shaping=pulse,
        rrc_alpha=0.35 if pulse == "rrc" else 0.0,
    )

    c = chan_gen.apply(m, profile_name="combined_medium", overrides={"snr_db": sig["snr_db"], "cfo_hz": sig["cfo_hz"]})
    cap = cap_gen.write(c, profile_name=sig["profile"], output_dir=OUTPUT_DIR, filename=sig["filename"])
    print(f"  -> Generated: {cap.file_path} ({cap.sample_count_complex} samples, {cap.duration_seconds:.2f}s)")

    entry = {
        "signal_id": sig["id"],
        "filename": sig["filename"],
        "format": sig["format"],
        "modulation": sig["mod"].upper(),
        "symbol_rate": sig["symbol_rate"],
        "sample_rate": sig["sample_rate"],
        "sps": sig["sps"],
        "snr_db": sig["snr_db"],
        "cfo_hz": sig["cfo_hz"],
        "fec": "none",
        "interleaver": "none",
        "expected_payload": sig["payload_text"],
        "description": sig["desc"]
    }

    if sig["id"] not in existing_ids:
        manifest.append(entry)
        existing_ids.add(sig["id"])
    else:
        # Update existing
        for i, m_item in enumerate(manifest):
            if m_item["signal_id"] == sig["id"]:
                manifest[i] = entry
                break

with open(manifest_path, "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)

print("\n" + "=" * 80)
print(f"SUCCESS: Generated {len(CUSTOM_SIGNALS)} custom payload signals in '{OUTPUT_DIR.resolve()}'")
print(f"Updated Manifest: '{manifest_path.resolve()}'")
print("=" * 80)
