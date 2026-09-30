import sys
import time
from pathlib import Path
import numpy as np

WORKSPACE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE))

from astra_synthetic.payload.models import PayloadRecord, text_to_bits, calculate_entropy
from astra_synthetic.framing.generator import FrameGenerator
from astra_synthetic.fec.generator import FECGenerator
from astra_synthetic.interleaving.generator import InterleaverGenerator
from astra_synthetic.modulation.generator import ModulationGenerator
from astra_synthetic.channel.generator import ChannelGenerator

from astra_config.classes import normalize_modulation_name
from astra_fusion.src.inference import ASTRAFusionEngine
from astra_symbol_rate.src.inference import SymbolRateEstimator
from astra_candidate_engine.src.inference import CandidateHypothesisEngine
from astra_synchronization.src.inference import SynchronizationEngine
from astra_demodulation.src.inference import DemodulationEngine
from astra_interleaver.src.inference import InterleaverTestingEngine
from astra_fec.src.inference import FECTestingEngine
from astra_validation.src.inference import ValidationEngine

# Initialize engines
p_text = "hi hello"
p_bits = text_to_bits(p_text)
p_rec = PayloadRecord(
    payload_id="test_p0",
    payload_type="text",
    bit_length=len(p_bits),
    byte_length=len(p_bits)//8,
    payload_bits=p_bits,
    payload_bytes=p_text.encode('utf-8'),
    source_text=p_text,
    entropy_estimate=calculate_entropy(p_bits)
)

f_gen = FrameGenerator()
f_rec = f_gen.generate(p_rec, num_frames=3)
print(f"Generated frame with {len(f_rec.frame_bits)} bits, sync word: {f_rec.sync_value}")

fec_gen = FECGenerator()
fec_rec = fec_gen.encode(f_rec, scheme="none")

int_gen = InterleaverGenerator()
int_rec = int_gen.interleave(fec_rec, scheme="none")

mod_gen = ModulationGenerator()
mod_rec = mod_gen.modulate(int_rec, modulation_type="qpsk", sample_rate=192000.0, symbol_rate=9600.0, samples_per_symbol=20)

chan_gen = ChannelGenerator()
chan_rec = chan_gen.apply(mod_rec, overrides={"snr_db": 25.0, "cfo_hz": 150.0})

raw_iq = chan_rec.impaired_iq
print(f"Synthesized clean IQ {len(mod_rec.clean_iq)} -> impaired IQ {len(raw_iq)}")

# Now run blind ASTRA pipeline:
fusion_engine = ASTRAFusionEngine(top_k=5, device="cpu")
sr_estimator = SymbolRateEstimator()
cand_engine = CandidateHypothesisEngine(config={
    "candidate_engine": {
        "modulation_top_k": 5,
        "symbol_rate_top_k": 3,
        "max_candidates": 15,
        "beam_width": 15
    }
})
sync_engine = SynchronizationEngine()
demod_engine = DemodulationEngine()
int_engine = InterleaverTestingEngine()
fec_engine = FECTestingEngine()
val_engine = ValidationEngine()

t0 = time.perf_counter()
f_pred = fusion_engine.predict(raw_iq[:2048])
top_mods = [
    normalize_modulation_name(c.get('class', c.get('class_name', str(c))))
    if isinstance(c, dict)
    else normalize_modulation_name(getattr(c, "class_name", str(c)))
    for c in f_pred.top_k[:5]
]
print(f"Modulation Top-5: {top_mods}")

sr_res = sr_estimator.estimate(raw_iq[:4096], sample_rate_hz=192000.0)
top_bauds = [float(getattr(c, "symbol_rate_hz", getattr(c, "rate_hz", 0.0)) if not isinstance(c, dict) else c.get("symbol_rate_hz", c.get("rate_hz", 0.0))) for c in sr_res.top_k[:3]]
print(f"Baud Top-3: {top_bauds}")

mod_dict = {"top_k": [{"class": m, "probability": 0.5} for m in top_mods]}
sr_dict = {"top_k": [{"symbol_rate_hz": b, "score": 0.5} for b in top_bauds]}
cand_set = cand_engine.generate(mod_dict, sr_dict, sample_rate_hz=192000.0)

for cand in cand_set.candidates:
    s_res = sync_engine.synchronize(raw_iq, cand)
    if not s_res.success:
        continue
    lock_sc = s_res.lock_metrics.get("composite_lock_score", 0.0)
    print(f"Sync locked for {cand.modulation} @ {cand.symbol_rate_hz}: lock_score={lock_sc:.3f}")
    
    d_res = demod_engine.demodulate(s_res)
    if not d_res.success:
        continue
    print(f"Demod success for {cand.modulation}: status={d_res.status}, hard_bits={len(d_res.hard_bits)}")
    
    # Check bit error rate against true frame bits
    variants = [d_res] + (d_res.phase_variants or [])
    for v_idx, var in enumerate(variants):
        rx_bits = var.hard_bits
        # Search for sync word in rx_bits
        sync_bits = f_rec.sync_bits
        # Find sync correlation
        corr = np.correlate(1.0 - 2.0*rx_bits[:len(sync_bits)*10], 1.0 - 2.0*sync_bits, mode='valid')
        max_corr = np.max(corr)
        peak_idx = int(np.argmax(corr))
        print(f"  Variant {v_idx}: max sync corr = {max_corr:.1f}/{len(sync_bits)} at pos {peak_idx}")
        if max_corr >= len(sync_bits) - 1:
            print("  -> Sync word FOUND exactly!")
            # Extract frame payload
            frame_len = len(f_rec.frame_bits)
            extracted_frame = rx_bits[peak_idx : peak_idx + frame_len]
            print(f"  -> Extracted frame len: {len(extracted_frame)}")
            # Validate with ValidationEngine
            fec_cand = fec_engine.decode_candidate(
                hypothesis=fec_engine.generate_candidates(len(rx_bits))[0],
                hard_bits=rx_bits
            )
            val_res = val_engine.validate(fec_cand, context={"candidate_frame_lengths": [frame_len]})
            print(f"  -> Validation result: status={val_res.validation_status}, score={val_res.overall_validation_score:.3f}")
            # Payload is at payload_start
            payload_start = peak_idx + len(sync_bits) + len(f_rec.header_bits)
            payload_len = len(p_bits)
            extracted_p_bits = rx_bits[payload_start : payload_start + payload_len]
            extracted_bytes = np.packbits(extracted_p_bits).tobytes()
            print(f"  -> Extracted Payload Hex: {extracted_bytes.hex()}")
            print(f"  -> Extracted Payload Text: {extracted_bytes}")
            break
