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
from astra_interleaver.src.inference import InterleaverTestingEngine
from astra_fec.src.inference import FECTestingEngine
from astra_validation.src.inference import ValidationEngine

# Generate sample IQ
fs = 192000.0
baud = 9600.0
sps = int(fs / baud)
bits = np.random.randint(0, 2, 400)
b_i = bits[0::2]
b_q = bits[1::2]
syms = ((2.0 * b_i - 1.0) + 1j * (2.0 * b_q - 1.0)).astype(np.complex64) / np.sqrt(2.0)
iq = np.repeat(syms, sps)

print("Starting benchmark of each stage...")

t0 = time.perf_counter()
fusion = ASTRAFusionEngine(device="cpu")
print(f"Fusion init: {time.perf_counter() - t0:.3f} s")

t0 = time.perf_counter()
pred = fusion.predict(iq[:2048])
print(f"Stage 3 fusion predict: {time.perf_counter() - t0:.3f} s, top: {[getattr(c, 'class_name', str(c)) for c in pred.top_k[:3]]}")

t0 = time.perf_counter()
sr_est = SymbolRateEstimator()
sr_res = sr_est.estimate(iq[:4096], sample_rate_hz=fs)
print(f"Stage 4 baud estimate: {time.perf_counter() - t0:.3f} s, best baud: {sr_res.best_symbol_rate_hz}")

t0 = time.perf_counter()
cand_engine = CandidateHypothesisEngine()
mod_dict = {"top_k": [{"class": "QPSK", "probability": 0.9}]}
sr_dict = {"top_k": [{"symbol_rate_hz": 9600.0, "score": 0.9}]}
cands = cand_engine.generate(mod_dict, sr_dict, sample_rate_hz=fs)
print(f"Stage 5 candidate gen: {time.perf_counter() - t0:.3f} s, cands count: {len(cands.candidates)}")

t0 = time.perf_counter()
sync = SynchronizationEngine()
s_res = sync.synchronize(iq[:8192], cands.candidates[0])
print(f"Stage 6 sync: {time.perf_counter() - t0:.3f} s, lock: {s_res.lock_metrics.get('composite_lock_score', 0.0):.3f}")

t0 = time.perf_counter()
demod = DemodulationEngine()
d_res = demod.demodulate(s_res)
print(f"Stage 7 demod: {time.perf_counter() - t0:.3f} s, hard bits count: {len(d_res.hard_bits) if d_res.hard_bits is not None else 0}")

t0 = time.perf_counter()
int_engine = InterleaverTestingEngine()
int_res = int_engine.test_candidates(d_res)
print(f"Stage 8 interleaver: {time.perf_counter() - t0:.3f} s, surviving: {len(int_res.surviving_candidates)}")

t0 = time.perf_counter()
fec_engine = FECTestingEngine()
fec_res = fec_engine.test_candidates(int_res.top_candidate)
print(f"Stage 9 FEC: {time.perf_counter() - t0:.3f} s, surviving: {len(fec_res.surviving_candidates)}")

t0 = time.perf_counter()
val_engine = ValidationEngine()
val_res = val_engine.validate(fec_res.top_candidate)
print(f"Stage 10 validation: {time.perf_counter() - t0:.3f} s, score: {val_res.overall_validation_score:.3f}")
