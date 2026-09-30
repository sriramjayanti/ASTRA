import json
import traceback
import numpy as np
import sys

sys.path.insert(0, ".")
from astra_synchronization.src.inference import SynchronizationEngine
from astra_demodulation.src.inference import DemodulationEngine

sync_engine = SynchronizationEngine()
demod_engine = DemodulationEngine()

manifest = "datasets/ASTRA_FINAL_TEST_SET_V2_BLIND/blind_benchmark_manifest.json"
with open(manifest) as f:
    d = json.load(f)

# Test diverse modulations: 2-FSK, 4-FSK, BPSK, QPSK, 8PSK, 16QAM, 64QAM, 256QAM
tested_mods = set()
for cap in d["captures"]:
    true_mod = cap["true_modulation"]
    if true_mod == "UNKNOWN" or true_mod in tested_mods:
        continue
    tested_mods.add(true_mod)
    
    iq = np.fromfile(cap["iq_path"], dtype=np.complex64)
    true_baud = float(cap["symbol_rate"])
    sr = float(cap["sample_rate"])
    true_cfo = float(cap.get("cfo_hz", 0.0))
    snr = float(cap.get("snr_db", 0.0))
    print(f"\n=======================================================")
    print(f"Testing {cap['source_id']} | {true_mod} | Baud={true_baud:.1f} | Fs={sr:.1f} | SPS={sr/true_baud:.2f} | True CFO={true_cfo:.1f} Hz | SNR={snr:.1f} dB")
    print(f"=======================================================")
    
    hyp = {"modulation": true_mod, "symbol_rate_hz": true_baud, "sample_rate_hz": sr}
    try:
        sync_res = sync_engine.synchronize(iq, hypothesis=hyp)
        print("  Sync Status:", sync_res.status, "| Success:", sync_res.success)
        print(f"  Estimated CFO: {sync_res.estimated_cfo_hz:.1f} Hz (Error: {abs(sync_res.estimated_cfo_hz - true_cfo):.1f} Hz)")
        print("  Lock Metrics:", sync_res.lock_metrics)
        syms = sync_res.symbol_samples
        print("  Symbol Samples Count:", len(syms) if syms is not None else 0)
        
        sync_dict = {
            "modulation": true_mod,
            "symbol_samples": syms,
            "symbol_rate_hz": true_baud,
            "input_sample_rate_hz": sr,
        }
        demod_res = demod_engine.demodulate(sync_dict)
        print("  Demod Bits Count:", len(demod_res.hard_bits), "| EVM:", f"{demod_res.evm_percent:.2f}%" if demod_res.evm_percent else "N/A")
        print("  Demod Constellation Mean Radius:", np.mean(np.abs(syms)) if len(syms) > 0 else "N/A")
    except Exception as e:
        print("  EXCEPTION:", e)
        traceback.print_exc()
