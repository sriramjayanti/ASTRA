import numpy as np
import json
import sys
sys.path.insert(0, ".")
from astra_synchronization.src.cfo import estimate_coarse_cfo_mth_power, estimate_coarse_cfo_qam

manifest = "datasets/ASTRA_FINAL_TEST_SET_V2_BLIND/blind_benchmark_manifest.json"
with open(manifest) as f:
    d = json.load(f)

for cap in d["captures"]:
    if cap["true_modulation"] in ["BPSK", "16QAM"]:
        iq = np.fromfile(cap["iq_path"], dtype=np.complex64)
        sr = float(cap["sample_rate"])
        rs = float(cap["symbol_rate"])
        true_cfo = float(cap["cfo_hz"])
        mod = cap["true_modulation"]
        
        if mod == "BPSK":
            est, ratio = estimate_coarse_cfo_mth_power(iq, sr, m_order=2)
            print(f"BPSK: True={true_cfo:.1f}, Est={est:.1f}, Ratio={ratio:.2f}")
        else:
            est, ratio = estimate_coarse_cfo_qam(iq, sr, rs)
            print(f"16QAM: True={true_cfo:.1f}, Est={est:.1f}, Ratio={ratio:.2f}")
