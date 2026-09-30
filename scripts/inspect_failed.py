import json
import sys
import os
sys.path.insert(0, os.path.abspath("."))
import numpy as np
from astra_symbol_rate.src.cyclostationary import extract_cyclostationary_evidence
from astra_symbol_rate.src.candidate_generator import SymbolRateCandidateGenerator

with open('datasets/ASTRA_FINAL_TEST_SET_V2_BLIND/blind_benchmark_manifest.json') as f:
    m = json.load(f)

high_caps = [c for c in m['captures'] if c['true_modulation'] not in ['UNKNOWN', '2-FSK', '4-FSK', 'MSK'] and c['snr_db'] >= 15]

failed = []
for cap in high_caps:
    fs = float(cap['sample_rate'])
    rs = float(cap['symbol_rate'])
    with open(cap['iq_path'], 'rb') as f:
        iq = np.fromfile(f, dtype=np.complex64)
    cyclo = extract_cyclostationary_evidence(iq, fs)
    gen = SymbolRateCandidateGenerator(merge_tolerance_percent=2.0)
    cands = gen.deduplicate_candidates(cyclo, fs)
    if not cands or abs(cands[0].rate_hz - rs)/rs > 0.03:
        failed.append({
            'mod': cap['true_modulation'],
            'rs': rs,
            'fs': fs,
            'cfo': cap['cfo_hz'],
            'sps': fs/rs,
            'top_rates': [c.rate_hz for c in cands[:4]] if cands else [],
        })

print(f'Total failed in High SNR: {len(failed)} / {len(high_caps)}')
print('First 15 failed cases:')
for item in failed[:15]:
    r0 = item['top_rates'][0] if item['top_rates'] else 0
    ratio = r0 / item['rs'] if item['rs'] > 0 else 0
    rounded_rates = [round(r, 1) for r in item['top_rates']]
    print(f"{item['mod']:8s} | True Rs: {item['rs']:8.1f} | Top1: {r0:8.1f} | Top1/Rs: {ratio:5.2f} | SPS: {item['sps']:5.1f} | All: {rounded_rates}")
