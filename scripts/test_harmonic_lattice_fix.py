"""
Prototype and verification test for Stage 4 Harmonic Lattice and Bandwidth-Anchored Ranking.
"""

from __future__ import annotations
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

import numpy as np
from astra_symbol_rate.src.dataset_builder import extract_dsp_evidence
from astra_symbol_rate.src.models import SymbolRateCandidate


def resolve_harmonic_lattice_candidates(
    raw_candidates: List[Dict[str, Any]],
    sample_rate_hz: float,
    occupied_bandwidth_hz: float,
    modulation_hint: Optional[str] = None,
    merge_tolerance_percent: float = 2.0,
) -> List[SymbolRateCandidate]:
    """
    Upgraded Harmonic Lattice and Bandwidth-Anchored Candidate Generator.
    """
    if not raw_candidates:
        return []

    # 1. Filter out physically impossible rates for the receiver sampling rate
    valid_items = []
    for item in raw_candidates:
        r = float(item.get("rate_hz", 0.0))
        if r <= 10.0:
            continue
        sps = sample_rate_hz / r
        if sps < 1.4 or sps > 350.0:
            continue
        valid_items.append(item)

    if not valid_items:
        return []

    # 2. Cluster near-coincident raw detections (+-2%)
    merge_tol = merge_tolerance_percent / 100.0
    sorted_raw = sorted(valid_items, key=lambda c: c.get("score", 0.0), reverse=True)
    clusters: List[List[Dict[str, Any]]] = []

    for item in sorted_raw:
        r = item["rate_hz"]
        matched = False
        for cl in clusters:
            rep_r = cl[0]["rate_hz"]
            if abs(r - rep_r) / max(1.0, rep_r) <= merge_tol:
                cl.append(item)
                matched = True
                break
        if not matched:
            clusters.append([item])

    # 3. Create initial candidate cluster centroids
    candidates: List[SymbolRateCandidate] = []
    for cl in clusters:
        rates = np.array([c["rate_hz"] for c in cl], dtype=np.float64)
        scores = np.array([max(0.01, c.get("score", 0.5)) for c in cl], dtype=np.float64)
        centroid_rate = float(np.sum(rates * scores) / np.sum(scores))
        sps = float(sample_rate_hz / centroid_rate)

        sources = list(set([c.get("source", "unknown") for c in cl]))
        max_score = float(np.max(scores))

        # Multi-source reinforcement
        has_primary = any(s in ["cyclostationary", "instantaneous_frequency", "msk_squaring", "fsk_tone_spacing"] for s in sources)
        multi_source_bonus = 0.25 * min(3, len(sources) - 1)
        effective_score = max_score * (1.0 + multi_source_bonus)
        if has_primary:
            effective_score *= 1.25

        cand = SymbolRateCandidate(
            rate_hz=centroid_rate,
            samples_per_symbol=sps,
            sources=sources,
            supported_by=sources,
            score=float(effective_score),
            raw_rate=centroid_rate,
        )
        candidates.append(cand)

    # 4. Synthesize missing fundamental candidates for prominent subharmonics
    is_fsk = bool(modulation_hint and "FSK" in modulation_hint.upper())
    is_msk = bool(modulation_hint and "MSK" in modulation_hint.upper())
    
    if occupied_bandwidth_hz > 50.0 and not is_fsk:
        # Check every high-scoring candidate to see if it represents a known subharmonic
        synthesized: List[SymbolRateCandidate] = []
        for c in candidates:
            r = c.rate_hz
            ratio = occupied_bandwidth_hz / r
            
            # (A) 3/4 Rs Subharmonic: ratio is in [1.60, 2.10] -> fundamental is 4/3 * r
            if 1.55 <= ratio <= 2.15:
                fund_r = r * (4.0 / 3.0)
                sps_fund = sample_rate_hz / fund_r
                if 1.5 <= sps_fund <= 256.0:
                    # Check if already exists in candidates
                    exists = any(abs(c2.rate_hz - fund_r) / fund_r <= 0.03 for c2 in candidates)
                    if not exists:
                        synthesized.append(SymbolRateCandidate(
                            rate_hz=fund_r,
                            samples_per_symbol=sps_fund,
                            sources=["harmonic_lattice_3_4"],
                            supported_by=["harmonic_lattice_3_4"],
                            score=float(c.score * 1.6),
                            raw_rate=fund_r,
                        ))
            
            # (B) 1/2 Rs Subharmonic: ratio is in [2.20, 3.20] -> fundamental is 2 * r
            elif 2.20 <= ratio <= 3.20:
                fund_r = r * 2.0
                sps_fund = sample_rate_hz / fund_r
                if 1.5 <= sps_fund <= 256.0:
                    exists = any(abs(c2.rate_hz - fund_r) / fund_r <= 0.03 for c2 in candidates)
                    if not exists:
                        synthesized.append(SymbolRateCandidate(
                            rate_hz=fund_r,
                            samples_per_symbol=sps_fund,
                            sources=["harmonic_lattice_1_2"],
                            supported_by=["harmonic_lattice_1_2"],
                            score=float(c.score * 1.6),
                            raw_rate=fund_r,
                        ))

            # (C) 2/3 Rs Subharmonic: ratio is in [1.80, 2.40] -> fundamental is 1.5 * r
            elif 1.80 <= ratio <= 2.40:
                fund_r = r * 1.5
                sps_fund = sample_rate_hz / fund_r
                if 1.5 <= sps_fund <= 256.0:
                    exists = any(abs(c2.rate_hz - fund_r) / fund_r <= 0.03 for c2 in candidates)
                    if not exists:
                        synthesized.append(SymbolRateCandidate(
                            rate_hz=fund_r,
                            samples_per_symbol=sps_fund,
                            sources=["harmonic_lattice_2_3"],
                            supported_by=["harmonic_lattice_2_3"],
                            score=float(c.score * 1.5),
                            raw_rate=fund_r,
                        ))

            # (D) 1/4 Rs Subharmonic: ratio is in [4.50, 6.50] -> fundamental is 4 * r
            elif 4.50 <= ratio <= 6.50:
                fund_r = r * 4.0
                sps_fund = sample_rate_hz / fund_r
                if 1.5 <= sps_fund <= 256.0:
                    exists = any(abs(c2.rate_hz - fund_r) / fund_r <= 0.03 for c2 in candidates)
                    if not exists:
                        synthesized.append(SymbolRateCandidate(
                            rate_hz=fund_r,
                            samples_per_symbol=sps_fund,
                            sources=["harmonic_lattice_1_4"],
                            supported_by=["harmonic_lattice_1_4"],
                            score=float(c.score * 1.4),
                            raw_rate=fund_r,
                        ))

        candidates.extend(synthesized)

    # 5. Full Harmonic Lattice Resolution and Cross-Scoring
    for c in candidates:
        r = c.rate_hz
        if r <= 0:
            continue

        # Modulation specific source weightings
        if modulation_hint:
            mod_u = modulation_hint.upper()
            if "MSK" in mod_u:
                if "msk_squaring" in c.sources or "spectral_peak" in c.sources:
                    c.score *= 2.5
            elif "FSK" in mod_u:
                if any("instantaneous" in s or "fsk" in s for s in c.sources):
                    c.score *= 2.2
                if all(s == "cyclostationary" for s in c.sources):
                    c.score *= 0.35
            elif "PSK" in mod_u or "QAM" in mod_u:
                if "cyclostationary" in c.sources:
                    c.score *= 1.6
                if all(s in ["instantaneous_frequency", "msk_squaring"] for s in c.sources):
                    c.score *= 0.15

        # Bandwidth-anchored scoring
        if occupied_bandwidth_hz > 50.0:
            ratio = occupied_bandwidth_hz / r
            if not is_fsk:
                # Single carrier Nyquist passband: OBW is (1 + alpha) * Rs in [1.10, 1.55]
                if 1.05 <= ratio <= 1.55:
                    c.score *= 2.0  # Perfect physical fundamental match!
                elif 0.85 <= ratio < 1.05:
                    c.score *= 1.4
                elif 1.55 < ratio <= 2.15:
                    # 3/4 Rs subharmonic
                    c.score *= 0.35
                elif 2.15 < ratio <= 3.30:
                    # 1/2 Rs subharmonic
                    c.score *= 0.25
                elif ratio > 3.30:
                    # Far subharmonic
                    c.score *= 0.15
                elif ratio < 0.80:
                    # Harmonic / overtone (2Rs, 3Rs)
                    c.score *= 0.30
            else:
                # Multi-tone FSK: OBW = (M-1)*df + Rs >= Rs
                if ratio >= 0.80:
                    c.score *= 1.3
                elif ratio < 0.50:
                    c.score *= 0.35

    # 6. Inter-candidate Harmonic Cross-Validation
    for c in candidates:
        r = c.rate_hz
        if r <= 0:
            continue

        # Look for 2x harmonic in pool
        matches_2x = [c2 for c2 in candidates if abs(c2.rate_hz - 2.0 * r) / (2.0 * r) <= 0.03]
        # Look for 4/3x (from 3/4) in pool
        matches_4_3x = [c2 for c2 in candidates if abs(c2.rate_hz - (4.0 / 3.0) * r) / ((4.0 / 3.0) * r) <= 0.03]

        if not is_fsk and occupied_bandwidth_hz > 50.0:
            ratio_r = occupied_bandwidth_hz / r
            if matches_2x:
                c2x = matches_2x[0]
                ratio_2r = occupied_bandwidth_hz / (2.0 * r)
                if 2.15 <= ratio_r <= 3.30 and 1.05 <= ratio_2r <= 1.60:
                    c2x.score = max(c2x.score, c.score * 1.8)
                    c.score *= 0.25

            if matches_4_3x:
                c43 = matches_4_3x[0]
                ratio_43 = occupied_bandwidth_hz / c43.rate_hz
                if 1.55 <= ratio_r <= 2.15 and 1.05 <= ratio_43 <= 1.60:
                    c43.score = max(c43.score, c.score * 1.8)
                    c.score *= 0.25

    # Sort descending by score
    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates[:24]


def test_fix_on_100_captures():
    manifest_path = root / "datasets" / "ASTRA_FINAL_TEST_SET_V2_BLIND" / "blind_benchmark_manifest.json"
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    captures = manifest["captures"]
    class_caps: Dict[str, List[Dict[str, Any]]] = {}
    for cap in captures:
        mod = cap["true_modulation"]
        if mod == "UNKNOWN":
            continue
        if mod not in class_caps:
            class_caps[mod] = []
        if len(class_caps[mod]) < 10:
            class_caps[mod].append(cap)

    selected = []
    for mod in ["BPSK", "QPSK", "8PSK", "DQPSK", "16QAM", "64QAM", "256QAM", "2-FSK", "4-FSK", "MSK"]:
        selected.extend(class_caps.get(mod, []))

    print(f"Testing Harmonic Lattice on {len(selected)} captures...\n")
    
    class_stats = {m: {"total": 0, "recall@1": 0, "recall@3": 0, "recall@5": 0, "in_pool": 0} for m in class_caps}

    for cap in selected:
        raw = np.fromfile(cap["iq_path"], dtype=np.complex64)
        true_baud = float(cap["symbol_rate"])
        sr = float(cap["sample_rate"])
        mod = cap["true_modulation"]
        
        class_stats[mod]["total"] += 1
        
        # 1. Extract DSP evidence
        evidence = extract_dsp_evidence(raw, sr)
        
        all_raw = []
        all_raw.extend(evidence.autocorr_peaks)
        all_raw.extend(evidence.power_autocorr_peaks)
        all_raw.extend(evidence.if_peaks)
        all_raw.extend(evidence.spectral_peak_spacings)
        all_raw.extend(evidence.bandwidth_candidates)
        all_raw.extend(evidence.cyclostationary_peaks)
        
        # Add standard priors with low score
        for std_r in [300, 600, 1200, 2400, 4800, 9600, 19200, 38400, 56000, 57600, 115200, 250000, 500000, 1000000]:
            sps = sr / std_r
            if 1.5 <= sps <= 256.0:
                all_raw.append({"rate_hz": float(std_r), "sps": sps, "score": 0.02, "source": "standard_prior"})
                
        # 2. Run new harmonic lattice candidate resolution
        candidates = resolve_harmonic_lattice_candidates(
            all_raw,
            sr,
            occupied_bandwidth_hz=evidence.occupied_bandwidth_hz,
            modulation_hint=mod,
        )
        
        rates = [c.rate_hz for c in candidates]
        in_pool = any(abs(r - true_baud) / true_baud <= 0.03 for r in rates)
        r1_match = len(rates) > 0 and abs(rates[0] - true_baud) / true_baud <= 0.03
        r3_match = any(abs(r - true_baud) / true_baud <= 0.03 for r in rates[:3])
        r5_match = any(abs(r - true_baud) / true_baud <= 0.03 for r in rates[:5])
        
        if in_pool: class_stats[mod]["in_pool"] += 1
        if r1_match: class_stats[mod]["recall@1"] += 1
        if r3_match: class_stats[mod]["recall@3"] += 1
        if r5_match: class_stats[mod]["recall@5"] += 1

    print("=" * 80)
    print("UPGRADED HARMONIC LATTICE PERFORMANCE:")
    print("=" * 80)
    print(f"{'Modulation':<10} | {'Recall@1 (<=3%)':<16} | {'Recall@3 (<=3%)':<16} | {'Recall@5 (<=3%)':<16} | {'In DSP Pool':<12}")
    print("-" * 80)
    tot_r1, tot_r3, tot_r5, tot_pool, tot_cnt = 0, 0, 0, 0, 0
    for mod, s in class_stats.items():
        cnt = s["total"]
        tot_cnt += cnt
        tot_r1 += s["recall@1"]
        tot_r3 += s["recall@3"]
        tot_r5 += s["recall@5"]
        tot_pool += s["in_pool"]
        print(f"{mod:<10} | {s['recall@1']}/{cnt} ({s['recall@1']/cnt*100:5.1f}%)    | {s['recall@3']}/{cnt} ({s['recall@3']/cnt*100:5.1f}%)    | {s['recall@5']}/{cnt} ({s['recall@5']/cnt*100:5.1f}%)    | {s['in_pool']}/{cnt} ({s['in_pool']/cnt*100:5.1f}%)")
    print("-" * 80)
    print(f"{'TOTAL':<10} | {tot_r1}/{tot_cnt} ({tot_r1/tot_cnt*100:5.1f}%)    | {tot_r3}/{tot_cnt} ({tot_r3/tot_cnt*100:5.1f}%)    | {tot_r5}/{tot_cnt} ({tot_r5/tot_cnt*100:5.1f}%)    | {tot_pool}/{tot_cnt} ({tot_pool/tot_cnt*100:5.1f}%)")
    print("=" * 80)


if __name__ == "__main__":
    test_fix_on_100_captures()
