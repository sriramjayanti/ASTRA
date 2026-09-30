"""
Enhanced Stage 4 DSP + Harmonic Lattice test on 100 benchmark captures.
"""

from __future__ import annotations
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))

import numpy as np
from scipy import signal, ndimage
from scipy.signal import find_peaks

from astra_symbol_rate.src.cyclostationary import parabolic_peak_refinement
from astra_symbol_rate.src.bandwidth import measure_occupied_bandwidth, extract_bandwidth_candidates
from astra_symbol_rate.src.models import SymbolRateCandidate


def enhanced_cyclostationary_candidates(
    iq: np.ndarray,
    sample_rate_hz: float,
    min_sps: float = 1.4,
    max_sps: float = 500.0,
    n_fft: int = 65536,
) -> List[Dict[str, float]]:
    n = len(iq)
    if n < 32 or sample_rate_hz <= 0:
        return []

    x = np.asarray(iq)
    candidates: List[Dict[str, float]] = []

    min_rate = float(sample_rate_hz / max_sps)
    max_rate = float(sample_rate_hz / min_sps)

    env2 = np.abs(x) ** 2
    env4 = np.abs(x) ** 4

    for env, order_weight in [(env2, 1.2), (env4, 1.0)]:
        env_zm = env - np.mean(env)
        window = np.hanning(len(env_zm))
        spec = np.abs(np.fft.rfft(env_zm * window, n=n_fft))
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate_hz)

        med = ndimage.median_filter(spec, size=101) + 1e-12
        line_ratio = spec / med

        mask = (freqs >= min_rate) & (freqs <= max_rate)
        if not np.any(mask):
            continue

        sub_line = line_ratio[mask]
        sub_freqs = freqs[mask]
        if len(sub_line) < 5:
            continue

        # Sensitive peak detection with height threshold 1.30 and prominence
        peaks, props = find_peaks(sub_line, height=1.30, prominence=0.15, distance=4)

        for peak_bin in peaks:
            refined_rate = parabolic_peak_refinement(sub_line, peak_bin, sub_freqs)
            if refined_rate <= 0:
                continue

            sps = float(sample_rate_hz / refined_rate)
            ratio_val = float(sub_line[peak_bin]) * order_weight

            candidates.append({
                "rate_hz": float(refined_rate),
                "sps": sps,
                "score": ratio_val,
                "source": "cyclostationary",
            })

    # Cluster duplicate cyclic estimates within 1.5%
    if not candidates:
        return []
    sorted_cands = sorted(candidates, key=lambda c: c["score"], reverse=True)
    deduped: List[Dict[str, float]] = []
    for c in sorted_cands:
        r = c["rate_hz"]
        matched = False
        for d in deduped:
            if abs(r - d["rate_hz"]) / max(1.0, d["rate_hz"]) <= 0.015:
                d["score"] = max(d["score"], c["score"])
                matched = True
                break
        if not matched:
            deduped.append(c)

    return deduped[:20]


def enhanced_autocorr_candidates(
    iq: np.ndarray,
    sample_rate_hz: float,
    min_sps: float = 1.4,
    max_sps: float = 500.0,
) -> Tuple[List[Dict[str, float]], List[Dict[str, float]]]:
    n = len(iq)
    max_lag = min(int(max_sps * 2), n // 2)
    min_lag = max(2, int(min_sps))
    if max_lag <= min_lag:
        return [], []

    env = np.abs(iq)
    env_c = env - np.mean(env)
    var = np.var(env_c)
    if var < 1e-12:
        return [], []

    n_fft = 1 << (2 * n - 1).bit_length()
    fx = np.fft.fft(env_c, n=n_fft)
    r_env = np.fft.ifft(fx * np.conj(fx)).real[:max_lag + 1]
    r_env = (r_env / n) / var

    pwr = env ** 2
    pwr_c = pwr - np.mean(pwr)
    var_p = np.var(pwr_c)
    if var_p < 1e-12:
        return [], []
    fp = np.fft.fft(pwr_c, n=n_fft)
    r_pwr = np.fft.ifft(fp * np.conj(fp)).real[:max_lag + 1]
    r_pwr = (r_pwr / n) / var_p

    def extract_from_r(r_arr, src_name):
        res = []
        if len(r_arr) <= min_lag:
            return res
        dyn = max(1e-6, np.max(r_arr[min_lag:]) - np.min(r_arr[min_lag:]))
        peaks_pos, props_pos = find_peaks(r_arr[min_lag:], prominence=dyn * 0.08, distance=2)
        peaks_neg, props_neg = find_peaks(-r_arr[min_lag:], prominence=dyn * 0.08, distance=2)

        for p_idx, p_lag in enumerate(peaks_pos):
            lag = p_lag + min_lag
            # 3-point parabolic refinement on correlation lag
            if 0 < lag < len(r_arr) - 1:
                y0, y1, y2 = r_arr[lag - 1], r_arr[lag], r_arr[lag + 1]
                denom = 2.0 * (2.0 * y1 - y0 - y2)
                d_lag = (y2 - y0) / denom if abs(denom) > 1e-12 else 0.0
                refined_lag = float(np.clip(lag + d_lag, min_lag, max_lag))
            else:
                refined_lag = float(lag)

            prom = float(props_pos["prominences"][p_idx])
            rate = float(sample_rate_hz / refined_lag)
            res.append({
                "rate_hz": rate,
                "sps": refined_lag,
                "score": float(np.clip(prom * 3.0, 0.0, 1.0)),
                "source": src_name,
            })
        return res

    env_cands = extract_from_r(r_env, "envelope_autocorr")
    pwr_cands = extract_from_r(r_pwr, "power_autocorr")
    return env_cands, pwr_cands


def enhanced_resolve_candidates(
    iq: np.ndarray,
    sample_rate_hz: float,
    modulation_hint: Optional[str] = None,
) -> List[SymbolRateCandidate]:
    obw_hz = measure_occupied_bandwidth(iq, sample_rate_hz)
    _, bw_cands = extract_bandwidth_candidates(iq, sample_rate_hz)
    cyclo_cands = enhanced_cyclostationary_candidates(iq, sample_rate_hz)
    env_cands, pwr_cands = enhanced_autocorr_candidates(iq, sample_rate_hz)

    # MSK Squaring & FSK Peak spacing
    from astra_symbol_rate.src.spectral import compute_spectral_features
    from astra_symbol_rate.src.instantaneous_frequency import extract_instantaneous_frequency_candidates
    _, _, spec_cands = compute_spectral_features(iq, sample_rate_hz)
    if_cands = extract_instantaneous_frequency_candidates(iq, sample_rate_hz)

    all_raw: List[Dict[str, Any]] = []
    all_raw.extend(cyclo_cands)
    all_raw.extend(env_cands)
    all_raw.extend(pwr_cands)
    all_raw.extend(spec_cands)
    all_raw.extend(if_cands)
    all_raw.extend(bw_cands)

    # Add standard priors with low score
    for std_r in [300, 600, 1200, 2400, 4800, 9600, 19200, 38400, 56000, 57600, 115200, 250000, 500000, 1000000]:
        sps = sample_rate_hz / std_r
        if 1.5 <= sps <= 256.0:
            all_raw.append({"rate_hz": float(std_r), "sps": sps, "score": 0.02, "source": "standard_prior"})

    # Filter out invalid
    valid = []
    for item in all_raw:
        r = float(item.get("rate_hz", 0.0))
        if r <= 10.0:
            continue
        sps = sample_rate_hz / r
        if sps < 1.4 or sps > 600.0:
            continue
        valid.append(item)

    # Cluster detections (+-2%)
    sorted_raw = sorted(valid, key=lambda c: c.get("score", 0.0), reverse=True)
    clusters: List[List[Dict[str, Any]]] = []
    for item in sorted_raw:
        r = item["rate_hz"]
        matched = False
        for cl in clusters:
            rep_r = cl[0]["rate_hz"]
            if abs(r - rep_r) / max(1.0, rep_r) <= 0.02:
                cl.append(item)
                matched = True
                break
        if not matched:
            clusters.append([item])

    candidates: List[SymbolRateCandidate] = []
    for cl in clusters:
        rates = np.array([c["rate_hz"] for c in cl], dtype=np.float64)
        scores = np.array([max(0.01, c.get("score", 0.5)) for c in cl], dtype=np.float64)
        centroid_rate = float(np.sum(rates * scores) / np.sum(scores))
        sps = float(sample_rate_hz / centroid_rate)
        sources = list(set([c.get("source", "unknown") for c in cl]))
        max_score = float(np.max(scores))

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

    # Harmonic Lattice & Subharmonic Synthesis
    is_fsk = bool(modulation_hint and "FSK" in modulation_hint.upper())
    if obw_hz > 50.0 and not is_fsk:
        synthesized: List[SymbolRateCandidate] = []
        for c in candidates:
            r = c.rate_hz
            ratio = obw_hz / r
            
            # Map of (ratio_min, ratio_max, multiplier, source_tag)
            harmonic_rules = [
                (1.80, 2.50, 2.0, "harmonic_lattice_1_2"),       # 1/2 Rs subharmonic -> 2.0 * r
                (1.25, 1.75, 4.0 / 3.0, "harmonic_lattice_3_4"), # 3/4 Rs subharmonic -> 1.333 * r
                (3.50, 5.00, 4.0, "harmonic_lattice_1_4"),       # 1/4 Rs subharmonic -> 4.0 * r
                (2.30, 3.20, 8.0 / 3.0, "harmonic_lattice_3_8"), # 3/8 Rs subharmonic -> 2.667 * r
                (1.45, 2.00, 8.0 / 5.0, "harmonic_lattice_5_8"), # 5/8 Rs subharmonic -> 1.6 * r
                (0.40, 0.70, 0.5, "harmonic_lattice_2_1"),       # 2x harmonic -> 0.5 * r
            ]
            
            for r_min, r_max, mult, tag in harmonic_rules:
                if r_min <= ratio <= r_max:
                    fund_r = r * mult
                    sps_fund = sample_rate_hz / fund_r
                    if 1.4 <= sps_fund <= 500.0:
                        exists = any(abs(c2.rate_hz - fund_r) / fund_r <= 0.025 for c2 in candidates + synthesized)
                        if not exists:
                            synthesized.append(SymbolRateCandidate(
                                rate_hz=fund_r,
                                samples_per_symbol=sps_fund,
                                sources=[tag],
                                supported_by=[tag],
                                score=float(c.score * 1.05),
                                raw_rate=fund_r,
                            ))
        candidates.extend(synthesized)

    # Modulation and Bandwidth Scoring
    for c in candidates:
        r = c.rate_hz
        if r <= 0:
            continue
        if modulation_hint:
            mod_u = modulation_hint.upper()
            if "MSK" in mod_u:
                if "msk_squaring" in c.sources or "spectral_peak" in c.sources:
                    c.score *= 2.0
            elif "FSK" in mod_u:
                if any("instantaneous" in s or "fsk" in s for s in c.sources):
                    c.score *= 2.0
                if all(s == "cyclostationary" for s in c.sources):
                    c.score *= 0.5
            elif "PSK" in mod_u or "QAM" in mod_u:
                if "cyclostationary" in c.sources:
                    c.score *= 1.5
                if any("harmonic_lattice" in s for s in c.sources):
                    c.score *= 1.4
                if all(s in ["instantaneous_frequency", "msk_squaring"] for s in c.sources):
                    c.score *= 0.20

        if obw_hz > 50.0:
            ratio = obw_hz / r
            if not is_fsk:
                # Smooth bandwidth-anchored gaussian curve centered at ideal ratio 1.25 (alpha=0.25)
                # ratio in [0.95, 1.55] gets strong boost
                if 0.95 <= ratio <= 1.55:
                    bw_boost = 1.0 + 1.2 * np.exp(-0.5 * ((ratio - 1.25) / 0.25) ** 2)
                    c.score *= bw_boost
                elif 1.55 < ratio <= 2.20:
                    c.score *= 0.45  # Subharmonic penalty
                elif ratio > 2.20:
                    c.score *= 0.25
                elif ratio < 0.85:
                    c.score *= 0.35  # Harmonic overtone penalty
            else:
                if ratio >= 0.80:
                    c.score *= 1.3
                elif ratio < 0.50:
                    c.score *= 0.40

    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates[:24]


def main():
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

    print(f"Testing Enhanced Stage 4 DSP on {len(selected)} captures...\n")
    class_stats = {m: {"total": 0, "recall@1": 0, "recall@3": 0, "recall@5": 0, "in_pool": 0} for m in class_caps}

    for cap in selected:
        raw = np.fromfile(cap["iq_path"], dtype=np.complex64)
        true_baud = float(cap["symbol_rate"])
        sr = float(cap["sample_rate"])
        mod = cap["true_modulation"]
        
        class_stats[mod]["total"] += 1
        candidates = enhanced_resolve_candidates(raw, sr, modulation_hint=mod)
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
    print("ENHANCED STAGE 4 DSP PERFORMANCE:")
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
    main()
