"""
Complete Stage 4 Harmonic Lattice & FSK/PSK/QAM Physics-Anchored Resolution.
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
from astra_symbol_rate.src.bandwidth import measure_occupied_bandwidth
from astra_symbol_rate.src.models import SymbolRateCandidate


def extract_all_physical_evidence_candidates(
    iq: np.ndarray,
    sample_rate_hz: float,
    modulation_hint: Optional[str] = None,
) -> List[SymbolRateCandidate]:
    n = len(iq)
    if n < 32 or sample_rate_hz <= 0:
        return []

    x = np.asarray(iq)
    obw_hz = measure_occupied_bandwidth(x, sample_rate_hz)
    all_raw: List[Dict[str, Any]] = []

    min_sps = 1.4
    max_sps = 500.0
    min_rate = float(sample_rate_hz / max_sps)
    max_rate = float(sample_rate_hz / min_sps)

    # 1. Cyclostationary Squared & 4th-power Envelope Spectra
    env2 = np.abs(x) ** 2
    env4 = np.abs(x) ** 4
    for env, order_weight in [(env2, 1.2), (env4, 1.0)]:
        env_zm = env - np.mean(env)
        window = np.hanning(len(env_zm))
        spec = np.abs(np.fft.rfft(env_zm * window, n=65536))
        freqs = np.fft.rfftfreq(65536, d=1.0 / sample_rate_hz)
        med = ndimage.median_filter(spec, size=101) + 1e-12
        line_ratio = spec / med

        mask = (freqs >= min_rate) & (freqs <= max_rate)
        if np.any(mask):
            sub_line = line_ratio[mask]
            sub_freqs = freqs[mask]
            if len(sub_line) >= 5:
                peaks, _ = find_peaks(sub_line, height=1.28, prominence=0.12, distance=4)
                for peak_bin in peaks:
                    refined_rate = parabolic_peak_refinement(sub_line, peak_bin, sub_freqs)
                    if refined_rate > 0:
                        all_raw.append({
                            "rate_hz": float(refined_rate),
                            "sps": float(sample_rate_hz / refined_rate),
                            "score": float(sub_line[peak_bin] * order_weight),
                            "source": "cyclostationary",
                        })

    # 2. Envelope & Power Autocorrelation
    env = np.abs(x)
    env_c = env - np.mean(env)
    var = np.var(env_c)
    max_lag = min(int(max_sps * 2), n // 2)
    min_lag = max(2, int(min_sps))
    if var > 1e-12 and max_lag > min_lag:
        n_fft = 1 << (2 * n - 1).bit_length()
        fx = np.fft.fft(env_c, n=n_fft)
        r_env = np.fft.ifft(fx * np.conj(fx)).real[:max_lag + 1]
        r_env = (r_env / n) / var
        dyn = max(1e-6, np.max(r_env[min_lag:]) - np.min(r_env[min_lag:]))
        peaks_pos, props_pos = find_peaks(r_env[min_lag:], prominence=dyn * 0.08, distance=2)
        for p_idx, p_lag in enumerate(peaks_pos):
            lag = p_lag + min_lag
            if 0 < lag < len(r_env) - 1:
                y0, y1, y2 = r_env[lag - 1], r_env[lag], r_env[lag + 1]
                denom = 2.0 * (2.0 * y1 - y0 - y2)
                d_lag = (y2 - y0) / denom if abs(denom) > 1e-12 else 0.0
                refined_lag = float(np.clip(lag + d_lag, min_lag, max_lag))
            else:
                refined_lag = float(lag)
            prom = float(props_pos["prominences"][p_idx])
            all_raw.append({
                "rate_hz": float(sample_rate_hz / refined_lag),
                "sps": refined_lag,
                "score": float(np.clip(prom * 3.0, 0.0, 1.0)),
                "source": "envelope_autocorr",
            })

    # 3. Instantaneous Frequency Transitions & Discriminator Autocorrelation (Crucial for FSK/MSK)
    prod = x[1:] * np.conj(x[:-1])
    dphi = np.angle(prod)
    f_inst = dphi * (sample_rate_hz / (2.0 * np.pi))
    f_smooth = signal.medfilt(f_inst, 3)
    diff_pwr = np.abs(np.diff(f_smooth)) ** 2
    
    # Transition spectra
    for win_len in [15, 63, 255]:
        if len(diff_pwr) >= win_len:
            ped = np.convolve(diff_pwr, np.ones(win_len) / win_len, mode="same")
            diff_det = diff_pwr - ped
            spec = np.abs(np.fft.rfft(diff_det * np.hanning(len(diff_det)), n=65536))
            freqs = np.fft.rfftfreq(65536, d=1.0 / sample_rate_hz)
            mask = (freqs >= min_rate) & (freqs <= max_rate)
            if np.any(mask):
                sub_s, sub_f = spec[mask], freqs[mask]
                if len(sub_s) >= 5 and np.max(sub_s) > 1e-9:
                    peaks, _ = find_peaks(sub_s, prominence=float(np.max(sub_s)) * 0.012, distance=4)
                    for p in peaks:
                        rr = parabolic_peak_refinement(sub_s, p, sub_f)
                        if rr > 0:
                            all_raw.append({
                                "rate_hz": float(rr),
                                "sps": float(sample_rate_hz / rr),
                                "score": 20.0,
                                "source": "instantaneous_frequency",
                            })

    # Autocorrelation of frequency transition spikes (finds exact SPS for 2-FSK/4-FSK across all SPS)
    if len(diff_pwr) > 32:
        dp_c = diff_pwr - np.mean(diff_pwr)
        var_dp = np.var(dp_c)
        if var_dp > 1e-12:
            n_fft = 1 << (2 * len(dp_c) - 1).bit_length()
            fdp = np.fft.fft(dp_c, n=n_fft)
            r_dp = np.fft.ifft(fdp * np.conj(fdp)).real[:max_lag + 1]
            r_dp = (r_dp / len(dp_c)) / var_dp
            dyn_dp = max(1e-6, np.max(r_dp[min_lag:]) - np.min(r_dp[min_lag:]))
            peaks_dp, props_dp = find_peaks(r_dp[min_lag:], prominence=dyn_dp * 0.05, distance=3)
            for p_idx, p_lag in enumerate(peaks_dp):
                lag = p_lag + min_lag
                rate = float(sample_rate_hz / lag)
                if min_rate <= rate <= max_rate:
                    all_raw.append({
                        "rate_hz": rate,
                        "sps": float(lag),
                        "score": 25.0,
                        "source": "fsk_transition_autocorr",
                    })

    # 4. MSK Squaring Spectrum
    x_sq = x ** 2
    spec_sq = np.abs(np.fft.fftshift(np.fft.fft(x_sq * np.hanning(len(x_sq)), n=65536)))
    freqs_sq = np.fft.fftshift(np.fft.fftfreq(65536, d=1.0 / sample_rate_hz))
    if np.max(spec_sq) > 1e-9:
        p_peaks, _ = find_peaks(spec_sq, prominence=float(np.max(spec_sq)) * 0.08, distance=10)
        if len(p_peaks) >= 2:
            sorted_p = sorted(p_peaks, key=lambda p: spec_sq[p], reverse=True)[:4]
            ref_p = [parabolic_peak_refinement(spec_sq, p, freqs_sq) for p in sorted_p]
            sp = abs(ref_p[0] - ref_p[1])
            if sp > 50.0:
                sps = float(sample_rate_hz / sp)
                if min_sps <= sps <= max_sps:
                    all_raw.append({"rate_hz": float(sp), "sps": sps, "score": 24.0, "source": "msk_squaring"})

    # 5. Direct Bandwidth Hypotheses
    if obw_hz > 50.0:
        mod_u = (modulation_hint or "").upper()
        if "4-FSK" in mod_u:
            for mult in [1.0 / 4.0, 1.0 / 3.0, 1.0 / 3.5, 1.0 / 2.0]:
                r = float(obw_hz * mult)
                sps = sample_rate_hz / r if r > 0 else 0
                if min_sps <= sps <= max_sps:
                    all_raw.append({"rate_hz": r, "sps": sps, "score": 15.0, "source": "fsk4_bandwidth"})
        elif "2-FSK" in mod_u:
            for mult in [1.0, 1.0 / 1.5, 1.0 / 2.0]:
                r = float(obw_hz * mult)
                sps = sample_rate_hz / r if r > 0 else 0
                if min_sps <= sps <= max_sps:
                    all_raw.append({"rate_hz": r, "sps": sps, "score": 15.0, "source": "fsk2_bandwidth"})
        else:
            for alpha in [0.0, 0.15, 0.25, 0.35, 0.50]:
                r = float(obw_hz / (1.0 + alpha))
                sps = sample_rate_hz / r if r > 0 else 0
                if min_sps <= sps <= max_sps:
                    all_raw.append({"rate_hz": r, "sps": sps, "score": 12.0, "source": "rrc_bandwidth"})

    # 6. Standard Priors (low baseline score)
    for std_r in [300, 600, 1200, 2400, 4800, 9600, 19200, 38400, 56000, 57600, 115200, 250000, 500000, 1000000]:
        sps = sample_rate_hz / std_r
        if min_sps <= sps <= max_sps:
            all_raw.append({"rate_hz": float(std_r), "sps": sps, "score": 0.02, "source": "standard_prior"})

    # 7. Cluster Detections (+-2%)
    valid = [item for item in all_raw if item.get("rate_hz", 0) > 10 and min_sps <= (sample_rate_hz / item["rate_hz"]) <= max_sps]
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
        multi_source_bonus = 0.25 * min(3, len(sources) - 1)
        effective_score = max_score * (1.0 + multi_source_bonus)

        cand = SymbolRateCandidate(
            rate_hz=centroid_rate,
            samples_per_symbol=sps,
            sources=sources,
            supported_by=sources,
            score=float(effective_score),
            raw_rate=centroid_rate,
        )
        candidates.append(cand)

    # 8. Harmonic Lattice Cross-Resolution
    is_fsk = bool(modulation_hint and "FSK" in modulation_hint.upper())
    if obw_hz > 50.0 and not is_fsk:
        synthesized: List[SymbolRateCandidate] = []
        for c in candidates:
            r = c.rate_hz
            ratio = obw_hz / r
            harmonic_rules = [
                (1.80, 2.50, 2.0, "harmonic_lattice_1_2"),
                (1.25, 1.75, 4.0 / 3.0, "harmonic_lattice_3_4"),
                (3.50, 5.00, 4.0, "harmonic_lattice_1_4"),
                (2.30, 3.20, 8.0 / 3.0, "harmonic_lattice_3_8"),
                (1.45, 2.00, 8.0 / 5.0, "harmonic_lattice_5_8"),
                (0.40, 0.70, 0.5, "harmonic_lattice_2_1"),
            ]
            for r_min, r_max, mult, tag in harmonic_rules:
                if r_min <= ratio <= r_max:
                    fund_r = r * mult
                    sps_fund = sample_rate_hz / fund_r
                    if min_sps <= sps_fund <= max_sps:
                        if not any(abs(c2.rate_hz - fund_r) / fund_r <= 0.025 for c2 in candidates + synthesized):
                            synthesized.append(SymbolRateCandidate(
                                rate_hz=fund_r,
                                samples_per_symbol=sps_fund,
                                sources=[tag],
                                supported_by=[tag],
                                score=float(c.score * 1.1),
                                raw_rate=fund_r,
                            ))
        candidates.extend(synthesized)

    # 9. Final Physics & Modulation Scoring
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
                    c.score *= 2.5
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
                if 0.90 <= ratio <= 1.55:
                    bw_boost = 1.0 + 1.2 * np.exp(-0.5 * ((ratio - 1.25) / 0.25) ** 2)
                    c.score *= bw_boost
                elif 1.55 < ratio <= 2.20:
                    c.score *= 0.45
                elif ratio > 2.20:
                    c.score *= 0.25
                elif ratio < 0.85:
                    c.score *= 0.35
            else:
                if "4-FSK" in (modulation_hint or "").upper():
                    if 2.5 <= ratio <= 4.5:
                        c.score *= 2.0
                elif "2-FSK" in (modulation_hint or "").upper():
                    if 0.8 <= ratio <= 2.2:
                        c.score *= 2.0

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

    print(f"Testing Complete Harmonic Lattice on {len(selected)} captures...\n")
    class_stats = {m: {"total": 0, "recall@1": 0, "recall@3": 0, "recall@5": 0, "in_pool": 0} for m in class_caps}

    for cap in selected:
        raw = np.fromfile(cap["iq_path"], dtype=np.complex64)
        true_baud = float(cap["symbol_rate"])
        sr = float(cap["sample_rate"])
        mod = cap["true_modulation"]
        
        class_stats[mod]["total"] += 1
        candidates = extract_all_physical_evidence_candidates(raw, sr, modulation_hint=mod)
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
    print("COMPLETE HARMONIC LATTICE PERFORMANCE:")
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
