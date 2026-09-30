"""
ASTRA Symbol-Rate Candidate Generation, Deduplication, and Harmonic Association V2.
Synthesizes raw candidates from all DSP estimators into robustly ranked hypotheses
with harmonic lattice resolution and bandwidth consistency validation.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np

from .models import DSPRateEvidence, SymbolRateCandidate


class SymbolRateCandidateGenerator:
    """
    Synthesizes raw candidates from all DSP estimators into deduplicated hypotheses.
    """

    def __init__(
        self,
        merge_tolerance_percent: float = 2.0,
        max_raw_candidates: int = 60,
        max_merged_candidates: int = 24,
        standard_rates: Optional[Sequence[float]] = None,
    ):
        self.merge_tolerance = merge_tolerance_percent / 100.0
        self.max_raw_candidates = max_raw_candidates
        self.max_merged_candidates = max_merged_candidates
        self.standard_rates = list(standard_rates) if standard_rates else [
            300, 600, 1200, 2400, 4800, 9600, 19200, 38400, 56000, 57600,
            115200, 250000, 500000, 1000000
        ]

    def deduplicate_candidates(
        self,
        raw_candidates: List[Dict[str, Any]],
        sample_rate_hz: float,
        occupied_bandwidth_hz: float = 0.0,
        modulation_hint: Optional[str] = None,
    ) -> List[SymbolRateCandidate]:
        """
        Clusters raw candidate rates within merge tolerance (e.g. +-2%) into unified hypotheses.
        Uses continuous confidence scoring and harmonic lattice resolution.
        """
        if not raw_candidates:
            return []

        # Filter out invalid frequencies outside physical sampling limits
        valid_items = []
        for item in raw_candidates:
            r = float(item.get("rate_hz", 0.0))
            if r <= 10.0 or (sample_rate_hz / r) < 1.4 or (sample_rate_hz / r) > 350.0:
                continue
            valid_items.append(item)

        if not valid_items:
            return []

        # Sort descending by raw score
        sorted_raw = sorted(valid_items, key=lambda c: c.get("score", 0.0), reverse=True)
        clusters: List[List[Dict[str, Any]]] = []

        for item in sorted_raw:
            r = item["rate_hz"]
            matched = False
            for cl in clusters:
                rep_r = cl[0]["rate_hz"]
                if abs(r - rep_r) / max(1.0, rep_r) <= self.merge_tolerance:
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

            # Composite continuous score: preserves full dynamic range (no 1.0 clipping)
            has_primary = any(s in ["cyclostationary", "instantaneous_frequency", "msk_squaring", "fsk_tone_spacing"] for s in sources)
            multi_source_bonus = 0.25 * min(3, len(sources) - 1)
            effective_score = max_score * (1.0 + multi_source_bonus)
            if has_primary:
                effective_score *= 1.2

            # Modulation-specific bonus and filtering
            if modulation_hint:
                mod_u = modulation_hint.upper()
                if "MSK" in mod_u:
                    if "msk_squaring" in sources:
                        effective_score *= 2.5
                elif "FSK" in mod_u:
                    if any("instantaneous" in s or "fsk" in s for s in sources):
                        effective_score *= 2.0
                    # Cyclostationary envelope alone is weak for constant envelope FSK
                    if all(s == "cyclostationary" for s in sources):
                        effective_score *= 0.4
                elif "PSK" in mod_u or "QAM" in mod_u:
                    if "cyclostationary" in sources:
                        effective_score *= 1.5
                    # Phase jumps cause spurious IF and MSK squaring artifacts for PSK/QAM
                    if all(s in ["instantaneous_frequency", "msk_squaring"] for s in sources):
                        effective_score *= 0.15

            # Bandwidth consistency validation
            is_fsk = bool(modulation_hint and "FSK" in modulation_hint.upper())
            if occupied_bandwidth_hz > 50.0:
                ratio = occupied_bandwidth_hz / centroid_rate
                if not is_fsk:
                    # Single-carrier Nyquist/RRC pulse shaping: OBW is ~ (1 + alpha) * Rs
                    if 0.8 <= ratio <= 1.4:
                        effective_score *= 1.4  # Ideal single-carrier passband match
                    elif ratio >= 1.7:
                        effective_score *= 0.5  # Likely subharmonic (Rs/2 or Rs/4)
                    elif ratio <= 0.6:
                        effective_score *= 0.4  # Likely harmonic (2*Rs or 3*Rs)
                else:
                    # Multi-tone FSK: OBW is (M-1)*df + Rs >= Rs
                    if ratio >= 0.8:
                        effective_score *= 1.2
                    elif ratio < 0.5:
                        effective_score *= 0.4

            # Standard grid proximity check (priors only, without overriding true non-standard bauds)
            snapped = None
            snap_dist = None
            if self.standard_rates:
                std_arr = np.array(self.standard_rates)
                closest_idx = np.argmin(np.abs(std_arr - centroid_rate))
                closest_std = float(std_arr[closest_idx])
                dist_pct = abs(centroid_rate - closest_std) / closest_std
                if dist_pct <= 0.02:  # Within 2% of standard rate
                    snapped = closest_std
                    snap_dist = float(dist_pct * 100.0)

            cand = SymbolRateCandidate(
                rate_hz=centroid_rate,
                samples_per_symbol=sps,
                sources=sources,
                supported_by=sources,
                score=float(effective_score),
                raw_rate=centroid_rate,
                snapped_rate=snapped,
                snap_distance_percent=snap_dist,
            )
            candidates.append(cand)

        # Detect harmonic relationships between candidates
        for i, c1 in enumerate(candidates):
            for j, c2 in enumerate(candidates):
                if i != j and c1.rate_hz > 0:
                    ratio = c2.rate_hz / c1.rate_hz
                    for target_ratio in [2.0, 0.5, 4.0, 0.25, 3.0, 1.0 / 3.0]:
                        if abs(ratio - target_ratio) / target_ratio <= 0.025:
                            c1.harmonic_ratio = float(target_ratio)
                            break

        # Multi-Order Harmonic Lattice Resolution & Subharmonic Synthesis
        is_fsk = bool(modulation_hint and "FSK" in modulation_hint.upper())
        if occupied_bandwidth_hz > 50.0 and not is_fsk:
            synthesized: List[SymbolRateCandidate] = []
            for c in candidates:
                r = c.rate_hz
                ratio = occupied_bandwidth_hz / r
                
                # Check for all standard physical subharmonic relationships
                harmonic_transforms = [
                    (1.75, 2.55, 2.0, "harmonic_lattice_1_2"),        # 1/2 Rs -> 2.0 * r
                    (1.25, 1.75, 4.0 / 3.0, "harmonic_lattice_3_4"),  # 3/4 Rs -> 1.333 * r
                    (3.40, 5.20, 4.0, "harmonic_lattice_1_4"),        # 1/4 Rs -> 4.0 * r
                    (2.25, 3.35, 8.0 / 3.0, "harmonic_lattice_3_8"),  # 3/8 Rs -> 2.667 * r
                    (1.40, 2.05, 8.0 / 5.0, "harmonic_lattice_5_8"),  # 5/8 Rs -> 1.6 * r
                    (0.40, 0.70, 0.5, "harmonic_lattice_2_1"),        # 2x overtone -> 0.5 * r
                ]
                
                for r_min, r_max, mult, tag in harmonic_transforms:
                    if r_min <= ratio <= r_max:
                        fund_r = r * mult
                        sps_fund = sample_rate_hz / fund_r
                        if 1.4 <= sps_fund <= 500.0:
                            if not any(abs(c2.rate_hz - fund_r) / fund_r <= 0.025 for c2 in candidates + synthesized):
                                synthesized.append(SymbolRateCandidate(
                                    rate_hz=fund_r,
                                    samples_per_symbol=sps_fund,
                                    sources=[tag],
                                    supported_by=[tag],
                                    score=float(c.score * 1.15),
                                    raw_rate=fund_r,
                                ))
            candidates.extend(synthesized)
        elif occupied_bandwidth_hz > 50.0 and is_fsk:
            # For multi-tone FSK, synthesize direct baud candidates from OBW
            mod_u = modulation_hint.upper() if modulation_hint else ""
            fsk_synth: List[SymbolRateCandidate] = []
            if "4-FSK" in mod_u:
                for mult, tag in [(1.0 / 4.0, "fsk4_obw_div4"), (1.0 / 3.5, "fsk4_obw_div3_5"), (1.0 / 3.0, "fsk4_obw_div3")]:
                    fr = occupied_bandwidth_hz * mult
                    sps = sample_rate_hz / fr if fr > 0 else 0
                    if 1.4 <= sps <= 500.0 and not any(abs(c2.rate_hz - fr) / fr <= 0.025 for c2 in candidates + fsk_synth):
                        fsk_synth.append(SymbolRateCandidate(
                            rate_hz=fr,
                            samples_per_symbol=sps,
                            sources=[tag],
                            supported_by=[tag],
                            score=16.0,
                            raw_rate=fr,
                        ))
            elif "2-FSK" in mod_u:
                for mult, tag in [(1.0, "fsk2_obw_1_0"), (1.0 / 1.5, "fsk2_obw_div1_5"), (1.0 / 2.0, "fsk2_obw_div2")]:
                    fr = occupied_bandwidth_hz * mult
                    sps = sample_rate_hz / fr if fr > 0 else 0
                    if 1.4 <= sps <= 500.0 and not any(abs(c2.rate_hz - fr) / fr <= 0.025 for c2 in candidates + fsk_synth):
                        fsk_synth.append(SymbolRateCandidate(
                            rate_hz=fr,
                            samples_per_symbol=sps,
                            sources=[tag],
                            supported_by=[tag],
                            score=16.0,
                            raw_rate=fr,
                        ))
            candidates.extend(fsk_synth)

        # Cross-validation and Bandwidth Consistency Scoring
        for c in candidates:
            r = c.rate_hz
            if r <= 0:
                continue

            if occupied_bandwidth_hz > 50.0:
                ratio = occupied_bandwidth_hz / r
                if not is_fsk:
                    # Ideal single-carrier passband: OBW is in [0.95, 1.55] * Rs
                    if 0.95 <= ratio <= 1.55:
                        bw_boost = 1.0 + 1.25 * float(np.exp(-0.5 * ((ratio - 1.25) / 0.25) ** 2))
                        c.score *= bw_boost
                    elif 1.55 < ratio <= 2.20:
                        c.score *= 0.40  # 3/4 or 5/8 subharmonic penalty
                    elif ratio > 2.20:
                        c.score *= 0.20  # Deep subharmonic penalty
                    elif ratio < 0.85:
                        c.score *= 0.30  # Overtone harmonic penalty
                else:
                    if "4-FSK" in (modulation_hint or "").upper():
                        if 2.5 <= ratio <= 4.8:
                            c.score *= 1.8
                    elif "2-FSK" in (modulation_hint or "").upper():
                        if 0.8 <= ratio <= 2.4:
                            c.score *= 1.8

        # Inter-candidate Harmonic Sibling Disambiguation (Resolves 3/4 Rs and 1/2 Rs traps)
        if not is_fsk and occupied_bandwidth_hz > 50.0:
            for c1 in candidates:
                r1 = c1.rate_hz
                if r1 <= 0:
                    continue
                # Look for 4/3x sibling (fundamental Rs for 3/4 Rs subharmonic)
                sibs_43 = [c2 for c2 in candidates if abs(c2.rate_hz - (4.0 / 3.0) * r1) / ((4.0 / 3.0) * r1) <= 0.03]
                if sibs_43:
                    c2 = sibs_43[0]
                    ratio_2 = occupied_bandwidth_hz / c2.rate_hz
                    if 0.85 <= ratio_2 <= 1.45:
                        c2.score = max(c2.score, c1.score * 1.4)
                        c1.score *= 0.30

                # Look for 2x sibling (fundamental Rs for 1/2 Rs subharmonic)
                sibs_2 = [c2 for c2 in candidates if abs(c2.rate_hz - 2.0 * r1) / (2.0 * r1) <= 0.03]
                if sibs_2:
                    c2 = sibs_2[0]
                    ratio_2 = occupied_bandwidth_hz / c2.rate_hz
                    if 0.85 <= ratio_2 <= 1.45:
                        c2.score = max(c2.score, c1.score * 1.4)
                        c1.score *= 0.30

        # Sort candidates descending by effective continuous score
        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates[: self.max_merged_candidates]

    def generate(
        self,
        evidence: DSPRateEvidence,
        sample_rate_hz: float,
        modulation_hint: Optional[str] = None,
    ) -> List[SymbolRateCandidate]:
        """
        Aggregates all candidates from DSPRateEvidence into deduplicated candidates.
        """
        all_raw: List[Dict[str, Any]] = []

        all_raw.extend(evidence.autocorr_peaks)
        all_raw.extend(evidence.power_autocorr_peaks)
        all_raw.extend(evidence.if_peaks)
        all_raw.extend(evidence.spectral_peak_spacings)
        all_raw.extend(evidence.bandwidth_candidates)
        all_raw.extend(evidence.cyclostationary_peaks)

        # Standard rate priors are added with low baseline score (0.02)
        for std_r in self.standard_rates:
            sps = sample_rate_hz / std_r
            if 1.5 <= sps <= 256.0:
                all_raw.append({
                    "rate_hz": float(std_r),
                    "sps": sps,
                    "score": 0.02,
                    "source": "standard_prior",
                })

        return self.deduplicate_candidates(
            all_raw,
            sample_rate_hz,
            occupied_bandwidth_hz=getattr(evidence, "occupied_bandwidth_hz", 0.0),
            modulation_hint=modulation_hint,
        )


def deduplicate_candidates(
    candidates: Sequence[SymbolRateCandidate],
    tolerance_percent: float = 2.0,
    sample_rate_hz: float = 192000.0,
) -> List[SymbolRateCandidate]:
    """
    Module-level helper to deduplicate already instantiated candidates.
    """
    raw_dicts = []
    for c in candidates:
        raw_dicts.append({
            "rate_hz": float(c.rate_hz),
            "score": float(c.score),
            "source": c.sources[0] if c.sources else "unknown",
        })
    gen = SymbolRateCandidateGenerator(merge_tolerance_percent=tolerance_percent)
    return gen.deduplicate_candidates(raw_dicts, sample_rate_hz)


def detect_harmonic_candidates(
    candidates: Sequence[SymbolRateCandidate],
    sample_rate_hz: float = 192000.0,
) -> List[SymbolRateCandidate]:
    """
    Module-level helper to detect harmonic relationships among candidates.
    """
    cands_copy = [
        SymbolRateCandidate(
            rate_hz=c.rate_hz,
            samples_per_symbol=c.samples_per_symbol,
            sources=list(c.sources),
            features=dict(c.features),
            score=c.score,
            supported_by=list(c.supported_by),
        )
        for c in candidates
    ]
    for i, c1 in enumerate(cands_copy):
        for j, c2 in enumerate(cands_copy):
            if i != j and c1.rate_hz > 0:
                ratio = c2.rate_hz / c1.rate_hz
                for target_ratio in [2.0, 0.5, 4.0, 0.25, 3.0, 1.0 / 3.0]:
                    if abs(ratio - target_ratio) / target_ratio <= 0.025:
                        c1.harmonic_ratio = float(target_ratio)
                        break
    return cands_copy


def generate_candidates(
    evidence: DSPRateEvidence,
    sample_rate_hz: float,
    config: Optional[Dict[str, Any]] = None,
    modulation_hint: Optional[str] = None,
) -> List[SymbolRateCandidate]:
    """
    Module-level helper to generate deduplicated candidates from DSP evidence.
    """
    cfg = config.get("symbol_rate", {}) if config else {}
    merge_tol = float(cfg.get("candidate_merge_tolerance_percent", 2.0))
    std_rates = cfg.get("standard_rates", {}).get("values", None) if cfg.get("standard_rates", {}).get("enabled", True) else None

    gen = SymbolRateCandidateGenerator(
        merge_tolerance_percent=merge_tol,
        standard_rates=std_rates,
    )
    return gen.generate(evidence, sample_rate_hz, modulation_hint=modulation_hint)
