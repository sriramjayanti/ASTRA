"""
inference.py
Main Bitstream Intelligence Engine orchestrator for ASTRA Stage 12.
"""

from typing import List, Dict, Optional, Any, Union
from pathlib import Path
import os
import yaml
import numpy as np

from .models import (
    StructureStatus,
    BitstreamIntelligenceResult,
    AutocorrPeak,
    PeriodicityCandidate,
    SyncPatternResult,
    RepeatedPattern,
    RunLengthStats,
    BitBalance,
    ByteAlignmentResult,
    StructuralRegion
)
from .validation import validate_bitstream
from .bit_balance import calculate_bit_balance
from .entropy import (
    binary_entropy,
    compute_all_ngram_entropies,
    sliding_entropy,
    detect_entropy_change_points
)
from .autocorrelation import (
    bit_autocorrelation,
    find_autocorrelation_peaks
)
from .periodicity import (
    merge_harmonics,
    rank_period_candidates
)
from .cross_correlation import search_known_sync
from .repeated_patterns import find_repeated_patterns
from .sync_candidates import discover_candidate_sync
from .frame_length import estimate_frame_lengths
from .segmentation import (
    build_frame_matrix,
    compute_position_stability,
    compute_position_entropy,
    detect_structural_regions,
    find_optimal_frame_offset
)
from .run_length import calculate_run_length_stats
from .byte_alignment import analyze_byte_offsets
from .feature_builder import BitstreamFeatureBuilder, FEATURE_SCHEMA_VERSION
from .utils import compute_bitstream_hash


class BitstreamIntelligenceEngine:
    """
    Production-ready Bitstream Intelligence Engine (ASTRA Stage 12).
    Discovers periodic framing, sync words, entropy transitions, byte alignment,
    and constructs multi-channel sequence feature representations for Stage 13 neural sequence modeling.
    """

    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.feature_builder = BitstreamFeatureBuilder(
            schema_version=self.config.get("feature_schema_version", FEATURE_SCHEMA_VERSION)
        )
        self._cache: Dict[str, BitstreamIntelligenceResult] = {}

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        default_config = {
            "version": "1.0.0",
            "feature_schema_version": "bitstream_features_v1",
            "min_bits_for_entropy": 16,
            "min_bits_for_autocorrelation": 64,
            "min_frames_for_repetition": 3,
            "min_sync_occurrences": 2,
            "entropy": {
                "enabled": True,
                "window_sizes": [64, 128, 256, 512],
                "sliding_step": 16,
                "ngram_sizes": [1, 2, 4, 8],
                "change_point_threshold": 0.15,
            },
            "autocorrelation": {
                "enabled": True,
                "max_lag_fraction": 0.50,
                "min_lag_bits": 16,
                "max_lag_bits": 4096,
                "top_k_peaks": 20,
                "peak_threshold": 0.08,
                "fft_threshold_bits": 2048,
            },
            "periodicity": {
                "top_k": 10,
                "harmonic_merge_tolerance": 0.05,
            },
            "frame_length": {
                "top_k": 5,
                "min_length_bits": 32,
                "max_length_bits": 4096,
                "candidate_standard_hints": [64, 128, 256, 512, 1024, 2048],
            },
            "sync": {
                "enabled": True,
                "known_patterns": ["EB90", "1ACFFC1D", "FAF334"],
                "candidate_lengths": [16, 24, 32, 64],
                "max_hamming_fraction": 0.0,
            },
            "repeated_patterns": {
                "enabled": True,
                "lengths": [8, 16, 24, 32, 64],
                "min_occurrences": 3,
            },
            "byte_alignment": {
                "enabled": True,
            },
            "segmentation": {
                "enabled": True,
                "max_frames_to_matrix": 64,
                "stability_fixed_threshold": 0.85,
                "stability_variable_threshold": 0.60,
            }
        }

        if config_path and os.path.exists(config_path):
            with open(config_path, "r") as f:
                loaded = yaml.safe_load(f)
                if loaded and "bitstream_intelligence" in loaded:
                    return loaded["bitstream_intelligence"]
                elif loaded:
                    return loaded

        pkg_conf = Path(__file__).parent.parent / "configs" / "bitstream_intelligence_config.yaml"
        if pkg_conf.exists():
            with open(pkg_conf, "r") as f:
                loaded = yaml.safe_load(f)
                if loaded and "bitstream_intelligence" in loaded:
                    return loaded["bitstream_intelligence"]

        return default_config

    def analyze(
        self,
        decoded_bits: np.ndarray,
        context: Optional[Dict[str, Any]] = None
    ) -> BitstreamIntelligenceResult:
        """
        Main analysis entrypoint for a recovered candidate bitstream.
        """
        context = context or {}
        pipeline_path_id = str(context.get("pipeline_path_id", "candidate_0"))
        soft_info = context.get("soft_information") if context.get("soft_information") is not None else context.get("soft_info")
        known_syncs = context.get("known_sync_words") or self.config["sync"]["known_patterns"]
        validation_hints = context.get("frame_length_hints") or []

        # 1. Validation & Sanitization
        sanitized_bits, sanitized_soft = validate_bitstream(
            decoded_bits,
            soft_info=soft_info,
            min_bits=self.config.get("min_bits_for_entropy", 16)
        )
        n_bits = len(sanitized_bits)

        # 2. Bitstream Hash & Deduplication check
        bit_hash = compute_bitstream_hash(sanitized_bits)
        if bit_hash in self._cache:
            cached_res = self._cache[bit_hash]
            cached_res.pipeline_path_id = pipeline_path_id
            return cached_res

        # 3. Global Bit Balance
        balance = calculate_bit_balance(sanitized_bits)

        # 4. Global Binary Entropy & Block N-Gram Entropies
        h_binary = binary_entropy(sanitized_bits)
        ngram_ents = compute_all_ngram_entropies(
            sanitized_bits,
            ngram_sizes=self.config["entropy"]["ngram_sizes"]
        )

        # 5. Sliding-Window Entropy & Change Points
        win_size = self.config["entropy"]["window_sizes"][1] if len(self.config["entropy"]["window_sizes"]) > 1 else 128
        step = self.config["entropy"]["sliding_step"]
        ent_windows = sliding_entropy(sanitized_bits, window_size=win_size, step=step)
        change_points = detect_entropy_change_points(
            ent_windows,
            threshold=self.config["entropy"]["change_point_threshold"],
            step=step
        )

        # 6. Run-Length Statistics
        run_stats = calculate_run_length_stats(sanitized_bits)

        # 7. Autocorrelation & Peak Detection
        peaks: List[AutocorrPeak] = []
        if n_bits >= self.config.get("min_bits_for_autocorrelation", 64):
            autocorr_curve = bit_autocorrelation(
                sanitized_bits,
                max_lag_fraction=self.config["autocorrelation"]["max_lag_fraction"],
                max_lag_limit=self.config["autocorrelation"]["max_lag_bits"],
                fft_threshold=self.config["autocorrelation"]["fft_threshold_bits"]
            )
            peaks = find_autocorrelation_peaks(
                autocorr_curve,
                min_lag=self.config["autocorrelation"]["min_lag_bits"],
                peak_threshold=self.config["autocorrelation"]["peak_threshold"],
                top_k=self.config["autocorrelation"]["top_k_peaks"]
            )

        # 8. Periodicity & Harmonic Merging
        period_candidates = merge_harmonics(
            peaks,
            tolerance=self.config["periodicity"]["harmonic_merge_tolerance"]
        )

        # 9. Search Known Sync Words
        sync_results: List[SyncPatternResult] = []
        if self.config["sync"]["enabled"] and known_syncs:
            for pattern in known_syncs:
                sr = search_known_sync(
                    sanitized_bits,
                    pattern=pattern,
                    pattern_id=f"sync_{pattern}",
                    max_hamming_fraction=self.config["sync"]["max_hamming_fraction"],
                    soft_info=sanitized_soft
                )
                if sr.match_count > 0:
                    sync_results.append(sr)

        # 10. Repeated Patterns Discovery
        repeated_patterns: List[RepeatedPattern] = []
        if self.config["repeated_patterns"]["enabled"]:
            repeated_patterns = find_repeated_patterns(
                sanitized_bits,
                pattern_lengths=self.config["repeated_patterns"]["lengths"],
                min_occurrences=self.config["repeated_patterns"]["min_occurrences"]
            )

        # 11. Blind Candidate Sync Discovery
        candidate_syncs = discover_candidate_sync(
            sanitized_bits,
            candidate_lengths=self.config["sync"]["candidate_lengths"],
            candidate_frame_lengths=period_candidates,
            soft_info=sanitized_soft
        )

        # 12. Frame-Length Estimation
        sync_spacings = [s.mean_spacing for s in sync_results + candidate_syncs if s.mean_spacing > 0 and s.confidence >= 0.65]
        ranked_periods = rank_period_candidates(
            period_candidates,
            sync_spacings=sync_spacings,
            validation_hints=validation_hints,
            top_k=self.config["periodicity"]["top_k"]
        )

        frame_candidates = estimate_frame_lengths(
            sanitized_bits,
            periodicity_candidates=ranked_periods,
            sync_results=sync_results,
            candidate_sync_words=candidate_syncs,
            validation_hints=validation_hints,
            standard_hints=self.config["frame_length"]["candidate_standard_hints"],
            top_k=self.config["frame_length"]["top_k"],
            min_length_bits=self.config["frame_length"]["min_length_bits"],
            max_length_bits=self.config["frame_length"]["max_length_bits"]
        )

        # 13. Deep Frame Segmentation & Regional Structure Analysis (on Top-1 frame candidate)
        pos_stability_map: List[float] = []
        pos_entropy_map: List[float] = []
        structural_regions: List[StructuralRegion] = []
        top_frame_len = None

        if frame_candidates:
            top_frame_len = frame_candidates[0].period_bits
            best_offset, _ = find_optimal_frame_offset(sanitized_bits, frame_length=top_frame_len)
            frame_mat = build_frame_matrix(
                sanitized_bits,
                frame_length=top_frame_len,
                offset=best_offset,
                max_frames=self.config["segmentation"]["max_frames_to_matrix"]
            )
            if frame_mat.shape[0] >= 2:
                stab = compute_position_stability(frame_mat)
                ent = compute_position_entropy(frame_mat)
                pos_stability_map = stab.tolist()
                pos_entropy_map = ent.tolist()
                structural_regions = detect_structural_regions(
                    stab, ent,
                    fixed_threshold=self.config["segmentation"]["stability_fixed_threshold"],
                    variable_threshold=self.config["segmentation"]["stability_variable_threshold"]
                )

        # 14. Byte Alignment Exploration
        byte_alignment = analyze_byte_offsets(sanitized_bits)

        # 15. Overall Status Classification
        top_frame_score = frame_candidates[0].score if frame_candidates else 0.0
        sync_found = (len(sync_results) > 0 and sync_results[0].confidence >= 0.70 and sync_results[0].match_count >= 2) or \
                     (len(candidate_syncs) > 0 and candidate_syncs[0].confidence >= 0.70 and candidate_syncs[0].match_count >= 2)
        strong_autocorr = len(peaks) > 0 and peaks[0].correlation >= 0.40

        if (top_frame_score >= 0.65 and sync_found) or (top_frame_score >= 0.80):
            status = StructureStatus.STRUCTURE_STRONG
        elif (top_frame_score >= 0.50 and strong_autocorr) or (sync_found and top_frame_score >= 0.40):
            status = StructureStatus.STRUCTURE_MODERATE
        elif len(peaks) > 0 or len(repeated_patterns) > 0 or top_frame_score > 0.20:
            status = StructureStatus.STRUCTURE_WEAK
        else:
            status = StructureStatus.STRUCTURE_UNKNOWN

        # 16. Stage 13 Feature Construction
        global_features = self.feature_builder.build_global_features(
            bit_count=n_bits,
            bit_balance=balance,
            binary_entropy_global=h_binary,
            ngram_entropies=ngram_ents,
            run_length_stats=run_stats,
            autocorr_peaks=peaks,
            periodicity_candidates=ranked_periods,
            frame_length_candidates=frame_candidates,
            repeated_patterns=repeated_patterns,
            byte_alignment=byte_alignment
        )

        confirmed_sync_positions = []
        if sync_results and sync_results[0].confidence >= 0.60:
            confirmed_sync_positions = sync_results[0].positions
        elif candidate_syncs and candidate_syncs[0].confidence >= 0.60:
            confirmed_sync_positions = candidate_syncs[0].positions

        sequence_map = self.feature_builder.build_sequence_feature_map(
            bits=sanitized_bits,
            soft_info=sanitized_soft,
            entropy_windows=ent_windows,
            window_step=step,
            top_frame_length=top_frame_len,
            position_stability=np.array(pos_stability_map, dtype=np.float32) if pos_stability_map else None,
            sync_positions=confirmed_sync_positions
        )

        # 17. Processing History Entry
        history_entry = {
            "stage": "bitstream_intelligence",
            "entropy_global": round(h_binary, 4),
            "top_frame_length": top_frame_len,
            "periodicity_score": round(top_frame_score, 4),
            "sync_candidates_count": len(sync_results) + len(candidate_syncs),
            "status": status.value
        }

        result = BitstreamIntelligenceResult(
            pipeline_path_id=pipeline_path_id,
            bit_count=n_bits,
            status=status,
            bit_balance=balance,
            binary_entropy_global=h_binary,
            ngram_entropies=ngram_ents,
            run_length_stats=run_stats,
            autocorrelation_peaks=peaks,
            periodicity_candidates=ranked_periods,
            frame_length_candidates=frame_candidates,
            sync_results=sync_results,
            candidate_sync_words=candidate_syncs,
            repeated_patterns=repeated_patterns,
            byte_alignment=byte_alignment,
            structural_regions=structural_regions,
            position_stability_map=pos_stability_map,
            position_entropy_map=pos_entropy_map,
            entropy_windows=ent_windows,
            entropy_change_points=change_points,
            global_features=global_features,
            sequence_feature_map=sequence_map,
            processing_history=[history_entry],
            decoded_bits_hash=bit_hash
        )

        self._cache[bit_hash] = result
        return result

    def analyze_candidate(
        self,
        candidate: Any,
        context: Optional[Dict[str, Any]] = None
    ) -> BitstreamIntelligenceResult:
        """
        Analyze a candidate pipeline object from Stage 10/11.
        """
        context = dict(context or {})

        bits = getattr(candidate, "decoded_hard_bits", None)
        if bits is None and isinstance(candidate, dict):
            bits = candidate.get("decoded_hard_bits") or candidate.get("bits")

        if bits is None:
            raise ValueError(f"Candidate {candidate} does not contain 'decoded_hard_bits'.")

        soft = getattr(candidate, "soft_information", None)
        if soft is None:
            soft = getattr(candidate, "decoded_soft_bits", None)
        if soft is None and isinstance(candidate, dict):
            soft = candidate.get("soft_information") or candidate.get("decoded_soft_bits")

        if soft is not None:
            context["soft_information"] = soft

        cand_id = getattr(candidate, "pipeline_path_id", None) or getattr(candidate, "candidate_id", None)
        if cand_id is None and isinstance(candidate, dict):
            cand_id = candidate.get("pipeline_path_id") or candidate.get("candidate_id")
        if cand_id is not None:
            context["pipeline_path_id"] = str(cand_id)

        return self.analyze(bits, context=context)

    def analyze_batch(
        self,
        candidates: List[Any],
        top_k: int = 3,
        context: Optional[Dict[str, Any]] = None
    ) -> List[BitstreamIntelligenceResult]:
        """
        Analyze top-K candidates from Stage 11 independently without mixing bitstreams.
        """
        results = []
        for cand in candidates[:top_k]:
            res = self.analyze_candidate(cand, context=context)
            results.append(res)
        return results
