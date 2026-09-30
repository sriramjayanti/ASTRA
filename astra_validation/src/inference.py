"""
inference.py
Main ValidationEngine pipeline entrypoint for ASTRA Stage 10.
Evaluates decoded candidate bitstreams from Stage 9 across multi-mechanism independent checks.
"""

from typing import Any, Dict, List, Optional, Union
import numpy as np
import yaml
import os
import time

from .models import (
    ValidationStatus,
    EvidenceCheckState,
    CRCProfile,
    CRCResult,
    ParityResult,
    SyndromeResult,
    ReencodingResult,
    FrameRepetitionResult,
    SyncWordResult,
    HeaderConsistencyResult,
    LengthConsistencyResult,
    StructuralResult,
    ValidationResult,
)
from .crc import load_crc_profiles, search_crc_candidates, check_crc_frame
from .parity import check_even_parity, check_odd_parity, check_block_parity
from .syndrome import normalize_syndrome_evidence
from .repetition import score_frame_repetition, segment_by_period
from .sync_word import search_sync_word, hex_to_bits
from .headers import validate_header_schema
from .length_checks import validate_length_field
from .structure import extract_structural_evidence
from .decoder_support import reencode_and_compare
from .scoring import (
    evaluate_evidence_groups,
    compute_overall_validation_score,
    determine_validation_status,
    extract_stage11_features,
)
from .utils import compute_bitstream_hash


class ValidationEngine:
    """
    ASTRA Stage 10 Validation Engine.
    Evaluates recovered candidate bitstreams from Stage 9 across independent validation mechanisms.
    """

    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.crc_profiles = load_crc_profiles()
        self.protocol_hints = self._load_protocol_hints()

    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        if config_path is None or not os.path.exists(config_path):
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            config_path = os.path.join(base_dir, "configs", "validation_config.yaml")

        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f).get("validation", {})
        return {}

    def _load_protocol_hints(self) -> Dict[str, Any]:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        hints_path = os.path.join(base_dir, "configs", "protocol_hints.yaml")
        if os.path.exists(hints_path):
            with open(hints_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f).get("protocol_hints", {})
        return {}

    def validate(
        self,
        fec_candidate: Any,
        context: Optional[Dict[str, Any]] = None,
    ) -> ValidationResult:
        """
        Validate a single FEC candidate result from Stage 9.
        """
        t0 = time.perf_counter()
        context = context or {}
        mode = context.get("mode", self.config.get("mode", "blind"))

        # Extract candidate attributes
        if hasattr(fec_candidate, "path_id"):
            path_id = fec_candidate.path_id
            cand_id = fec_candidate.candidate_id
            parent_cand_id = getattr(fec_candidate, "parent_candidate_id", cand_id)
            demod_id = fec_candidate.demod_variant_id
            inter_id = fec_candidate.interleaver_candidate_id
            fec_id = fec_candidate.fec_candidate_id
            family = fec_candidate.fec_family
            params = fec_candidate.fec_parameters
            decoded_bits = fec_candidate.decoded_hard_bits
            history = list(getattr(fec_candidate, "lineage", getattr(fec_candidate, "processing_history", [])))
        elif isinstance(fec_candidate, dict):
            path_id = fec_candidate.get("path_id", "path_0")
            cand_id = fec_candidate.get("candidate_id", "cand_0")
            parent_cand_id = fec_candidate.get("parent_candidate_id", cand_id)
            demod_id = fec_candidate.get("demod_variant_id", "demod_0")
            inter_id = fec_candidate.get("interleaver_candidate_id", "inter_0")
            fec_id = fec_candidate.get("fec_candidate_id", "fec_0")
            family = fec_candidate.get("fec_family", "none")
            params = fec_candidate.get("fec_parameters", {})
            decoded_bits = fec_candidate.get("decoded_hard_bits")
            history = list(fec_candidate.get("lineage", fec_candidate.get("processing_history", [])))
        else:
            decoded_bits = np.asarray(fec_candidate, dtype=np.uint8) if fec_candidate is not None else np.array([])
            path_id = "direct_bits"
            cand_id = "cand_0"
            parent_cand_id = "cand_0"
            demod_id = "demod_0"
            inter_id = "inter_0"
            fec_id = "fec_0"
            family = "none"
            params = {}
            history = []

        if decoded_bits is None or len(decoded_bits) == 0:
            return ValidationResult(
                pipeline_path_id=path_id,
                candidate_id=cand_id,
                parent_candidate_id=parent_cand_id,
                demod_variant_id=demod_id,
                interleaver_candidate_id=inter_id,
                fec_candidate_id=fec_id,
                lineage=history,
                validation_status=ValidationStatus.VALIDATION_FAILED,
                failure_reason="No decoded bitstream available",
            )

        bits = np.asarray(decoded_bits, dtype=np.uint8).ravel()
        bits_hash = compute_bitstream_hash(bits)

        # 1. CRC Verification
        crc_results = []
        if self.config.get("crc", {}).get("enabled", True):
            try:
                candidate_lens = context.get("candidate_frame_lengths", self.config.get("crc", {}).get("candidate_frame_lengths", [64, 128, 256, 512, 1024]))
                crc_results = search_crc_candidates(
                    bits,
                    self.crc_profiles,
                    candidate_lens,
                    max_offsets=self.config.get("crc", {}).get("max_search_offsets", 8),
                )
            except Exception as e:
                crc_results = []

        # 2. Parity Checks
        parity_results = []
        if self.config.get("parity", {}).get("enabled", True):
            try:
                p_even = check_even_parity(bits, block_size=8)
                parity_results.append(p_even)
            except Exception:
                pass

        # 3. FEC Syndrome / Metrics Normalization
        syndrome_res = SyndromeResult()
        if self.config.get("syndrome", {}).get("enabled", True):
            try:
                syndrome_res = normalize_syndrome_evidence(fec_candidate)
            except Exception:
                syndrome_res = SyndromeResult(check_state=EvidenceCheckState.NOT_TESTED)

        # 4. Frame Repetition & Periodicity
        repetition_res = FrameRepetitionResult()
        if self.config.get("repetition", {}).get("enabled", True):
            try:
                repetition_res = score_frame_repetition(
                    bits,
                    min_period=self.config.get("repetition", {}).get("min_period_bits", 32),
                    max_period=self.config.get("repetition", {}).get("max_period_bits", 2048),
                )
            except Exception:
                repetition_res = FrameRepetitionResult(check_state=EvidenceCheckState.NOT_TESTED)

        # 5. Sync Word Detection
        sync_results = []
        if self.config.get("sync_word", {}).get("enabled", True):
            try:
                # Test known protocol hints or custom syncs
                tested_syncs = context.get("sync_patterns", ["EB90", "1ACFFC1D"])
                for s_hex in tested_syncs:
                    s_res = search_sync_word(
                        bits,
                        sync_pattern=s_hex,
                        pattern_id=f"sync_{s_hex}",
                        max_hamming_fraction=self.config.get("sync_word", {}).get("max_hamming_fraction", 0.125),
                    )
                    sync_results.append(s_res)
            except Exception:
                sync_results = []

        # 6. Header Consistency
        header_results = []
        proto_hint = context.get("protocol_hint")
        if self.config.get("headers", {}).get("enabled", True) and proto_hint and repetition_res.candidate_period_bits > 0:
            try:
                frames = segment_by_period(bits, repetition_res.candidate_period_bits)
                if len(frames) >= 2 and "header_schema" in proto_hint:
                    header_results = validate_header_schema(frames, proto_hint["header_schema"])
            except Exception:
                header_results = []

        # 7. Length Consistency
        length_res = LengthConsistencyResult()
        if repetition_res.candidate_period_bits > 0 and proto_hint and "header_schema" in proto_hint:
            try:
                frames = segment_by_period(bits, repetition_res.candidate_period_bits)
                payload_len_spec = proto_hint["header_schema"].get("payload_length_bytes") or proto_hint["header_schema"].get("packet_length")
                if payload_len_spec:
                    length_res = validate_length_field(
                        frames,
                        offset_bits=payload_len_spec.get("offset_bits", 32),
                        length_bits=payload_len_spec.get("length_bits", 8),
                        unit=payload_len_spec.get("unit", "bytes"),
                        observed_frame_length_bits=repetition_res.candidate_period_bits,
                    )
            except Exception:
                length_res = LengthConsistencyResult(check_state=EvidenceCheckState.NOT_TESTED)

        # 8. Re-encoding Verification
        reencoding_res = ReencodingResult()
        received_channel_bits = context.get("received_channel_bits")
        if self.config.get("reencoding", {}).get("enabled", True) and received_channel_bits is not None:
            try:
                reencoding_res = reencode_and_compare(
                    bits,
                    family,
                    params,
                    received_channel_bits,
                    max_comparison_bits=self.config.get("reencoding", {}).get("max_comparison_bits", 2048),
                )
            except Exception:
                reencoding_res = ReencodingResult(check_state=EvidenceCheckState.NOT_TESTED)

        # 9. Generic Structure
        struct_res = extract_structural_evidence(bits)

        # 10. Evidence Grouping & Contradictions
        (
            raw_count,
            indep_count,
            strong_count,
            contradictions,
            group_scores,
        ) = evaluate_evidence_groups(
            crc_results=crc_results,
            parity_results=parity_results,
            syndrome_result=syndrome_res,
            reencoding_result=reencoding_res,
            repetition_result=repetition_res,
            sync_results=sync_results,
            header_results=header_results,
            length_result=length_res,
            structural_result=struct_res,
        )

        # 11. Overall Score & Status
        overall_score = compute_overall_validation_score(
            group_scores,
            len(contradictions),
            self.config.get("scoring"),
        )

        status = determine_validation_status(
            overall_score=overall_score,
            strong_evidence_groups=strong_count,
            independent_evidence_groups=indep_count,
            contradiction_count=len(contradictions),
            mode=mode,
        )

        # Update processing history
        runtime_ms = (time.perf_counter() - t0) * 1000.0
        best_crc_pass = any(c.check_state == EvidenceCheckState.PASS for c in crc_results)
        history.append({
            "stage": "validation",
            "crc_pass": best_crc_pass,
            "frame_period_bits": repetition_res.candidate_period_bits,
            "validation_score": overall_score,
            "status": status.value,
            "runtime_ms": round(runtime_ms, 2),
        })

        result = ValidationResult(
            pipeline_path_id=path_id,
            candidate_id=cand_id,
            parent_candidate_id=parent_cand_id,
            demod_variant_id=demod_id,
            interleaver_candidate_id=inter_id,
            fec_candidate_id=fec_id,
            lineage=history,
            validation_status=status,
            overall_validation_score=overall_score,
            crc_results=crc_results,
            parity_results=parity_results,
            syndrome_result=syndrome_res,
            reencoding_result=reencoding_res,
            frame_repetition_result=repetition_res,
            sync_word_results=sync_results,
            header_results=header_results,
            length_result=length_res,
            structural_result=struct_res,
            evidence_count=indep_count,
            strong_evidence_count=strong_count,
            contradiction_count=len(contradictions),
            contradiction_details=contradictions,
            processing_history=history,
            decoded_bits_hash=bits_hash,
        )

        result.validation_features = extract_stage11_features(result)
        return result

    def validate_batch(
        self,
        fec_candidates: List[Any],
        context: Optional[Dict[str, Any]] = None,
        prune_to_beam: bool = True,
    ) -> List[ValidationResult]:
        """
        Validate multiple FEC candidates and rank them by validation strength.
        """
        results = [self.validate(cand, context) for cand in fec_candidates]
        # Sort descending by overall validation score
        results.sort(key=lambda r: r.overall_validation_score, reverse=True)

        if prune_to_beam:
            beam_w = self.config.get("pruning", {}).get("beam_width", 5)
            near_tie = self.config.get("pruning", {}).get("near_tie_threshold", 0.05)
            if len(results) > beam_w:
                cutoff = results[beam_w - 1].overall_validation_score - near_tie
                surviving = [r for r in results if r.overall_validation_score >= cutoff]
                return surviving[:beam_w * 2]

        return results

    def export_stage11_dataset(
        self,
        validation_results: List[ValidationResult],
        upstream_features_list: Optional[List[Dict[str, Any]]] = None,
        candidate_correct_labels: Optional[List[int]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Build complete dataset rows combining Stage 5-10 features for Stage 11 XGBoost training.
        """
        dataset = []
        for i, val_res in enumerate(validation_results):
            row = {}
            # Upstream features
            if upstream_features_list and i < len(upstream_features_list):
                row.update(upstream_features_list[i])

            # Validation features
            row.update(val_res.validation_features)

            # Lineage identifiers
            row["pipeline_path_id"] = val_res.pipeline_path_id
            row["candidate_id"] = val_res.candidate_id
            row["validation_status"] = val_res.validation_status.value

            # Training label if provided
            if candidate_correct_labels and i < len(candidate_correct_labels):
                row["candidate_correct"] = int(candidate_correct_labels[i])

            dataset.append(row)

        return dataset
