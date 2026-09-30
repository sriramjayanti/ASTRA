"""
inference.py
Main entry point for ASTRA Stage 9 — FEC Candidate Testing Engine.
Orchestrates candidate generation, multi-family decoding, metric normalization, scoring, and beam pruning.
"""

from typing import Dict, Any, List, Optional, Union
import time
import os
import yaml
import numpy as np

from .models import (
    FECStatus,
    FECFamily,
    FECProfile,
    DecoderResult,
    FECCandidateResult,
    FECTestResult
)
from .profiles import ProfileRegistry, get_profile_registry
from .candidate_generator import FECCandidateGenerator, FECHypothesis
from .router import execute_decoder
from .metrics import normalize_decoder_metrics
from .scoring import FECScorer
from .pruning import prune_fec_candidates
from .validators import validate_fec_config, validate_interleaver_input


class FECTestingEngine:
    """
    Production-grade FEC Candidate Testing Engine (ASTRA Stage 9).
    Receives deinterleaved hard bits + soft LLRs from Stage 8 and outputs ranked decoded candidate bitstreams for Stage 10 Validation.
    """
    def __init__(
        self,
        config_path: Optional[str] = None,
        config_dict: Optional[Dict[str, Any]] = None,
        profiles_path: Optional[str] = None
    ):
        if config_dict is not None:
            self.config = config_dict
        elif config_path is not None and os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f)
        else:
            default_path = os.path.join(
                os.path.dirname(__file__), "..", "configs", "fec_config.yaml"
            )
            if os.path.exists(default_path):
                with open(default_path, "r", encoding="utf-8") as f:
                    self.config = yaml.safe_load(f)
            else:
                self.config = {}

        is_valid, errors = validate_fec_config(self.config)
        if not is_valid:
            raise ValueError(f"Invalid FEC configuration: {', '.join(errors)}")

        self.registry = ProfileRegistry(profiles_path)
        self.generator = FECCandidateGenerator(self.config)
        self.scorer = FECScorer(self.config.get("fec", self.config))
        
        search_cfg = self.config.get("fec", {}).get("search", {})
        self.beam_width = int(search_cfg.get("beam_width", 8))
        self.prefer_soft = bool(self.config.get("fec", {}).get("viterbi", {}).get("prefer_soft", True))

    def generate_candidates(self, bit_count: int) -> List[FECHypothesis]:
        """Generates candidate FEC hypotheses for a given bitstream length."""
        return self.generator.generate(bit_count=bit_count)

    def decode_candidate(
        self,
        hypothesis: FECHypothesis,
        hard_bits: np.ndarray,
        soft_llrs: Optional[np.ndarray] = None,
        candidate_id: str = "cand0",
        variant_id: str = "var0",
        interleaver_id: str = "int_none_0001",
        parent_candidate_id: str = "",
        incoming_lineage: Optional[List[Dict[str, Any]]] = None
    ) -> FECCandidateResult:
        """
        Applies a single candidate FEC decoder, normalizes telemetry, and scores the result.
        """
        path_id = f"path_{candidate_id}_{variant_id}_{interleaver_id}_{hypothesis.candidate_id}"
        
        # Apply bit offset if tested
        off = hypothesis.bit_offset
        if off > 0:
            rx_hard = hard_bits[off:]
            rx_soft = soft_llrs[off:] if soft_llrs is not None else None
        else:
            rx_hard = hard_bits
            rx_soft = soft_llrs
            
        try:
            dec_res = execute_decoder(
                profile=hypothesis.profile,
                hard_bits=rx_hard,
                soft_llrs=rx_soft,
                prefer_soft=self.prefer_soft
            )
            
            norm_metrics = normalize_decoder_metrics(dec_res, hypothesis.family)
            fec_quality, structural_sc, status = self.scorer.score_candidate(
                decoder_result=dec_res,
                norm_metrics=norm_metrics,
                family=hypothesis.family
            )
            
            profile_id = hypothesis.profile.profile_id if hypothesis.profile else (hypothesis.parameters.get("profile", "none"))
            errors_corr = int(norm_metrics.get("errors_corrected", 0))
            syn_weight = float(norm_metrics.get("syndrome_score", 0.0))
            path_met = float(norm_metrics.get("normalized_path_metric", 0.0))
            parity_ok = bool(norm_metrics.get("parity_success", False))
            conv_status = "CONVERGED" if dec_res.success else "FAILED"
            
            lineage = list(incoming_lineage or []) + [{
                "stage": "fec",
                "family": hypothesis.family,
                "decoder_type": hypothesis.family,
                "profile": profile_id,
                "decoder": dec_res.decoder_name,
                "fec_quality_score": round(fec_quality, 4),
                "errors_corrected": errors_corr,
                "status": status.value
            }]

            return FECCandidateResult(
                candidate_id=candidate_id,
                parent_candidate_id=parent_candidate_id or candidate_id,
                demod_variant_id=variant_id,
                interleaver_candidate_id=interleaver_id,
                fec_candidate_id=hypothesis.candidate_id,
                path_id=path_id,
                fec_family=hypothesis.family,
                decoder_type=hypothesis.family,
                profile_id=profile_id,
                fec_parameters=hypothesis.parameters,
                convergence_status=conv_status,
                decoder_score=fec_quality,
                corrected_error_info={"errors_corrected": errors_corr},
                syndrome_metrics={"syndrome_weight": syn_weight, "parity_check_success": parity_ok},
                path_metrics={"path_metric": path_met, "iterations": int(dec_res.metrics.get("iterations_used", 0))},
                lineage=lineage,
                input_bit_count=len(rx_hard),
                output_bit_count=len(dec_res.decoded_bits),
                decoded_hard_bits=dec_res.decoded_bits,
                decoded_soft_information_if_available=dec_res.decoded_soft_info,
                decoder_success=dec_res.success,
                decoder_status=status,
                decoder_metrics=norm_metrics,
                code_rate=float(norm_metrics.get("code_rate", 1.0)),
                estimated_errors_corrected=errors_corr,
                iterations=int(dec_res.metrics.get("iterations_used", 0)),
                syndrome_weight=syn_weight,
                path_metric=path_met,
                parity_check_success=parity_ok,
                structural_score=structural_sc,
                fec_quality_score=fec_quality,
                processing_history=lineage,
                rejected=not dec_res.success,
                rejection_reason=dec_res.failure_reason
            )
            
        except Exception as e:
            return FECCandidateResult(
                candidate_id=candidate_id,
                parent_candidate_id=parent_candidate_id or candidate_id,
                demod_variant_id=variant_id,
                interleaver_candidate_id=interleaver_id,
                fec_candidate_id=hypothesis.candidate_id,
                path_id=path_id,
                fec_family=hypothesis.family,
                decoder_type=hypothesis.family,
                profile_id=hypothesis.parameters.get("profile", "none"),
                fec_parameters=hypothesis.parameters,
                convergence_status="FAILED",
                decoder_score=0.0,
                corrected_error_info={"error": str(e)},
                syndrome_metrics={},
                path_metrics={},
                lineage=list(incoming_lineage or []) + [{"stage": "fec", "error": str(e)}],
                input_bit_count=len(hard_bits),
                output_bit_count=0,
                decoded_hard_bits=np.array([], dtype=np.uint8),
                decoder_success=False,
                decoder_status=FECStatus.FEC_FAILED,
                rejected=True,
                rejection_reason=str(e)
            )

    def test_candidates(
        self,
        interleaver_candidate: Any,
        context: Optional[Dict[str, Any]] = None
    ) -> FECTestResult:
        """
        Evaluates all candidate FEC profiles against a deinterleaved bitstream variant.
        
        Args:
            interleaver_candidate: InterleaverCandidateResult or dict containing:
                                   'candidate_id', 'demod_variant_id', 'interleaver_candidate_id',
                                   'deinterleaved_hard_bits' (or 'hard_bits'), 'deinterleaved_soft_llrs' (or 'soft_llrs')
            context: Optional contextual parameters
            
        Returns:
            FECTestResult
        """
        start_time = time.perf_counter()
        
        # Extract inputs from dataclass or dict
        if hasattr(interleaver_candidate, "candidate_id"):
            cand_id = getattr(interleaver_candidate, "candidate_id")
            parent_cand_id = getattr(interleaver_candidate, "parent_candidate_id", cand_id)
            var_id = getattr(interleaver_candidate, "demod_variant_id", "rot0")
            int_id = getattr(interleaver_candidate, "interleaver_candidate_id", "int_none_0001")
            incoming_lineage = getattr(interleaver_candidate, "lineage", getattr(interleaver_candidate, "processing_history", []))
            hard_bits = getattr(interleaver_candidate, "deinterleaved_hard_bits", None)
            if hard_bits is None:
                hard_bits = getattr(interleaver_candidate, "hard_bits", None)
            soft_llrs = getattr(interleaver_candidate, "deinterleaved_soft_llrs", None)
            if soft_llrs is None:
                soft_llrs = getattr(interleaver_candidate, "soft_llrs", None)
        elif isinstance(interleaver_candidate, dict):
            cand_id = interleaver_candidate.get("candidate_id", "cand_001")
            parent_cand_id = interleaver_candidate.get("parent_candidate_id", cand_id)
            var_id = interleaver_candidate.get("demod_variant_id", "rot0")
            int_id = interleaver_candidate.get("interleaver_candidate_id", "int_none_0001")
            incoming_lineage = interleaver_candidate.get("lineage", interleaver_candidate.get("processing_history", []))
            hard_bits = interleaver_candidate.get("deinterleaved_hard_bits", interleaver_candidate.get("hard_bits"))
            soft_llrs = interleaver_candidate.get("deinterleaved_soft_llrs", interleaver_candidate.get("soft_llrs"))
        else:
            raise ValueError(f"Unsupported interleaver_candidate type: {type(interleaver_candidate)}")
            
        is_valid, msg, clean_hard, clean_soft = validate_interleaver_input(hard_bits, soft_llrs)
        if not is_valid:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return FECTestResult(
                candidate_id=cand_id,
                demod_variant_id=var_id,
                interleaver_candidate_id=int_id,
                tested_candidates_count=0,
                surviving_candidates=[],
                top_candidate=None,
                execution_time_ms=elapsed_ms,
                metadata={"error": msg}
            )
            
        N = len(clean_hard)
        hypotheses = self.generate_candidates(bit_count=N)
        
        tested_results: List[FECCandidateResult] = []
        for hyp in hypotheses:
            res = self.decode_candidate(
                hypothesis=hyp,
                hard_bits=clean_hard,
                soft_llrs=clean_soft,
                candidate_id=cand_id,
                variant_id=var_id,
                interleaver_id=int_id
            )
            tested_results.append(res)
            
        # Beam pruning
        survivors = prune_fec_candidates(
            candidates=tested_results,
            beam_width=self.beam_width,
            always_preserve_none=True,
            enforce_family_diversity=True
        )
        
        top_cand = survivors[0] if survivors else None
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        
        return FECTestResult(
            candidate_id=cand_id,
            demod_variant_id=var_id,
            interleaver_candidate_id=int_id,
            tested_candidates_count=len(hypotheses),
            surviving_candidates=survivors,
            top_candidate=top_cand,
            execution_time_ms=elapsed_ms,
            metadata={
                "input_bit_count": N,
                "beam_width": self.beam_width,
                "surviving_count": len(survivors)
            }
        )
