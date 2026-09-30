"""
inference.py
Main entry point for ASTRA Stage 8 — Interleaver Candidate Testing Engine.
Orchestrates candidate generation, deinterleaving execution, structural feature extraction, rule scoring, and beam pruning.
"""

from typing import Dict, Any, List, Optional, Union
import time
import os
import yaml
import numpy as np

from .models import (
    InterleaverStatus,
    InterleaverFamily,
    PermutationMapping,
    ConvolutionalDeinterleaverState,
    StructuralFeatures,
    InterleaverCandidateResult,
    InterleaverTestResult
)
from .candidate_generator import InterleaverCandidateGenerator, CandidateHypothesis
from .router import execute_deinterleaver
from .structural_features import extract_structural_features
from .scoring import CandidateScorer
from .pruning import prune_candidates
from .validators import validate_config, validate_demod_input


class InterleaverTestingEngine:
    """
    Production-grade Interleaver Candidate Testing Engine (ASTRA Stage 8).
    Processes demodulated bitstream variants from Stage 7 and produces ranked candidate deinterleaver hypotheses for Stage 9 FEC.
    """
    def __init__(
        self,
        config_path: Optional[str] = None,
        config_dict: Optional[Dict[str, Any]] = None
    ):
        if config_dict is not None:
            self.config = config_dict
        elif config_path is not None and os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f)
        else:
            default_path = os.path.join(
                os.path.dirname(__file__), "..", "configs", "interleaver_config.yaml"
            )
            if os.path.exists(default_path):
                with open(default_path, "r", encoding="utf-8") as f:
                    self.config = yaml.safe_load(f)
            else:
                self.config = {}

        is_valid, errors = validate_config(self.config)
        if not is_valid:
            raise ValueError(f"Invalid interleaver configuration: {', '.join(errors)}")

        self.generator = InterleaverCandidateGenerator(self.config)
        self.scorer = CandidateScorer(self.config.get("interleaver", self.config))
        
        search_cfg = self.config.get("interleaver", {}).get("search", {})
        self.beam_width = int(search_cfg.get("beam_width", 10))
        self.lazy_materialization = bool(search_cfg.get("lazy_materialization", False))

    def generate_candidates(self, bit_count: int, variant_id: str = "var0") -> List[CandidateHypothesis]:
        """Generates candidate deinterleaving hypotheses for a given bitstream length."""
        return self.generator.generate(bit_count=bit_count, variant_id=variant_id)

    def apply_candidate(
        self,
        hypothesis: CandidateHypothesis,
        hard_bits: np.ndarray,
        soft_llrs: Optional[np.ndarray] = None,
        candidate_id: str = "cand0",
        variant_id: str = "var0",
        parent_candidate_id: str = "",
        incoming_lineage: Optional[List[Dict[str, Any]]] = None,
        state: Optional[ConvolutionalDeinterleaverState] = None
    ) -> InterleaverCandidateResult:
        """
        Applies a single candidate deinterleaver hypothesis and extracts structural features.
        """
        try:
            deint_hard, deint_soft, mapping, new_state, rem_frac = execute_deinterleaver(
                family=hypothesis.family,
                parameters=hypothesis.parameters,
                hard_bits=hard_bits,
                soft_llrs=soft_llrs,
                state=state
            )
            
            # Extract structural features
            features = extract_structural_features(
                hard_bits=deint_hard,
                soft_llrs=deint_soft,
                remainder_fraction=rem_frac
            )
            
            # Score candidate
            overall_sc, period_sc, autocorr_sc, entropy_sc, run_sc, status = self.scorer.score_candidate(
                features=features,
                family=hypothesis.family,
                success=True
            )
            
            perm_hash = mapping.mapping_hash if mapping else (hypothesis.mapping_hash or "")
            evidence_dict = features.to_dict()
            lineage = list(incoming_lineage or []) + [{
                "stage": "interleaver",
                "family": hypothesis.family,
                "interleaver_type": hypothesis.family,
                "parameters": hypothesis.parameters,
                "structural_score": round(overall_sc, 4),
                "status": status.value
            }]

            result = InterleaverCandidateResult(
                candidate_id=candidate_id,
                parent_candidate_id=parent_candidate_id or candidate_id,
                demod_variant_id=variant_id,
                interleaver_candidate_id=hypothesis.candidate_id,
                interleaver_family=hypothesis.family,
                interleaver_type=hypothesis.family,
                parameters=hypothesis.parameters,
                score=overall_sc,
                evidence=evidence_dict,
                lineage=lineage,
                success=True,
                status=status,
                deinterleaved_hard_bits=deint_hard if not self.lazy_materialization else None,
                deinterleaved_soft_llrs=deint_soft if not self.lazy_materialization else None,
                structural_score=overall_sc,
                periodicity_score=period_sc,
                entropy_score=entropy_sc,
                autocorrelation_score=autocorr_sc,
                run_length_score=run_sc,
                candidate_prior=float(self.scorer.family_priors.get(hypothesis.family, 0.5)),
                overall_interleaver_score=overall_sc,
                structural_features=features,
                permutation_hash=perm_hash,
                processing_history=lineage,
                lazy_materialized=self.lazy_materialization,
                permutation_mapping=mapping,
                state=new_state
            )
            return result
            
        except Exception as e:
            return InterleaverCandidateResult(
                candidate_id=candidate_id,
                parent_candidate_id=parent_candidate_id or candidate_id,
                demod_variant_id=variant_id,
                interleaver_candidate_id=hypothesis.candidate_id,
                interleaver_family=hypothesis.family,
                interleaver_type=hypothesis.family,
                parameters=hypothesis.parameters,
                score=0.0,
                evidence={"error": str(e)},
                lineage=list(incoming_lineage or []) + [{"stage": "interleaver", "error": str(e)}],
                success=False,
                status=InterleaverStatus.INTERLEAVER_INVALID,
                rejected=True,
                rejection_reason=str(e)
            )

    def test_candidates(
        self,
        demodulation_variant: Any,
        context: Optional[Dict[str, Any]] = None
    ) -> InterleaverTestResult:
        """
        Evaluates all generated candidate hypotheses against a demodulation variant.
        
        Args:
            demodulation_variant: DemodulationResult / DemodulationVariant dataclass or dict with keys:
                                  'variant_id', 'hard_bits', 'soft_llrs', 'candidate_id'
            context: Optional contextual parameters (e.g. sync patterns, known frame lengths)
            
        Returns:
            InterleaverTestResult
        """
        start_time = time.perf_counter()
        
        # Extract inputs from DemodulationResult / DemodulationVariant or dict
        if hasattr(demodulation_variant, "variant_id") or hasattr(demodulation_variant, "hard_bits"):
            variant_id = getattr(demodulation_variant, "variant_id", getattr(demodulation_variant, "phase_variant", "var0"))
            hard_bits = getattr(demodulation_variant, "hard_bits", None)
            soft_llrs = getattr(demodulation_variant, "soft_llrs", None)
            cand_id = getattr(demodulation_variant, "candidate_id", "cand_demod_001")
            parent_cand_id = getattr(demodulation_variant, "parent_candidate_id", cand_id)
            incoming_lineage = getattr(demodulation_variant, "lineage", getattr(demodulation_variant, "processing_history", []))
        elif isinstance(demodulation_variant, dict):
            variant_id = demodulation_variant.get("variant_id", demodulation_variant.get("phase_variant", "var0"))
            hard_bits = demodulation_variant.get("hard_bits")
            soft_llrs = demodulation_variant.get("soft_llrs", None)
            cand_id = demodulation_variant.get("candidate_id", "cand_demod_001")
            parent_cand_id = demodulation_variant.get("parent_candidate_id", cand_id)
            incoming_lineage = demodulation_variant.get("lineage", demodulation_variant.get("processing_history", []))
        else:
            raise ValueError(f"Unsupported demodulation_variant format: {type(demodulation_variant)}")
            
        is_valid, msg, clean_hard, clean_soft = validate_demod_input(hard_bits, soft_llrs)
        if not is_valid:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            return InterleaverTestResult(
                candidate_id=cand_id,
                demod_variant_id=variant_id,
                tested_candidates_count=0,
                surviving_candidates=[],
                top_candidate=None,
                execution_time_ms=elapsed_ms,
                metadata={"error": msg}
            )
            
        N = len(clean_hard)
        hypotheses = self.generate_candidates(bit_count=N, variant_id=variant_id)
        
        tested_results: List[InterleaverCandidateResult] = []
        for hyp in hypotheses:
            res = self.apply_candidate(
                hypothesis=hyp,
                hard_bits=clean_hard,
                soft_llrs=clean_soft,
                candidate_id=cand_id,
                variant_id=variant_id,
                parent_candidate_id=parent_cand_id,
                incoming_lineage=incoming_lineage
            )
            tested_results.append(res)
            
        # Beam pruning
        survivors = prune_candidates(
            candidates=tested_results,
            beam_width=self.beam_width,
            always_preserve_identity=True
        )
        
        top_cand = survivors[0] if survivors else None
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        
        return InterleaverTestResult(
            candidate_id=cand_id,
            demod_variant_id=variant_id,
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
