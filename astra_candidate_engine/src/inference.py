"""
inference.py
Main CandidateHypothesisEngine class for ASTRA Stage 5.
"""

import os
import yaml
import logging
from typing import Dict, Any, Optional, List, Union

from .models import ReceiverHypothesis, CandidateSet, CandidateStatus
from .validators import parse_modulation_input, parse_symbol_rate_input
from .grid_generator import generate_candidate_grid
from .pruning import merge_duplicate_rates, prune_and_rank_candidates
from .explainability import explain_candidate, format_candidate_summary_table

logger = logging.getLogger("ASTRA.CandidateEngine")


class CandidateHypothesisEngine:
    """
    ASTRA Stage 5 Candidate / Hypothesis Engine.
    Generates a bounded, ranked, and beam-pruned set of receiver hypotheses from
    uncertain Modulation Fusion, Symbol-Rate Estimation, and optional RF/Constellation evidence.
    """

    def __init__(self, config: Optional[Union[str, Dict[str, Any]]] = None):
        self.config = self._load_config(config)
        self.engine_cfg = self.config.get("candidate_engine", {})
        
        self.mod_top_k = int(self.engine_cfg.get("modulation_top_k", 3))
        self.rate_top_k = int(self.engine_cfg.get("symbol_rate_top_k", 3))
        self.max_candidates = int(self.engine_cfg.get("max_candidates", 9))
        self.beam_width = int(self.engine_cfg.get("beam_width", 5))
        self.rate_merge_tol = float(self.engine_cfg.get("rate_merge_tolerance_percent", 2.0))
        self.min_sps = float(self.engine_cfg.get("min_sps", 1.0))
        self.max_sps = float(self.engine_cfg.get("max_sps", 2000.0))

    def _load_config(self, config: Optional[Union[str, Dict[str, Any]]]) -> Dict[str, Any]:
        if isinstance(config, dict):
            return config
        elif isinstance(config, str) and os.path.exists(config):
            with open(config, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        
        # Default fallback config
        default_path = os.path.join(
            os.path.dirname(__file__), "..", "configs", "candidate_config.yaml"
        )
        if os.path.exists(default_path):
            with open(default_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {}

    def generate(
        self,
        modulation_prediction: Any,
        symbol_rate_prediction: Any,
        sample_rate_hz: float,
        signal_id: str = "sig_001",
        rf_support: Optional[Any] = None,
        constellation_evidence: Optional[Any] = None,
        constellation_stage: str = "pre_sync"
    ) -> CandidateSet:
        """
        Generate candidate receiver hypotheses.
        
        Args:
            modulation_prediction: Fusion engine output (dict or list)
            symbol_rate_prediction: Symbol rate estimator output (dict or list)
            sample_rate_hz: Sampling frequency in Hz (e.g. 192000.0)
            signal_id: Identifier for signal (avoids copying full IQ buffer)
            rf_support: Optional Random Forest family probabilities dict
            constellation_evidence: Optional constellation support dict
            constellation_stage: 'pre_sync' or 'post_sync'
        """
        logger.info(f"Generating candidate hypotheses for signal {signal_id} (Fs={sample_rate_hz:g} Hz)")

        # 1. Parse & Validate Modulation Input
        mod_candidates = parse_modulation_input(modulation_prediction, top_k_limit=self.mod_top_k)
        if not mod_candidates:
            logger.warning("No valid modulation candidates provided. Using UNKNOWN placeholder.")
            mod_candidates = [{"modulation": "UNKNOWN", "probability": 0.1}]

        # 2. Parse & Validate Symbol Rate Input
        rate_source_tag = "symbol_rate_estimator"
        rate_candidates = parse_symbol_rate_input(symbol_rate_prediction, top_k_limit=self.rate_top_k)
        
        if not rate_candidates:
            # Check fallback mode
            fallback_cfg = self.engine_cfg.get("fallback", {})
            if fallback_cfg.get("enabled", True):
                std_rates = fallback_cfg.get("standard_rates_hz", [1200, 2400, 4800, 9600, 19200, 38400])
                fallback_score = float(fallback_cfg.get("default_fallback_score", 0.10))
                rate_source_tag = "fallback_standard_grid"
                rate_candidates = [
                    {"symbol_rate_hz": float(r), "score": fallback_score} for r in std_rates[:self.rate_top_k]
                ]
                logger.warning(f"Symbol rate estimation missing. Activated standard rate fallback ({len(rate_candidates)} rates).")
            else:
                logger.error("No valid symbol rates found and fallback is disabled.")

        # 3. Deduplicate / Merge Close Symbol Rates
        deduped_rates = merge_duplicate_rates(rate_candidates, tolerance_percent=self.rate_merge_tol)
        logger.debug(f"Merged {len(rate_candidates)} rates down to {len(deduped_rates)} unique rates.")

        # 4. Extract Optional RF and Constellation Support Probs
        rf_fam_probs = None
        if isinstance(rf_support, dict):
            rf_fam_probs = rf_support.get("family_probabilities", rf_support.get("probabilities", rf_support))
            
        const_support = None
        if isinstance(constellation_evidence, dict):
            const_support = constellation_evidence.get("modulation_support", constellation_evidence.get("support", constellation_evidence))

        # 5. Generate Cartesian Grid (Modulation × Baud)
        raw_hypotheses = generate_candidate_grid(
            modulation_candidates=mod_candidates,
            symbol_rate_candidates=deduped_rates,
            sample_rate_hz=sample_rate_hz,
            signal_id=signal_id,
            rf_family_probs=rf_fam_probs,
            constellation_support=const_support,
            constellation_stage=constellation_stage,
            config=self.engine_cfg,
            rate_source_tag=rate_source_tag
        )

        # 6. Apply Hard Physical Validity Checks and Soft Beam Pruning
        all_candidates, beam_survivors, rejected_list = prune_and_rank_candidates(
            candidates=raw_hypotheses,
            beam_width=self.beam_width,
            max_candidates=self.max_candidates,
            min_sps=self.min_sps,
            max_sps=self.max_sps
        )

        valid_count = len(all_candidates) - len(rejected_list)
        pruned_count = sum(1 for c in all_candidates if c.pruned_by_beam)

        meta = {
            "signal_id": signal_id,
            "sample_rate_hz": sample_rate_hz,
            "modulation_top_k_input": len(mod_candidates),
            "symbol_rate_top_k_input": len(rate_candidates),
            "merged_rate_count": len(deduped_rates),
            "raw_grid_size": len(raw_hypotheses),
            "beam_width": self.beam_width,
            "max_candidates": self.max_candidates,
            "rf_support_used": rf_fam_probs is not None,
            "constellation_support_used": const_support is not None,
            "rate_source": rate_source_tag,
        }

        candidate_set = CandidateSet(
            all_generated_count=len(raw_hypotheses),
            valid_candidate_count=valid_count,
            pruned_candidate_count=pruned_count,
            rejected_candidate_count=len(rejected_list),
            candidates=all_candidates,
            top_candidates=all_candidates[:min(3, len(all_candidates))],
            beam_candidates=beam_survivors,
            generation_metadata=meta
        )

        logger.info(
            f"Generated {len(raw_hypotheses)} raw candidates -> {len(beam_survivors)} in beam "
            f"(top-1: {beam_survivors[0].modulation if beam_survivors else 'None'} "
            f"@ {beam_survivors[0].symbol_rate_hz if beam_survivors else 0:g} Bd)"
        )

        return candidate_set

    def rank_candidates(self, candidates: List[ReceiverHypothesis]) -> List[ReceiverHypothesis]:
        """Sort candidates descending by initial score and update ranks."""
        candidates.sort(key=lambda x: x.initial_score, reverse=True)
        for idx, c in enumerate(candidates, start=1):
            c.candidate_rank = idx
        return candidates

    def prune_candidates(
        self,
        candidates: List[ReceiverHypothesis],
        beam_width: Optional[int] = None
    ) -> List[ReceiverHypothesis]:
        """Prune candidates to top beam_width."""
        bw = beam_width or self.beam_width
        survivors = []
        for rank_idx, c in enumerate(candidates, start=1):
            if rank_idx <= bw:
                c.pruned_by_beam = False
                survivors.append(c)
            else:
                c.pruned_by_beam = True
        return survivors

    def explain(self, candidate: ReceiverHypothesis) -> Dict[str, Any]:
        """Explain an individual candidate."""
        return explain_candidate(candidate)

    def print_summary(self, candidate_set: CandidateSet) -> str:
        """Return formatted summary table for candidate set."""
        return format_candidate_summary_table(candidate_set)
