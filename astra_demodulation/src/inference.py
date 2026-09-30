"""
inference.py
Main DemodulationEngine class for ASTRA Stage 7.
"""

import os
import yaml
import logging
from typing import Dict, Any, List, Optional, Union
import numpy as np

from .models import DemodulationResult, DemodStatus
from .validators import validate_demod_input
from .router import DemodulationRouter

logger = logging.getLogger("ASTRA.DemodulationEngine")


class DemodulationEngine:
    """
    ASTRA Stage 7 Demodulation Engine.
    Converts synchronized complex symbol streams into hard bits, soft LLRs,
    and rotational phase-ambiguity candidate bitstreams.
    """

    def __init__(self, config: Optional[Union[str, Dict[str, Any]]] = None):
        self.config = self._load_config(config)
        self.demod_cfg = self.config.get("demodulation", {})

    def _load_config(self, config: Optional[Union[str, Dict[str, Any]]]) -> Dict[str, Any]:
        if isinstance(config, dict):
            return config
        elif isinstance(config, str) and os.path.exists(config):
            with open(config, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}

        default_path = os.path.join(
            os.path.dirname(__file__), "..", "configs", "demodulation_config.yaml"
        )
        if os.path.exists(default_path):
            with open(default_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {}

    def demodulate(
        self,
        sync_result: Any,
        hypothesis: Optional[Any] = None,
        symbols: Optional[np.ndarray] = None
    ) -> DemodulationResult:
        """
        Demodulate synchronized symbols from Stage 6.
        
        Args:
            sync_result: SynchronizationResult object or dictionary
            hypothesis: Optional ReceiverHypothesis from Stage 5
            symbols: Optional direct complex symbol array override
        """
        # 1. Extract parameters from sync_result / hypothesis
        if isinstance(sync_result, dict):
            cand_id = sync_result.get("candidate_id", "cand_unknown")
            parent_cand_id = sync_result.get("parent_candidate_id", cand_id)
            mod = sync_result.get("modulation", "QPSK")
            fam = sync_result.get("modulation_family", "PSK")
            fs = float(sync_result.get("input_sample_rate_hz", sync_result.get("sample_rate_hz", 192000.0)))
            rs = float(sync_result.get("symbol_rate_hz", sync_result.get("baud", 9600.0)))
            syms = symbols if symbols is not None else sync_result.get("symbol_samples")
            noise_var = sync_result.get("noise_variance", 0.05)
            lock_metrics = sync_result.get("lock_metrics", {})
            sync_score = float(lock_metrics.get("overall_sync_score", sync_result.get("sync_score", 0.0)))
            timing_score = float(lock_metrics.get("timing_score", sync_result.get("timing_score", 0.0)))
            carrier_score = float(lock_metrics.get("carrier_score", sync_result.get("carrier_score", 0.0)))
            residual_cfo = float(lock_metrics.get("residual_cfo_hz", sync_result.get("residual_cfo", 0.0)))
            sync_lineage = sync_result.get("lineage", sync_result.get("processing_history", []))
            sync_meta = sync_result.get("metadata", {})
        else:
            cand_id = getattr(sync_result, "candidate_id", "cand_unknown")
            parent_cand_id = getattr(sync_result, "parent_candidate_id", cand_id)
            mod = getattr(sync_result, "modulation", "QPSK")
            fam = getattr(sync_result, "modulation_family", "PSK")
            fs = float(getattr(sync_result, "input_sample_rate_hz", getattr(sync_result, "sample_rate_hz", 192000.0)))
            rs = float(getattr(sync_result, "symbol_rate_hz", getattr(sync_result, "baud", 9600.0)))
            syms = symbols if symbols is not None else getattr(sync_result, "symbol_samples", None)
            noise_var = getattr(sync_result, "noise_variance", 0.05)
            lock_metrics = getattr(sync_result, "lock_metrics", {}) or {}
            sync_score = float(lock_metrics.get("overall_sync_score", getattr(sync_result, "sync_score", 0.0)))
            timing_score = float(lock_metrics.get("timing_score", getattr(sync_result, "timing_score", 0.0)))
            carrier_score = float(lock_metrics.get("carrier_score", getattr(sync_result, "carrier_score", 0.0)))
            residual_cfo = float(lock_metrics.get("residual_cfo_hz", getattr(sync_result, "residual_cfo", 0.0)))
            sync_lineage = getattr(sync_result, "lineage", getattr(sync_result, "processing_history", []))
            sync_meta = getattr(sync_result, "metadata", {}) or {}

        if "FSK" in mod.upper() or "MSK" in mod.upper():
            fam = "FSK"
        elif "QAM" in mod.upper():
            fam = "QAM"
        elif any(k in mod.upper() for k in ["BPSK", "QPSK", "8PSK", "DQPSK"]):
            fam = "PSK"

        # 2. Input Validation
        is_valid, err = validate_demod_input(syms, modulation=mod, modulation_family=fam)
        if not is_valid:
            logger.error(f"Demodulation validation failed for {cand_id}: {err}")
            res = DemodulationResult(
                candidate_id=cand_id,
                parent_candidate_id=parent_cand_id,
                modulation=mod,
                modulation_family=fam,
                baud=rs,
                sample_rate=fs,
                sync_score=sync_score,
                timing_score=timing_score,
                carrier_score=carrier_score,
                residual_cfo=residual_cfo,
                lineage=list(sync_lineage),
                metadata=dict(sync_meta),
                success=False,
                status=DemodStatus.DEMOD_FAILED.value,
                failure_reason=err
            )
            self._update_candidate_object(hypothesis, res)
            return res

        # 3. Family Demodulation Execution
        result = DemodulationRouter.route_and_demodulate(
            symbols=syms,
            candidate_id=cand_id,
            modulation=mod,
            modulation_family=fam,
            sample_rate_hz=fs,
            symbol_rate_hz=rs,
            sync_noise_var=noise_var,
            config=self.demod_cfg
        )

        # Standardize canonical metadata
        result.parent_candidate_id = parent_cand_id
        result.baud = rs
        result.sample_rate = fs
        result.sync_score = sync_score
        result.timing_score = timing_score
        result.carrier_score = carrier_score
        result.residual_cfo = residual_cfo
        result.evm = result.quality.evm_percent if result.quality else 0.0
        result.demod_quality = result.quality.demodulation_quality_score if result.quality else 0.0
        if result.soft_llrs is not None and len(result.soft_llrs) > 0:
            result.llr_confidence = float(np.mean(np.abs(result.soft_llrs)))
        result.lineage = list(sync_lineage) + [{"stage": "demodulation", "status": result.status, "evm": result.evm}]
        result.metadata = dict(sync_meta)

        # Propagate to all variants
        for var in result.phase_variants:
            var.parent_candidate_id = parent_cand_id
            var.modulation = mod
            var.baud = rs
            var.sample_rate = fs
            var.sync_score = sync_score
            var.timing_score = timing_score
            var.carrier_score = carrier_score
            var.residual_cfo = residual_cfo
            var.evm = var.quality.evm_percent if var.quality else result.evm
            var.demod_quality = var.quality.demodulation_quality_score if var.quality else result.demod_quality
            if var.soft_llrs is not None and len(var.soft_llrs) > 0:
                var.llr_confidence = float(np.mean(np.abs(var.soft_llrs)))
            else:
                var.llr_confidence = result.llr_confidence
            var.lineage = list(result.lineage) + [{"stage": "phase_ambiguity", "variant_id": var.candidate_id}]
            var.metadata = dict(sync_meta)

        logger.info(
            f"Demodulated {cand_id} ({mod}): Bits={result.bit_count}, "
            f"EVM={result.quality.evm_percent:.1f}%, Status={result.status}, "
            f"Ambiguity Variants={len(result.phase_variants)}"
        )

        # 4. Attach result to Candidate hypothesis if passed
        self._update_candidate_object(hypothesis, result)
        return result

    def demodulate_batch(
        self,
        sync_results: List[Any],
        hypotheses: Optional[List[Any]] = None
    ) -> List[DemodulationResult]:
        """Demodulate a batch of synchronized candidate outputs."""
        results = []
        for idx, sync_res in enumerate(sync_results):
            hyp = hypotheses[idx] if (hypotheses and idx < len(hypotheses)) else None
            res = self.demodulate(sync_result=sync_res, hypothesis=hyp)
            results.append(res)
        return results

    def _update_candidate_object(self, hypothesis: Any, result: DemodulationResult):
        """Update candidate hypothesis lifecycle and history."""
        if hasattr(hypothesis, "demod_result"):
            hypothesis.demod_result = result.to_dict(include_arrays=False)
            if hasattr(hypothesis, "add_history_entry"):
                hypothesis.add_history_entry(
                    stage="demodulation",
                    status=result.status,
                    details={
                        "bit_count": result.bit_count,
                        "evm_percent": result.quality.evm_percent,
                        "quality_score": result.quality.demodulation_quality_score,
                        "variants_count": len(result.phase_variants)
                    }
                )
