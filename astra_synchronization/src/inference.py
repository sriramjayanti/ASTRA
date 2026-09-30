"""
inference.py
Main SynchronizationEngine API for ASTRA Stage 6.
Provides robust adapter accepting both ReceiverHypothesis objects and direct keyword arguments.
"""

import os
import yaml
import logging
from typing import Dict, Any, List, Optional, Union
import numpy as np

from .models import SynchronizationResult, SyncStatus
from .validators import validate_iq_input
from .router import SynchronizationRouter

logger = logging.getLogger("ASTRA.SynchronizationEngine")


class SynchronizationEngine:
    """
    ASTRA Stage 6 Synchronization Engine.
    Executes family-aware carrier recovery, pulse matched filtering, and timing recovery
    to convert raw IQ into a symbol-aligned, phase-corrected constellation.
    """

    def __init__(self, config: Optional[Union[str, Dict[str, Any]]] = None):
        self.config = self._load_config(config)
        self.sync_cfg = self.config.get("synchronization", {})

    def _load_config(self, config: Optional[Union[str, Dict[str, Any]]]) -> Dict[str, Any]:
        if isinstance(config, dict):
            return config
        elif isinstance(config, str) and os.path.exists(config):
            with open(config, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}

        default_path = os.path.join(
            os.path.dirname(__file__), "..", "configs", "synchronization_config.yaml"
        )
        if os.path.exists(default_path):
            with open(default_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {}

    def synchronize(
        self,
        iq: np.ndarray,
        hypothesis: Any = None,
        sample_rate_hz: Optional[float] = None,
        **kwargs
    ) -> SynchronizationResult:
        """
        Synchronize an input IQ waveform against a specific receiver candidate hypothesis
        or direct keyword parameters.
        
        Args:
            iq: Raw complex IQ NumPy array
            hypothesis: ReceiverHypothesis dataclass, dictionary, or None (if passing kwargs)
            sample_rate_hz: Optional sampling rate in Hz
            kwargs: Direct parameters (sample_rate, nominal_baud, symbol_rate_hz, modulation, etc.)
        """
        # 1. Extract hypothesis parameters from hypothesis object, dict, or kwargs
        if hypothesis is not None:
            if isinstance(hypothesis, dict):
                cand_id = hypothesis.get("candidate_id", kwargs.get("candidate_id", "cand_unknown"))
                mod = hypothesis.get("modulation", kwargs.get("modulation", "QPSK"))
                fam = hypothesis.get("modulation_family", kwargs.get("modulation_family", "PSK"))
                sym_rate = float(hypothesis.get("symbol_rate_hz", kwargs.get("nominal_baud", kwargs.get("symbol_rate_hz", 9600.0))))
                fs = float(sample_rate_hz or hypothesis.get("sample_rate_hz", kwargs.get("sample_rate", 192000.0)))
            else:
                cand_id = getattr(hypothesis, "candidate_id", "cand_unknown")
                mod = getattr(hypothesis, "modulation", "QPSK")
                fam = getattr(hypothesis, "modulation_family", "PSK")
                sym_rate = float(getattr(hypothesis, "symbol_rate_hz", 9600.0))
                fs = float(sample_rate_hz or getattr(hypothesis, "sample_rate_hz", 192000.0))
        else:
            cand_id = kwargs.get("candidate_id", "cand_direct")
            mod = kwargs.get("modulation", "QPSK")
            fam = kwargs.get("modulation_family", "PSK")
            sym_rate = float(kwargs.get("symbol_rate_hz", kwargs.get("nominal_baud", kwargs.get("symbol_rate", 9600.0))))
            fs = float(sample_rate_hz or kwargs.get("sample_rate_hz", kwargs.get("sample_rate", kwargs.get("fs", 192000.0))))

        mod_upper = mod.upper().replace("-", "")
        if "FSK" in mod_upper or "MSK" in mod_upper:
            fam = "FSK"
        elif "QAM" in mod_upper:
            fam = "QAM"
        elif any(k in mod_upper for k in ["BPSK", "QPSK", "8PSK", "DQPSK"]):
            fam = "PSK"

        logger.info(f"Starting synchronization for candidate {cand_id} ({mod} @ {sym_rate:g} Bd)")

        # 2. Input Validation
        is_valid, val_err = validate_iq_input(
            iq=iq,
            sample_rate_hz=fs,
            symbol_rate_hz=sym_rate
        )
        if not is_valid:
            logger.error(f"Input validation failed for candidate {cand_id}: {val_err}")
            res = SynchronizationResult(
                candidate_id=cand_id,
                success=False,
                status=SyncStatus.SYNC_FAILED.value,
                modulation=mod,
                modulation_family=fam,
                input_sample_rate_hz=fs,
                symbol_rate_hz=sym_rate,
                failure_reason=val_err,
                lock_metrics={"overall_sync_score": 0.0}
            )
            self._update_candidate_object(hypothesis, res)
            return res

        # 3. Family-Aware Pipeline Execution via Router
        result = SynchronizationRouter.route_and_execute(
            iq_raw=iq,
            candidate_id=cand_id,
            modulation=mod,
            modulation_family=fam,
            symbol_rate_hz=sym_rate,
            sample_rate_hz=fs,
            config=self.sync_cfg
        )

        logger.info(
            f"Synchronization finished for {cand_id}: Status={result.status}, "
            f"Score={result.lock_metrics.get('overall_sync_score', 0):.4f}, "
            f"CFO={result.estimated_cfo_hz:+.1f} Hz, Symbols={result.symbol_count}"
        )

        self._update_candidate_object(hypothesis, result)
        return result

    def synchronize_candidates(
        self,
        iq: np.ndarray,
        candidates: List[Any],
        sample_rate_hz: Optional[float] = None
    ) -> List[SynchronizationResult]:
        """Synchronize multiple candidate hypotheses against the same IQ capture."""
        results = []
        for cand in candidates:
            res = self.synchronize(iq=iq, hypothesis=cand, sample_rate_hz=sample_rate_hz)
            results.append(res)
        return results

    def _update_candidate_object(self, hypothesis: Any, result: SynchronizationResult):
        """Update candidate hypothesis lifecycle and history."""
        if hasattr(hypothesis, "sync_result"):
            hypothesis.sync_result = result.to_dict(include_arrays=False)
            if hasattr(hypothesis, "add_history_entry"):
                hypothesis.add_history_entry(
                    stage="synchronization",
                    status=result.status,
                    details={
                        "score": result.lock_metrics.get("overall_sync_score", 0.0),
                        "cfo_hz": result.estimated_cfo_hz,
                        "symbols": result.symbol_count
                    }
                )
