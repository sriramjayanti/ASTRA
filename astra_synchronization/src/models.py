"""
models.py
Data models and state containers for ASTRA Stage 6 — Synchronization Engine.
"""

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Optional, Any
import numpy as np
import json


class SyncStatus(str, Enum):
    SYNC_PASSED = "SYNC_PASSED"
    SYNC_WEAK = "SYNC_WEAK"
    SYNC_FAILED = "SYNC_FAILED"


@dataclass
class CarrierState:
    """State tracking for carrier recovery loops (Costas / DD-PLL)."""
    phase_rad: float = 0.0
    freq_rad_per_sample: float = 0.0
    integrator: float = 0.0
    lock_counter: int = 0
    phase_error_history: List[float] = field(default_factory=list)


@dataclass
class TimingState:
    """State tracking for timing recovery loops (Gardner / M&M)."""
    mu: float = 0.0  # Fractional timing offset [0, 1)
    sps: float = 2.0  # Current effective samples per symbol
    timing_error_history: List[float] = field(default_factory=list)
    timing_phase_history: List[float] = field(default_factory=list)


@dataclass
class SyncState:
    """Composite state for windowed/streaming synchronization tracking."""
    carrier: CarrierState = field(default_factory=CarrierState)
    timing: TimingState = field(default_factory=TimingState)
    cfo_coarse_hz: float = 0.0
    total_processed_samples: int = 0


@dataclass
class SynchronizationResult:
    """
    Output of the Synchronization Engine containing synchronized IQ stream,
    recovered symbol-center samples, and multi-dimensional lock quality metrics.
    """
    candidate_id: str
    success: bool = False
    status: str = SyncStatus.SYNC_FAILED.value

    modulation: str = "Unknown"
    modulation_family: str = "UNKNOWN"

    input_sample_rate_hz: float = 0.0
    working_sample_rate_hz: float = 0.0
    symbol_rate_hz: float = 0.0

    # CFO and Phase Telemetry
    estimated_cfo_hz: float = 0.0
    residual_cfo_hz: float = 0.0
    estimated_phase_rad: float = 0.0
    phase_ambiguity_states: List[float] = field(default_factory=list)

    # Timing Telemetry
    timing_offset_samples: float = 0.0
    recovered_samples_per_symbol: float = 1.0
    symbol_count: int = 0

    # DSP Method Metadata
    matched_filter_type: str = "rrc"
    matched_filter_rolloff: float = 0.35
    timing_method: str = "gardner"
    carrier_method: str = "costas"

    # Multi-Dimensional Lock Metrics
    lock_metrics: Dict[str, float] = field(default_factory=dict)
    constellation_quality_before: float = 0.0
    constellation_quality_after: float = 0.0

    # Synchronized Outputs (NumPy arrays omitted from standard JSON, available directly)
    synchronized_iq: Optional[np.ndarray] = None
    symbol_samples: Optional[np.ndarray] = None

    processing_history: List[Dict[str, Any]] = field(default_factory=list)
    failure_reason: Optional[str] = None

    def add_history_entry(self, stage: str, status: str, details: Optional[Dict[str, Any]] = None):
        self.processing_history.append({
            "stage": stage,
            "status": status,
            "details": details or {}
        })

    def to_dict(self, include_arrays: bool = False) -> Dict[str, Any]:
        data = {
            "candidate_id": self.candidate_id,
            "success": self.success,
            "status": self.status,
            "modulation": self.modulation,
            "modulation_family": self.modulation_family,
            "input_sample_rate_hz": self.input_sample_rate_hz,
            "working_sample_rate_hz": self.working_sample_rate_hz,
            "symbol_rate_hz": self.symbol_rate_hz,
            "estimated_cfo_hz": round(self.estimated_cfo_hz, 2),
            "residual_cfo_hz": round(self.residual_cfo_hz, 2),
            "estimated_phase_rad": round(self.estimated_phase_rad, 4),
            "phase_ambiguity_states": [round(a, 4) for a in self.phase_ambiguity_states],
            "timing_offset_samples": round(self.timing_offset_samples, 3),
            "recovered_samples_per_symbol": round(self.recovered_samples_per_symbol, 3),
            "symbol_count": self.symbol_count,
            "matched_filter_type": self.matched_filter_type,
            "matched_filter_rolloff": self.matched_filter_rolloff,
            "timing_method": self.timing_method,
            "carrier_method": self.carrier_method,
            "lock_metrics": {k: round(v, 4) for k, v in self.lock_metrics.items()},
            "constellation_quality_before": round(self.constellation_quality_before, 4),
            "constellation_quality_after": round(self.constellation_quality_after, 4),
            "failure_reason": self.failure_reason,
            "processing_history": self.processing_history,
        }
        if include_arrays:
            data["symbol_samples_count"] = len(self.symbol_samples) if self.symbol_samples is not None else 0
            data["synchronized_iq_count"] = len(self.synchronized_iq) if self.synchronized_iq is not None else 0
        return data

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(include_arrays=False), indent=indent)
