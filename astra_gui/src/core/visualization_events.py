"""
ASTRA Stage 16: Visualization Event Model.
Defines atomic events emitted by processing stages to drive the 3D Signal World and UI timelines.
"""

from dataclasses import dataclass, field
from enum import Enum
import time
from typing import Dict, Any, Optional


class EventType(str, Enum):
    SIGNAL_LOADED = "SIGNAL_LOADED"
    SIGNAL_PREPROCESSED = "SIGNAL_PREPROCESSED"
    DSP_FILTERED = "DSP_FILTERED"
    MODULATION_HYPOTHESIS = "MODULATION_HYPOTHESIS"
    SYMBOL_RATE_ESTIMATED = "SYMBOL_RATE_ESTIMATED"
    CANDIDATES_GENERATED = "CANDIDATES_GENERATED"
    CFO_CORRECTED = "CFO_CORRECTED"
    TIMING_LOCKED = "TIMING_LOCKED"
    CARRIER_LOCKED = "CARRIER_LOCKED"
    CONSTELLATION_CLUSTERED = "CONSTELLATION_CLUSTERED"
    DEMODULATION_COMPLETED = "DEMODULATION_COMPLETED"
    DEINTERLEAVING_APPLIED = "DEINTERLEAVING_APPLIED"
    FEC_CORRECTED = "FEC_CORRECTED"
    CRC_CHECKED = "CRC_CHECKED"
    PIPELINE_RANKED = "PIPELINE_RANKED"
    AUTOCORRELATION_PEAKS = "AUTOCORRELATION_PEAKS"
    FRAME_LENGTH_DISCOVERED = "FRAME_LENGTH_DISCOVERED"
    TRANSFORMER_REGIONS = "TRANSFORMER_REGIONS"
    HEADER_PARSED = "HEADER_PARSED"
    PAYLOAD_EXTRACTED = "PAYLOAD_EXTRACTED"
    EXPLANATION_SYNTHESIZED = "EXPLANATION_SYNTHESIZED"
    CONTRADICTION_FLAGGED = "CONTRADICTION_FLAGGED"
    PIPELINE_ERROR = "PIPELINE_ERROR"


@dataclass
class VisualizationEvent:
    """Atomic event packaging real signal transformation data for visual playback."""
    stage_id: int
    event_type: EventType
    payload: Dict[str, Any] = field(default_factory=dict)
    event_id: str = ""
    timestamp: float = field(default_factory=time.time)
    progress: float = 1.0
    source_result_id: str = ""
    duration_hint_ms: int = 500
    priority: int = 1

    def __post_init__(self):
        if not self.event_id:
            self.event_id = f"evt_s{self.stage_id}_{self.event_type.value}_{int(self.timestamp*1000)}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "stage_id": self.stage_id,
            "event_type": self.event_type.value,
            "progress": self.progress,
            "payload": self.payload,
            "source_result_id": self.source_result_id,
            "duration_hint_ms": self.duration_hint_ms,
            "priority": self.priority
        }
