"""
ASTRA Central Event Bus.
Qt-based reactive signal bus decoupling DSP/AI processing from GUI rendering.
"""

from PySide6.QtCore import QObject, Signal
from typing import Optional, Dict, Any
from .visualization_events import VisualizationEvent


class VisualizationEventBus(QObject):
    """Singleton Qt Event Bus for ASTRA Workstation."""

    _instance: Optional["VisualizationEventBus"] = None

    # Pipeline execution signals
    stage_started = Signal(int, str)                # (stage_id, stage_name)
    stage_progress = Signal(int, float, str)        # (stage_id, progress 0-1, status_text)
    stage_completed = Signal(int, dict)             # (stage_id, result_summary)
    pipeline_completed = Signal(dict)               # (final_result_summary)
    error_occurred = Signal(str, str)               # (source_stage, error_message)

    # Hypothesis & Candidate signals
    candidate_created = Signal(dict)                # (candidate_dict)
    candidate_rejected = Signal(dict)               # (candidate_dict)
    candidate_selected = Signal(dict)               # (candidate_dict)

    # Scientific metric and status updates
    metric_updated = Signal(str, object)            # (metric_key, metric_value)
    system_status_updated = Signal(str, str)        # (status_title, details)

    # Direct 3D World Visualization Event stream
    visualization_event = Signal(VisualizationEvent)

    # User timeline & navigation signals
    timeline_seek = Signal(int)                     # (target_stage_id)
    timeline_playback_state = Signal(bool)          # (is_playing)

    def __init__(self):
        super().__init__()

    @classmethod
    def get_instance(cls) -> "VisualizationEventBus":
        if cls._instance is None:
            cls._instance = VisualizationEventBus()
        return cls._instance

    def emit_event(self, event: VisualizationEvent) -> None:
        """Helper to dispatch a VisualizationEvent safely."""
        self.visualization_event.emit(event)
