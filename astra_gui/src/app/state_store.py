"""
ASTRA Central State Store.
Single source of truth for capture status, pipeline results, active candidates, and UI modes.
"""

from PySide6.QtCore import QObject, Signal
from typing import Dict, Any, Optional, List
import numpy as np


class GUIStateStore(QObject):
    """Observable state store for ASTRA Desktop Workstation."""

    _instance: Optional["GUIStateStore"] = None

    # State modification signals
    capture_loaded = Signal(dict)               # (capture_metadata)
    stage_selected = Signal(int)                # (stage_id)
    candidate_selected = Signal(str)            # (pipeline_id)
    mode_changed = Signal(str)                  # ("AUTO" or "EXPERT")
    playback_state_changed = Signal(bool)       # (is_playing)
    playback_speed_changed = Signal(float)      # (speed multiplier)
    timeline_scrubbed = Signal(int)             # (stage_id)
    stage_status_changed = Signal(int, str)     # (stage_id, status_str)
    stage_result_updated = Signal(int, dict)    # (stage_id, result_dict)
    override_applied = Signal(str, object)      # (field_name, override_val)
    reduced_motion_toggled = Signal(bool)       # (enabled)
    logs_appended = Signal(str, str, str)       # (level, stage, message)

    def __init__(self):
        super().__init__()
        # 1. Capture State
        self.capture_file: Optional[str] = None
        self.capture_name: str = "No Capture Loaded"
        self.sample_rate: float = 192000.0
        self.duration_s: float = 0.0
        self.sample_count: int = 0
        self.raw_iq: Optional[np.ndarray] = None
        self.decimated_iq: Optional[np.ndarray] = None

        # 2. Pipeline Execution State
        self.mode: str = "AUTO"  # "AUTO" or "EXPERT"
        self.active_stage_id: int = 1
        self.is_analyzing: bool = False
        self.stage_statuses: Dict[int, str] = {i: "WAITING" for i in range(1, 16)}
        self.stage_results: Dict[int, Dict[str, Any]] = {}
        self.final_explainability_result: Optional[Dict[str, Any]] = None

        # 3. Candidates State
        self.candidates: List[Dict[str, Any]] = []
        self.selected_pipeline_id: Optional[str] = None

        # 4. Playback / Timeline State
        self.is_playing: bool = False
        self.playback_speed: float = 1.0
        self.timeline_stage: int = 1

        # 5. User Overrides & Settings
        self.user_overrides: Dict[str, Any] = {}
        self.reduced_motion: bool = False
        self.particles_enabled: bool = True
        self.beginner_mode: bool = False

        # 6. Logs & Telemetry
        self.logs: List[Dict[str, str]] = []

    @classmethod
    def get_instance(cls) -> "GUIStateStore":
        if cls._instance is None:
            cls._instance = GUIStateStore()
        return cls._instance

    def set_capture(
        self,
        name: str,
        iq_data: np.ndarray,
        sample_rate: float = 192000.0,
        filepath: Optional[str] = None
    ):
        """Loads a new capture into state."""
        self.capture_name = name
        self.capture_file = filepath
        self.sample_rate = float(sample_rate)
        self.sample_count = len(iq_data)
        self.duration_s = float(self.sample_count / self.sample_rate) if self.sample_rate > 0 else 0.0
        self.raw_iq = iq_data

        # Precompute intelligent decimation for display buffers
        if len(iq_data) > 32768:
            step = int(np.ceil(len(iq_data) / 32768))
            self.decimated_iq = iq_data[::step][:32768]
        else:
            self.decimated_iq = iq_data.copy()

        # Reset pipeline state
        self.reset_pipeline_state()

        meta = {
            "name": self.capture_name,
            "filepath": self.capture_file,
            "sample_rate": self.sample_rate,
            "sample_count": self.sample_count,
            "duration_s": self.duration_s
        }
        self.capture_loaded.emit(meta)
        self.append_log("INFO", "Signal Ingestion", f"Loaded capture '{name}': {self.sample_count} samples ({self.duration_s:.3f}s)")

    def reset_pipeline_state(self):
        """Clears previous stage outputs for a fresh run."""
        self.stage_statuses = {i: "WAITING" for i in range(1, 16)}
        self.stage_results.clear()
        self.candidates.clear()
        self.selected_pipeline_id = None
        self.final_explainability_result = None
        self.active_stage_id = 1
        self.timeline_stage = 1

    def set_active_stage(self, stage_id: int):
        if 1 <= stage_id <= 15:
            self.active_stage_id = stage_id
            self.timeline_stage = stage_id
            self.stage_selected.emit(stage_id)

    def set_mode(self, mode: str):
        if mode in ("AUTO", "EXPERT"):
            self.mode = mode
            self.mode_changed.emit(mode)
            self.append_log("INFO", "System", f"Switched workstation to {mode} mode.")

    def set_stage_status(self, stage_id: int, status: str):
        self.stage_statuses[stage_id] = status
        self.stage_status_changed.emit(stage_id, status)

    def set_stage_result(self, stage_id: int, result: Dict[str, Any]):
        self.stage_results[stage_id] = result
        self.stage_result_updated.emit(stage_id, result)

    def set_user_override(self, field_name: str, value: Any):
        self.user_overrides[field_name] = value
        self.override_applied.emit(field_name, value)
        self.append_log("WARNING", "Expert Mode", f"User manually applied override for '{field_name}': {value}")

    def select_candidate(self, pipeline_id: str):
        self.selected_pipeline_id = pipeline_id
        self.candidate_selected.emit(pipeline_id)

    def append_log(self, level: str, stage: str, message: str):
        log_entry = {"level": level, "stage": stage, "message": message}
        self.logs.append(log_entry)
        self.logs_appended.emit(level, stage, message)
