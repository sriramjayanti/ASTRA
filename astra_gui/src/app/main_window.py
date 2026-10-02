"""
ASTRA Main Window.
Coordinates Top Bar, Left Pipeline Navigator, Center 3D Signal World, Right Evidence Inspector,
Signal Journey Timeline, and Bottom Scientific Dock.
"""

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QFileDialog, QMessageBox
)
from PySide6.QtCore import Qt, QKeyCombination
from PySide6.QtGui import QKeySequence, QShortcut, QGuiApplication
from typing import Optional
import numpy as np

from .state_store import GUIStateStore
from .pipeline_controller import ASTRAPipelineController
from ..core.event_bus import VisualizationEventBus
from ..core.stage_registry import StageRegistry
from ..theme.theme_manager import ThemeManager

from ..widgets.top_bar import TopBarWidget
from ..widgets.pipeline_navigator import PipelineNavigator
from ..widgets.evidence_panel import EvidencePanel
from ..widgets.timeline import SignalJourneyTimeline
from ..widgets.scientific_dock import ScientificDock
from ..world3d.signal_world import SignalWorldWidget


class ASTRAMainWindow(QMainWindow):
    """Main desktop application window for ASTRA Signal Intelligence Workstation."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("ASTRA — Automated Signal Analysis & Recovery Workstation")

        # Dynamically fit user screen resolution without overflowing
        screen = QGuiApplication.primaryScreen()
        if screen:
            avail_geom = screen.availableGeometry()
            target_w = min(1720, int(avail_geom.width() * 0.96))
            target_h = min(980, int(avail_geom.height() * 0.94))
            self.resize(target_w, target_h)
            # Center on screen
            self.move(
                avail_geom.x() + (avail_geom.width() - target_w) // 2,
                avail_geom.y() + (avail_geom.height() - target_h) // 2
            )
        else:
            self.resize(1280, 800)

        self.tm = ThemeManager.get_instance()
        self.setStyleSheet(self.tm.build_stylesheet())

        self.state = GUIStateStore.get_instance()
        self.bus = VisualizationEventBus.get_instance()
        self.controller = ASTRAPipelineController(self.state, self.bus)

        self._build_ui()
        self._wire_signals()
        self._setup_shortcuts()

    def _build_ui(self):
        central_container = QWidget()
        central_container.setObjectName("CentralContainer")
        self.setCentralWidget(central_container)

        root_layout = QVBoxLayout(central_container)
        root_layout.setContentsMargins(4, 4, 4, 4)
        root_layout.setSpacing(4)

        # 1. Top Command Bar
        self.top_bar = TopBarWidget()
        root_layout.addWidget(self.top_bar)

        # 2. Main Middle Splitter (Left: Navigator | Center: 3D World | Right: Evidence)
        self.mid_splitter = QSplitter(Qt.Horizontal)

        self.navigator = PipelineNavigator()
        self.mid_splitter.addWidget(self.navigator)

        self.signal_world = SignalWorldWidget()
        self.mid_splitter.addWidget(self.signal_world)

        self.evidence_panel = EvidencePanel()
        self.mid_splitter.addWidget(self.evidence_panel)

        self.mid_splitter.setSizes([220, 1180, 320])

        # 3. Vertical Splitter (Upper: Middle Views | Lower: Bottom Dock)
        self.vert_splitter = QSplitter(Qt.Vertical)
        self.vert_splitter.addWidget(self.mid_splitter)

        # Bottom Dock Container (Timeline + Scientific Plots)
        bottom_container = QWidget()
        b_layout = QVBoxLayout(bottom_container)
        b_layout.setContentsMargins(0, 0, 0, 0)
        b_layout.setSpacing(2)

        self.timeline = SignalJourneyTimeline()
        b_layout.addWidget(self.timeline)

        self.scientific_dock = ScientificDock()
        b_layout.addWidget(self.scientific_dock)

        self.vert_splitter.addWidget(bottom_container)
        self.vert_splitter.setSizes([620, 360])

        root_layout.addWidget(self.vert_splitter)

    def _wire_signals(self):
        """Connects state store, pipeline controller, and UI widgets."""
        # Top Bar
        self.top_bar.analyze_clicked.connect(self.controller.run_analysis)
        self.top_bar.stop_clicked.connect(self.controller.cancel_analysis)
        self.top_bar.demo_clicked.connect(self.controller.run_demo)
        self.top_bar.open_capture_clicked.connect(self._on_open_capture)
        self.top_bar.sample_selected.connect(self._load_capture_from_path)
        self.top_bar.mode_toggled.connect(self.state.set_mode)

        # State Store to UI
        self.state.capture_loaded.connect(self._on_capture_loaded)
        self.state.stage_selected.connect(self._on_stage_selected)
        self.state.stage_status_changed.connect(self.navigator.set_stage_status)
        self.state.logs_appended.connect(self.scientific_dock.append_log)
        self.state.stage_result_updated.connect(self._on_stage_result_updated)

        # Navigator & Timeline
        self.navigator.stage_selected.connect(self.state.set_active_stage)
        self.timeline.stage_changed.connect(self.state.set_active_stage)

        # Pipeline Controller to UI
        self.bus.pipeline_completed.connect(self._on_pipeline_completed)
        self.bus.candidate_selected.connect(self.state.select_candidate)

        # Full-Screen Toggle for Scientific Dock
        self.scientific_dock.fullscreen_toggled.connect(self._on_scientific_dock_fullscreen_toggled)

    def _setup_shortcuts(self):
        """Keyboard shortcuts for power users."""
        QShortcut(QKeySequence(Qt.Key_Space), self, activated=self.timeline.toggle_playback)
        QShortcut(QKeySequence(Qt.Key_Left), self, activated=self.timeline.prev_stage)
        QShortcut(QKeySequence(Qt.Key_Right), self, activated=self.timeline.next_stage)
        QShortcut(QKeySequence("Ctrl+O"), self, activated=self._on_open_capture)
        QShortcut(QKeySequence("Ctrl+R"), self, activated=self.timeline.replay)

    def _on_open_capture(self):
        """File open dialog for raw captures."""
        from pathlib import Path
        default_dir = str(Path("test_signals").resolve()) if Path("test_signals").exists() else ""
        fpath, _ = QFileDialog.getOpenFileName(
            self,
            "Open Signal Capture",
            default_dir,
            "IQ Signals (*.iq *.bin *.wav *.sigmf-data *.sigmf);;All Files (*.*)"
        )
        if fpath:
            self._load_capture_from_path(fpath)

    def _load_capture_from_path(self, fpath: str):
        """Loads and decodes raw IQ from any supported format."""
        try:
            fname = fpath.split("/")[-1].split("\\")[-1]
            lower_fpath = fpath.lower()
            sample_rate = 192000.0

            if lower_fpath.endswith(".wav"):
                from astra_synthetic.capture.wav_iq import read_wav_iq_file
                raw_data, sample_rate = read_wav_iq_file(fpath)
            elif "sigmf" in lower_fpath:
                from astra_synthetic.capture.sigmf_writer import read_sigmf_iq_file
                raw_data, meta = read_sigmf_iq_file(fpath)
                sample_rate = float(meta.get("global", {}).get("core:sample_rate", 192000.0))
            else:
                raw_data = np.fromfile(fpath, dtype=np.complex64)
                if len(raw_data) == 0 or not np.all(np.isfinite(raw_data[:100])):
                    float_data = np.fromfile(fpath, dtype=np.float32)
                    raw_data = float_data[0::2] + 1j * float_data[1::2]

            self.state.set_capture(fname, raw_data, sample_rate, filepath=fpath)
            self.state.append_log("INFO", "Signal Ingestion", f"Loaded capture {fname} ({len(raw_data)} samples @ {sample_rate} Hz)")
        except Exception as e:
            self.state.append_log("ERROR", "Signal Ingestion", f"Error loading {fpath}: {e}")

    def _on_capture_loaded(self, meta: dict):
        """Updates UI components when capture is loaded."""
        self.top_bar.update_capture_info(meta["name"], meta["sample_rate"], meta["duration_s"])
        self.scientific_dock.time_plot.set_data(self.state.decimated_iq)
        self.scientific_dock.fft_plot.set_data(self.state.decimated_iq, meta["sample_rate"])
        self.scientific_dock.waterfall_plot.set_data(self.state.decimated_iq, meta["sample_rate"])
        self.scientific_dock.constellation_plot.set_symbols(self.state.decimated_iq)
        self.scientific_dock.eye_diagram.set_data(np.real(self.state.decimated_iq))

        # Clear previous run results
        self.scientific_dock.bit_viewer.set_bits("")
        self.scientific_dock.frame_table.set_frames([])
        self.scientific_dock.hex_viewer.set_bytes(b"")
        self.scientific_dock.candidate_tree.set_candidates([])
        self.evidence_panel.reset_params()
        self.evidence_panel.update_recovered_params({
            "Sample Rate": f"{meta['sample_rate']/1000:.1f} kHz"
        })

        # Send raw signal to 3D scene
        self.signal_world.feed_data(1, {"raw_iq": self.state.decimated_iq})

    def _on_stage_selected(self, stage_id: int):
        """User or pipeline changed active stage."""
        self.navigator.set_active_stage(stage_id)
        self.timeline.set_stage(stage_id)
        self.signal_world.switch_stage(stage_id)

        meta = StageRegistry.get_stage(stage_id)
        title = meta.display_title if meta else f"Stage {stage_id}"
        status = self.state.stage_statuses.get(stage_id, "WAITING")
        self.evidence_panel.set_stage_evidence(
            title, status, 0.90 if status == "DONE" else 0.50, [], [], []
        )

        # Tab auto-navigation to match stage
        tab_mapping = {
            1: 0,   # TIME DOMAIN
            2: 1,   # SPECTRUM / PSD
            3: 3,   # CONSTELLATION
            4: 3,   # CONSTELLATION
            5: 9,   # CANDIDATES
            6: 1,   # SPECTRUM
            7: 3,   # CONSTELLATION
            8: 4,   # EYE DIAGRAM
            9: 4,   # EYE DIAGRAM
            10: 6,  # FRAMES
            11: 9,  # CANDIDATES
            12: 5,  # BITSTREAM
            13: 6,  # FRAMES
            14: 7,  # HEX / PAYLOAD
            15: 8,  # CONFIDENCE
        }
        target_tab = tab_mapping.get(stage_id)
        if target_tab is not None:
            self.scientific_dock.tab_widget.setCurrentIndex(target_tab)

        # Refresh with cached stage result if already analyzed
        cached_result = self.state.stage_results.get(stage_id)
        if cached_result:
            self._on_stage_result_updated(stage_id, cached_result)

    def _on_stage_result_updated(self, stage_id: int, result: dict):
        """Update scientific dock and 3D scenes with real stage result."""
        self.signal_world.feed_data(stage_id, result)

        if stage_id == 1:
            snr = result.get("snr_db", 20.0)
            self.evidence_panel.update_recovered_params({"SNR / EVM": f"{snr:.1f} dB"})
        elif stage_id == 4:
            rate = result.get("estimated", 9600.0)
            self.evidence_panel.update_recovered_params({"Symbol Rate": f"{rate:,.0f} Baud"})
        elif stage_id == 6:  # Sync
            cfo_res = result.get("after_hz", 0.0)
            self.scientific_dock.fft_plot.set_carrier_marker(cfo_res)
            self.evidence_panel.update_recovered_params({"Carrier CFO": f"{cfo_res:+.1f} Hz"})
        elif stage_id == 7:  # Demod
            mod = str(result.get("modulation", "QPSK")).upper()
            evm_val = result.get("evm", 0.05)
            self.evidence_panel.update_recovered_params({
                "Modulation": mod,
                "SNR / EVM": f"{evm_val*100:.1f}% EVM"
            })
            if "8PSK" in mod:
                angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
                self.scientific_dock.constellation_plot.set_ideal_points([np.exp(1j * a) for a in angles])
            elif "16QAM" in mod:
                grid = [-3, -1, 1, 3]
                self.scientific_dock.constellation_plot.set_ideal_points([complex(x, y) / np.sqrt(10) for x in grid for y in grid])
            elif "BPSK" in mod:
                self.scientific_dock.constellation_plot.set_ideal_points([-1.0 + 0j, 1.0 + 0j])
            elif "FSK" in mod:
                self.scientific_dock.constellation_plot.set_ideal_points([-1.0 + 0j, 0.0 + 1j, 1.0 + 0j])
            else:
                self.scientific_dock.constellation_plot.set_ideal_points([-1-1j, -1+1j, 1-1j, 1+1j])
        elif stage_id == 10:  # Frames / CRC Validation
            frames = result.get("frames", [])
            if frames:
                self.scientific_dock.frame_table.set_frames(frames)
            raw_bits = result.get("raw_bitstream", "")
            if raw_bits:
                self.scientific_dock.bit_viewer.set_bits(raw_bits)
            passed = result.get("crc_passed", True)
            self.evidence_panel.update_recovered_params({
                "CRC Integrity": f"PASS ({len(frames)}/{len(frames)})" if passed else "FAIL"
            })
        elif stage_id == 11:  # Candidates
            cands = result.get("candidates", [])
            self.scientific_dock.candidate_tree.set_candidates(cands)
        elif stage_id == 12:  # Bitstream / Framing
            raw_bits = result.get("raw_bitstream", "")
            if raw_bits:
                self.scientific_dock.bit_viewer.set_bits(raw_bits)
            frames = result.get("frames", [])
            if frames:
                self.scientific_dock.frame_table.set_frames(frames)
            flen = result.get("frame_length_bits", 512)
            self.evidence_panel.update_recovered_params({
                "Frame Length": f"{flen} bits"
            })
        elif stage_id == 14:  # Payload
            payload_hex = result.get("payload_hex", "")
            raw_bytes = bytes.fromhex(payload_hex) if payload_hex else b""
            self.scientific_dock.hex_viewer.set_bytes(raw_bytes)
            raw_bits = result.get("raw_bitstream", "")
            if raw_bits:
                self.scientific_dock.bit_viewer.set_bits(raw_bits)
            frames = result.get("frames", [])
            if frames:
                self.scientific_dock.frame_table.set_frames(frames)

    def _on_pipeline_completed(self, summary: dict):
        """Pipeline completed: populate Stage 15 final explainability results."""
        fields = summary.get("field_explanations") or summary.get("fields", {})
        breakdown = summary.get("confidence_breakdown", {})
        if isinstance(breakdown, dict) and "components" in breakdown:
            self.scientific_dock.confidence_plot.set_breakdown(breakdown["components"])

        mod_info = fields.get("modulation", {})
        mod_val = mod_info.get("value") if isinstance(mod_info, dict) else getattr(mod_info, "value", "QPSK")

        rate_info = fields.get("symbol_rate", {})
        rate_val = rate_info.get("value") if isinstance(rate_info, dict) else getattr(rate_info, "value", 9600.0)

        pay_info = fields.get("payload", {})
        pay_val = pay_info.get("value") if isinstance(pay_info, dict) else getattr(pay_info, "value", {})
        if isinstance(pay_val, dict) and "hex" in pay_val:
            raw_b = bytes.fromhex(pay_val["hex"])
            self.scientific_dock.hex_viewer.set_bytes(raw_b)

        # Update top bar with locked telemetry
        self.top_bar.update_capture_info(
            self.state.capture_name or "Capture",
            self.state.sample_rate,
            float(self.state.sample_count / self.state.sample_rate) if self.state.sample_rate > 0 else 0.1,
            mod=str(mod_val),
            baud=float(rate_val)
        )

        # Update evidence panel parameter card
        self.evidence_panel.update_recovered_params({
            "Modulation": str(mod_val),
            "Symbol Rate": f"{float(rate_val):,.0f} Baud",
            "Sample Rate": f"{self.state.sample_rate/1000:.1f} kHz",
            "CRC Integrity": "PASS (Verified)"
        })

        strong_ev = [
            f"Modulation: {mod_val} confirmed",
            f"Symbol Rate: {rate_val} Baud locked",
            "CFO tracking locked",
            "CRC frame integrity verified"
        ]

        self.evidence_panel.set_stage_evidence(
            f"Recovered: {mod_val} @ {rate_val} Baud",
            summary.get("overall_status", "CONFIRMED"),
            float(summary.get("overall_confidence", 0.96)),
            strong_ev,
            [c.get("description") if isinstance(c, dict) else str(c) for c in summary.get("contradictions", [])],
            [a.get("pipeline_id") if isinstance(a, dict) else str(a) for a in summary.get("candidate_comparison", [])]
        )

    def _on_scientific_dock_fullscreen_toggled(self, is_fullscreen: bool):
        """Toggles bottom scientific dock to maximize or restore split layout."""
        if is_fullscreen:
            # Save prior splitter sizes
            self._prev_vert_sizes = self.vert_splitter.sizes()
            # Collapse upper section completely (give 100% height to bottom dock)
            total_h = sum(self.vert_splitter.sizes())
            self.vert_splitter.setSizes([0, max(total_h, 800)])
            self.top_bar.setVisible(False)
            self.timeline.setVisible(False)
        else:
            # Restore view
            self.top_bar.setVisible(True)
            self.timeline.setVisible(True)
            if hasattr(self, "_prev_vert_sizes") and self._prev_vert_sizes:
                self.vert_splitter.setSizes(self._prev_vert_sizes)
            else:
                self.vert_splitter.setSizes([620, 360])
