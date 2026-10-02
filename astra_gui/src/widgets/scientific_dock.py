"""
ASTRA Bottom Scientific Dock.
Tabbed container hosting all scientific 2D plots, bitstream viewers, frame tables, and logs.
"""

from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QTabWidget, QTextEdit, QPushButton
from PySide6.QtCore import Signal
from typing import Optional

from ..plots.time_plot import TimePlotWidget
from ..plots.fft_plot import FFTPlotWidget
from ..plots.waterfall import WaterfallPlotWidget
from ..plots.constellation_2d import Constellation2DWidget
from ..plots.eye_diagram import EyeDiagramWidget
from ..plots.confidence_plot import ConfidencePlotWidget
from .bit_viewer import BitViewerWidget
from .hex_viewer import HexViewerWidget
from .frame_table import FrameTableWidget
from .candidate_tree import CandidateTreeWidget


class ScientificDock(QWidget):
    """Bottom scientific analysis dock containing all 2D measurement views."""

    fullscreen_toggled = Signal(bool)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.is_fullscreen_mode = False

        self.tab_widget = QTabWidget()

        # Full Screen / Maximize Button in Tab Bar Corner
        self.btn_fullscreen = QPushButton("⛶ FULL SCREEN")
        self.btn_fullscreen.setToolTip("Toggle Full Screen for bottom scientific panels")
        self.btn_fullscreen.setStyleSheet("""
            QPushButton {
                background-color: #1a2232;
                color: #00d2ff;
                border: 1px solid #00d2ff;
                border-radius: 4px;
                padding: 4px 12px;
                font-size: 11px;
                font-weight: bold;
                margin-right: 6px;
                margin-top: 2px;
            }
            QPushButton:hover {
                background-color: #00d2ff;
                color: #0a0d14;
            }
        """)
        self.btn_fullscreen.clicked.connect(self._toggle_fullscreen)
        self.tab_widget.setCornerWidget(self.btn_fullscreen)

        # Tabs
        self.time_plot = TimePlotWidget()
        self.tab_widget.addTab(self.time_plot, "TIME DOMAIN")

        self.fft_plot = FFTPlotWidget()
        self.tab_widget.addTab(self.fft_plot, "SPECTRUM / PSD")

        self.waterfall_plot = WaterfallPlotWidget()
        self.tab_widget.addTab(self.waterfall_plot, "WATERFALL")

        self.constellation_plot = Constellation2DWidget()
        self.tab_widget.addTab(self.constellation_plot, "CONSTELLATION")

        self.eye_diagram = EyeDiagramWidget()
        self.tab_widget.addTab(self.eye_diagram, "EYE DIAGRAM")

        self.bit_viewer = BitViewerWidget()
        self.tab_widget.addTab(self.bit_viewer, "BITSTREAM")

        self.frame_table = FrameTableWidget()
        self.tab_widget.addTab(self.frame_table, "FRAMES")

        self.hex_viewer = HexViewerWidget()
        self.tab_widget.addTab(self.hex_viewer, "HEX / PAYLOAD")

        self.confidence_plot = ConfidencePlotWidget()
        self.tab_widget.addTab(self.confidence_plot, "CONFIDENCE")

        self.candidate_tree = CandidateTreeWidget()
        self.tab_widget.addTab(self.candidate_tree, "CANDIDATES")

        self.logs_edit = QTextEdit()
        self.logs_edit.setReadOnly(True)
        self.logs_edit.setStyleSheet("background-color: #0a0c10; color: #9bb0cf; font-family: Consolas, monospace; font-size: 11px;")
        self.tab_widget.addTab(self.logs_edit, "LOGS")

        self.layout.addWidget(self.tab_widget)

    def _toggle_fullscreen(self):
        self.is_fullscreen_mode = not self.is_fullscreen_mode
        if self.is_fullscreen_mode:
            self.btn_fullscreen.setText("🗗 RESTORE VIEW")
            self.btn_fullscreen.setStyleSheet("""
                QPushButton {
                    background-color: #ff0055;
                    color: #ffffff;
                    border: 1px solid #ff0055;
                    border-radius: 4px;
                    padding: 4px 12px;
                    font-size: 11px;
                    font-weight: bold;
                    margin-right: 6px;
                    margin-top: 2px;
                }
                QPushButton:hover {
                    background-color: #ffffff;
                    color: #ff0055;
                }
            """)
        else:
            self.btn_fullscreen.setText("⛶ FULL SCREEN")
            self.btn_fullscreen.setStyleSheet("""
                QPushButton {
                    background-color: #1a2232;
                    color: #00d2ff;
                    border: 1px solid #00d2ff;
                    border-radius: 4px;
                    padding: 4px 12px;
                    font-size: 11px;
                    font-weight: bold;
                    margin-right: 6px;
                    margin-top: 2px;
                }
                QPushButton:hover {
                    background-color: #00d2ff;
                    color: #0a0d14;
                }
            """)
        self.fullscreen_toggled.emit(self.is_fullscreen_mode)

    def append_log(self, level: str, stage: str, msg: str):
        color = "#00d2ff" if level == "INFO" else ("#ffbe0b" if level == "WARNING" else "#ff0055")
        self.logs_edit.append(f"<span style='color: {color};'><b>[{level}]</b></span> <span style='color: #5d6d88;'>({stage})</span> {msg}")
