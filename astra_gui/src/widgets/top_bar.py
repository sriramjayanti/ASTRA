"""
ASTRA Top Command Bar.
Contains title branding, capture telemetry, AUTO/EXPERT toggle, execution triggers, and demo launch button.
"""

from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QLabel, QPushButton, QComboBox, QFrame
)
from PySide6.QtCore import Signal
from typing import Optional
from pathlib import Path
import json

from ..theme.theme_manager import ThemeManager


class TopBarWidget(QWidget):
    """Top command bar for ASTRA Desktop Workstation."""

    analyze_clicked = Signal()
    stop_clicked = Signal()
    demo_clicked = Signal()
    mode_toggled = Signal(str)      # "AUTO" or "EXPERT"
    open_capture_clicked = Signal()
    sample_selected = Signal(str)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.tm = ThemeManager.get_instance()
        self.setFixedHeight(54)
        self._build_ui()

    def _build_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 8, 16, 8)
        layout.setSpacing(12)

        # 1. Branding Title
        title_label = QLabel("ASTRA")
        title_label.setStyleSheet("""
            font-size: 18px;
            font-weight: 900;
            color: #00d2ff;
            letter-spacing: 2px;
        """)
        layout.addWidget(title_label)

        sub_label = QLabel("SIGNAL INTELLIGENCE WORKSTATION")
        sub_label.setStyleSheet("font-size: 10px; font-weight: bold; color: #5d6d88; letter-spacing: 1px;")
        layout.addWidget(sub_label)

        # Separator
        layout.addWidget(self._create_separator())

        # 2. File Loading / Open
        self.btn_open = QPushButton("OPEN CAPTURE")
        self.btn_open.clicked.connect(self.open_capture_clicked.emit)
        layout.addWidget(self.btn_open)

        # Quick Load Sample Signals Dropdown
        self.sample_combo = QComboBox()
        self.sample_combo.setStyleSheet("""
            QComboBox {
                background-color: #181e2b;
                color: #64dfdf;
                border: 1px solid #263043;
                border-radius: 4px;
                padding: 4px 10px;
                font-size: 11px;
                font-weight: 600;
            }
            QComboBox QAbstractItemView {
                background-color: #10141d;
                color: #f0f4fc;
                selection-background-color: #263043;
                selection-color: #00d2ff;
            }
        """)
        self.sample_combo.addItem("Select Test Signal...")
        self._populate_samples()
        self.sample_combo.currentIndexChanged.connect(self._on_sample_changed)
        layout.addWidget(self.sample_combo)

        # Capture Name & Telemetry Badge
        self.capture_badge = QLabel("No Capture Loaded")
        self.capture_badge.setStyleSheet("""
            background-color: #181e2b;
            color: #9bb0cf;
            border: 1px solid #263043;
            border-radius: 4px;
            padding: 4px 10px;
            font-size: 11px;
            font-weight: 600;
        """)
        layout.addWidget(self.capture_badge)

        self.telemetry_label = QLabel("")
        self.telemetry_label.setStyleSheet("color: #5d6d88; font-size: 11px;")
        layout.addWidget(self.telemetry_label)

        layout.addStretch()

        # 3. Mode Toggle (AUTO vs EXPERT)
        mode_label = QLabel("MODE:")
        mode_label.setStyleSheet("color: #9bb0cf; font-weight: bold; font-size: 11px;")
        layout.addWidget(mode_label)

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["AUTO", "EXPERT"])
        self.mode_combo.currentTextChanged.connect(self.mode_toggled.emit)
        self.mode_combo.setStyleSheet("""
            QComboBox {
                background-color: #181e2b;
                color: #00d2ff;
                border: 1px solid #263043;
                border-radius: 4px;
                padding: 4px 10px;
                font-weight: bold;
            }
        """)
        layout.addWidget(self.mode_combo)

        # 4. Action Buttons
        self.btn_demo = QPushButton("RUN DEMO ('hi hello')")
        self.btn_demo.setStyleSheet("""
            QPushButton {
                background-color: #212838;
                border: 1px solid #9d4edd;
                color: #f0f4fc;
                border-radius: 4px;
                padding: 6px 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #9d4edd;
                color: #ffffff;
            }
        """)
        self.btn_demo.clicked.connect(self.demo_clicked.emit)
        layout.addWidget(self.btn_demo)

        self.btn_analyze = QPushButton("ANALYZE")
        self.btn_analyze.setObjectName("PrimaryAction")
        self.btn_analyze.clicked.connect(self.analyze_clicked.emit)
        layout.addWidget(self.btn_analyze)

        self.btn_stop = QPushButton("STOP")
        self.btn_stop.setStyleSheet("""
            QPushButton {
                background-color: #38121f;
                border: 1px solid #ff0055;
                color: #ff88a3;
                border-radius: 4px;
                padding: 6px 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #ff0055;
                color: #ffffff;
            }
        """)
        self.btn_stop.clicked.connect(self.stop_clicked.emit)
        layout.addWidget(self.btn_stop)

    def _create_separator(self) -> QFrame:
        sep = QFrame()
        sep.setFrameShape(QFrame.VLine)
        sep.setStyleSheet("color: #263043;")
        return sep

    def _populate_samples(self):
        manifest_path = Path("test_signals/MANIFEST.json")
        if manifest_path.exists():
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    manifest = json.load(f)
                for item in manifest:
                    fname = item.get("filename", "")
                    payload = item.get("expected_payload")
                    mod = item.get("modulation", "")
                    label = f"{fname} [{mod}]"
                    if payload:
                        label += f" ('{payload[:15]}')"
                    full_path = str((Path("test_signals") / fname).resolve())
                    self.sample_combo.addItem(label, full_path)
            except Exception:
                pass

    def _on_sample_changed(self, idx: int):
        if idx > 0:
            fpath = self.sample_combo.itemData(idx)
            if fpath:
                self.sample_selected.emit(fpath)

    def update_capture_info(self, name: str, sample_rate: float, duration_s: float, mod: str = "", baud: float = 0.0):
        self.capture_badge.setText(name)
        info = f"SR: {sample_rate/1000:.1f} kHz | Dur: {duration_s:.3f}s"
        if mod:
            info += f" | Mod: {mod}"
        if baud > 0:
            info += f" | Baud: {baud:,.0f}"
        self.telemetry_label.setText(info)
