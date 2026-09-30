"""
ASTRA Signal Journey Timeline.
Horizontal scrub bar and interactive playback controls for the 3D signal transformation story.
"""

from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QPushButton, QSlider, QLabel, QComboBox, QCheckBox
)
from PySide6.QtCore import Signal, Qt, QTimer
from typing import Optional

from ..theme.theme_manager import ThemeManager


class SignalJourneyTimeline(QWidget):
    """Horizontal interactive playback timeline bar."""

    stage_changed = Signal(int)             # Emitted when scrubbed or played
    playback_toggled = Signal(bool)         # (is_playing)
    speed_changed = Signal(float)           # (multiplier)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.tm = ThemeManager.get_instance()
        self.setFixedHeight(50)
        self.is_playing = False
        self.current_stage = 1
        self.playback_speed = 1.0

        # Auto-play timer
        self.play_timer = QTimer(self)
        self.play_timer.timeout.connect(self._on_play_tick)

        self._build_ui()

    def _build_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 6, 16, 6)
        layout.setSpacing(10)

        # 1. Transport Controls
        self.btn_prev = QPushButton("◀")
        self.btn_prev.setFixedSize(32, 28)
        self.btn_prev.clicked.connect(self.prev_stage)
        layout.addWidget(self.btn_prev)

        self.btn_play = QPushButton("PLAY ▶")
        self.btn_play.setStyleSheet("""
            QPushButton {
                background-color: #181e2b;
                border: 1px solid #00d2ff;
                color: #00d2ff;
                font-weight: bold;
                border-radius: 4px;
                padding: 4px 10px;
            }
            QPushButton:hover {
                background-color: #004b66;
            }
        """)
        self.btn_play.clicked.connect(self.toggle_playback)
        layout.addWidget(self.btn_play)

        self.btn_next = QPushButton("▶")
        self.btn_next.setFixedSize(32, 28)
        self.btn_next.clicked.connect(self.next_stage)
        layout.addWidget(self.btn_next)

        self.btn_replay = QPushButton("↺ REPLAY")
        self.btn_replay.clicked.connect(self.replay)
        layout.addWidget(self.btn_replay)

        # 2. Timeline Scrubber
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(1, 15)
        self.slider.setValue(1)
        self.slider.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 6px;
                background: #181e2b;
                border-radius: 3px;
            }
            QSlider::sub-page:horizontal {
                background: #00d2ff;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #f0f4fc;
                border: 2px solid #00d2ff;
                width: 14px;
                margin-top: -4px;
                margin-bottom: -4px;
                border-radius: 7px;
            }
        """)
        self.slider.valueChanged.connect(self._on_slider_moved)
        layout.addWidget(self.slider)

        self.stage_label = QLabel("Stage 1 / 15: Signal Ingestion")
        self.stage_label.setFixedWidth(210)
        self.stage_label.setStyleSheet("color: #9bb0cf; font-weight: bold; font-size: 11px;")
        layout.addWidget(self.stage_label)

        # 3. Playback Speed Selector
        speed_lbl = QLabel("SPEED:")
        speed_lbl.setStyleSheet("color: #5d6d88; font-size: 10px; font-weight: bold;")
        layout.addWidget(speed_lbl)

        self.speed_combo = QComboBox()
        self.speed_combo.addItems(["0.25x", "0.5x", "1.0x", "2.0x", "Instant"])
        self.speed_combo.setCurrentText("1.0x")
        self.speed_combo.currentTextChanged.connect(self._on_speed_changed)
        layout.addWidget(self.speed_combo)

    def set_stage(self, stage_id: int):
        self.current_stage = stage_id
        self.slider.blockSignals(True)
        self.slider.setValue(stage_id)
        self.slider.blockSignals(False)
        self.stage_label.setText(f"Stage {stage_id} / 15")

    def _on_slider_moved(self, val: int):
        self.current_stage = val
        self.stage_label.setText(f"Stage {val} / 15")
        self.stage_changed.emit(val)

    def toggle_playback(self):
        self.is_playing = not self.is_playing
        self.btn_play.setText("PAUSE ❚❚" if self.is_playing else "PLAY ▶")
        if self.is_playing:
            interval = int(1200 / self.playback_speed)
            self.play_timer.start(interval)
        else:
            self.play_timer.stop()
        self.playback_toggled.emit(self.is_playing)

    def _on_play_tick(self):
        if self.current_stage < 15:
            self.next_stage()
        else:
            self.toggle_playback()  # Stop at end

    def prev_stage(self):
        if self.current_stage > 1:
            self.set_stage(self.current_stage - 1)
            self.stage_changed.emit(self.current_stage)

    def next_stage(self):
        if self.current_stage < 15:
            self.set_stage(self.current_stage + 1)
            self.stage_changed.emit(self.current_stage)

    def replay(self):
        self.set_stage(1)
        self.stage_changed.emit(1)
        if not self.is_playing:
            self.toggle_playback()

    def _on_speed_changed(self, text: str):
        speed_map = {"0.25x": 0.25, "0.5x": 0.5, "1.0x": 1.0, "2.0x": 2.0, "Instant": 10.0}
        self.playback_speed = speed_map.get(text, 1.0)
        if self.is_playing:
            self.play_timer.setInterval(int(1200 / self.playback_speed))
        self.speed_changed.emit(self.playback_speed)
