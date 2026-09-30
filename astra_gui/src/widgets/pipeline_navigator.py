"""
ASTRA Pipeline Navigator.
Left sidebar displaying all 15 stages with execution indicators (WAITING, RUNNING, DONE, WARNING, FAILED).
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QScrollArea, QPushButton, QLabel, QHBoxLayout, QFrame
)
from PySide6.QtCore import Signal, Qt
from typing import Optional, Dict

from ..core.stage_registry import StageRegistry, StageMetadata
from ..theme.theme_manager import ThemeManager


class StageNavItem(QFrame):
    """Single stage button row in the navigator sidebar."""

    clicked = Signal(int)

    def __init__(self, metadata: StageMetadata, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.meta = metadata
        self.tm = ThemeManager.get_instance()
        self.is_active = False
        self.status = "WAITING"
        self._build_ui()

    def _build_ui(self):
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedHeight(40)
        self.update_style()

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 4, 10, 4)
        layout.setSpacing(8)

        # Status dot / symbol
        self.status_lbl = QLabel("●")
        self.status_lbl.setStyleSheet("color: #4a4e69; font-size: 10px;")
        layout.addWidget(self.status_lbl)

        # Stage number and title
        self.title_lbl = QLabel(self.meta.display_title)
        self.title_lbl.setStyleSheet("color: #9bb0cf; font-weight: 600; font-size: 11px;")
        layout.addWidget(self.title_lbl)

        layout.addStretch()

        # Domain badge
        self.domain_lbl = QLabel(self.meta.domain)
        self.domain_lbl.setStyleSheet("""
            background-color: #12161f;
            color: #5d6d88;
            border-radius: 3px;
            padding: 1px 4px;
            font-size: 9px;
            font-weight: bold;
        """)
        layout.addWidget(self.domain_lbl)

    def set_active(self, active: bool):
        self.is_active = active
        self.update_style()

    def set_status(self, status: str):
        self.status = status
        color_map = {
            "WAITING": "#4a4e69",
            "RUNNING": "#00d2ff",
            "DONE": "#00f59b",
            "WARNING": "#ffbe0b",
            "FAILED": "#ff0055"
        }
        col = color_map.get(status, "#4a4e69")
        self.status_lbl.setStyleSheet(f"color: {col}; font-size: 10px;")
        self.update_style()

    def update_style(self):
        if self.is_active:
            self.setStyleSheet("""
                QFrame {
                    background-color: #212838;
                    border-left: 3px solid #00d2ff;
                    border-radius: 3px;
                }
            """)
        else:
            self.setStyleSheet("""
                QFrame {
                    background-color: transparent;
                    border-left: 3px solid transparent;
                    border-radius: 3px;
                }
                QFrame:hover {
                    background-color: #181e2b;
                }
            """)

    def mousePressEvent(self, event):
        self.clicked.emit(self.meta.stage_id)


class PipelineNavigator(QWidget):
    """Navigator sidebar containing Stage 1 through Stage 15."""

    stage_selected = Signal(int)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setFixedWidth(240)
        self.items_map: Dict[int, StageNavItem] = {}
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 10, 6, 10)
        layout.setSpacing(4)

        header = QLabel("PIPELINE NAVIGATOR")
        header.setStyleSheet("color: #5d6d88; font-weight: bold; font-size: 10px; letter-spacing: 1px; padding-left: 8px;")
        layout.addWidget(header)

        # Scroll area for stages
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet("background: transparent;")

        container = QWidget()
        c_layout = QVBoxLayout(container)
        c_layout.setContentsMargins(0, 0, 0, 0)
        c_layout.setSpacing(3)

        for meta in StageRegistry.get_all_stages():
            item = StageNavItem(meta)
            item.clicked.connect(self._on_item_clicked)
            self.items_map[meta.stage_id] = item
            c_layout.addWidget(item)

        c_layout.addStretch()
        scroll.setWidget(container)
        layout.addWidget(scroll)

        # Set default active
        self.set_active_stage(1)

    def _on_item_clicked(self, stage_id: int):
        self.set_active_stage(stage_id)
        self.stage_selected.emit(stage_id)

    def set_active_stage(self, stage_id: int):
        for sid, it in self.items_map.items():
            it.set_active(sid == stage_id)

    def set_stage_status(self, stage_id: int, status: str):
        if stage_id in self.items_map:
            self.items_map[stage_id].set_status(status)
