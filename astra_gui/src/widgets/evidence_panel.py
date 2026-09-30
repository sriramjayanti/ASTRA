"""
ASTRA Evidence Inspector Panel.
Right-hand analytical dock displaying active stage metrics, confidence badges, supporting/contradicting evidence, and alternatives.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QScrollArea, QFrame, QPushButton, QTextEdit
)
from PySide6.QtCore import Signal, Qt
from typing import Optional, Dict, Any

from ..theme.theme_manager import ThemeManager
from ..theme.styles import get_badge_stylesheet


class EvidencePanel(QWidget):
    """Right-hand evidence & confidence inspector."""

    override_requested = Signal(str, str)  # (field_name, override_value)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.tm = ThemeManager.get_instance()
        self.setFixedWidth(300)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        # Header
        title = QLabel("EVIDENCE INSPECTOR")
        title.setStyleSheet("color: #5d6d88; font-weight: bold; font-size: 10px; letter-spacing: 1px;")
        layout.addWidget(title)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet("background: transparent;")

        container = QWidget()
        self.c_layout = QVBoxLayout(container)
        self.c_layout.setContentsMargins(0, 0, 0, 0)
        self.c_layout.setSpacing(10)

        # 1. Active Stage Card
        self.stage_card = QFrame()
        self.stage_card.setStyleSheet("background-color: #12161f; border: 1px solid #263043; border-radius: 6px; padding: 8px;")
        sc_layout = QVBoxLayout(self.stage_card)
        self.stage_title_lbl = QLabel("Stage 1: Signal Ingestion")
        self.stage_title_lbl.setStyleSheet("color: #00d2ff; font-weight: bold; font-size: 12px;")
        sc_layout.addWidget(self.stage_title_lbl)

        self.status_badge = QLabel("WAITING")
        self.status_badge.setStyleSheet(get_badge_stylesheet("WAITING"))
        sc_layout.addWidget(self.status_badge)

        self.confidence_lbl = QLabel("ASTRA Confidence: N/A")
        self.confidence_lbl.setStyleSheet("color: #9bb0cf; font-size: 11px;")
        sc_layout.addWidget(self.confidence_lbl)
        self.c_layout.addWidget(self.stage_card)

        # 2. Recovered RF Parameters Card
        self.params_card = QFrame()
        self.params_card.setStyleSheet("background-color: #12161f; border: 1px solid #263043; border-radius: 6px; padding: 8px;")
        pc_layout = QVBoxLayout(self.params_card)
        pc_layout.setContentsMargins(6, 6, 6, 6)
        pc_layout.setSpacing(4)

        pc_title = QLabel("RECOVERED PARAMETERS")
        pc_title.setStyleSheet("color: #00d2ff; font-weight: bold; font-size: 10px; letter-spacing: 1px;")
        pc_layout.addWidget(pc_title)

        self.param_labels = {}
        param_keys = [
            ("Modulation", "—"),
            ("Symbol Rate", "—"),
            ("Sample Rate", "—"),
            ("Carrier CFO", "—"),
            ("SNR / EVM", "—"),
            ("Frame Length", "—"),
            ("CRC Integrity", "—"),
        ]
        for key, default_val in param_keys:
            row_w = QWidget()
            r_lay = QHBoxLayout(row_w)
            r_lay.setContentsMargins(2, 2, 2, 2)
            k_lbl = QLabel(f"{key}:")
            k_lbl.setStyleSheet("color: #7b8ea8; font-size: 10px; font-weight: bold;")
            v_lbl = QLabel(default_val)
            v_lbl.setStyleSheet("color: #64dfdf; font-size: 11px; font-family: Consolas, monospace; font-weight: bold;")
            v_lbl.setAlignment(Qt.AlignRight)
            r_lay.addWidget(k_lbl)
            r_lay.addStretch()
            r_lay.addWidget(v_lbl)
            pc_layout.addWidget(row_w)
            self.param_labels[key] = v_lbl

        self.c_layout.addWidget(self.params_card)

        # 2. Supporting Evidence Box
        self.sup_card = QFrame()
        self.sup_card.setStyleSheet("background-color: #12161f; border: 1px solid #263043; border-radius: 6px; padding: 8px;")
        sup_layout = QVBoxLayout(self.sup_card)
        sup_title = QLabel("SUPPORTING EVIDENCE")
        sup_title.setStyleSheet("color: #00f59b; font-weight: bold; font-size: 10px; letter-spacing: 1px;")
        sup_layout.addWidget(sup_title)

        self.sup_text = QTextEdit()
        self.sup_text.setReadOnly(True)
        self.sup_text.setStyleSheet("background-color: #0a0c10; color: #9bb0cf; border: 1px solid #263043; font-size: 11px;")
        self.sup_text.setFixedHeight(90)
        sup_layout.addWidget(self.sup_text)
        self.c_layout.addWidget(self.sup_card)

        # 3. Contradictions & Warnings Box
        self.contra_card = QFrame()
        self.contra_card.setStyleSheet("background-color: #12161f; border: 1px solid #263043; border-radius: 6px; padding: 8px;")
        contra_layout = QVBoxLayout(self.contra_card)
        contra_title = QLabel("CONTRADICTIONS & UNCERTAINTIES")
        contra_title.setStyleSheet("color: #ff0055; font-weight: bold; font-size: 10px; letter-spacing: 1px;")
        contra_layout.addWidget(contra_title)

        self.contra_text = QTextEdit()
        self.contra_text.setReadOnly(True)
        self.contra_text.setStyleSheet("background-color: #0a0c10; color: #ff88a3; border: 1px solid #263043; font-size: 11px;")
        self.contra_text.setFixedHeight(80)
        contra_layout.addWidget(self.contra_text)
        self.c_layout.addWidget(self.contra_card)

        # 4. Alternative Hypotheses
        self.alt_card = QFrame()
        self.alt_card.setStyleSheet("background-color: #12161f; border: 1px solid #263043; border-radius: 6px; padding: 8px;")
        alt_layout = QVBoxLayout(self.alt_card)
        alt_title = QLabel("COMPETING CANDIDATES")
        alt_title.setStyleSheet("color: #9d4edd; font-weight: bold; font-size: 10px; letter-spacing: 1px;")
        alt_layout.addWidget(alt_title)

        self.alt_text = QTextEdit()
        self.alt_text.setReadOnly(True)
        self.alt_text.setStyleSheet("background-color: #0a0c10; color: #9bb0cf; border: 1px solid #263043; font-size: 11px;")
        self.alt_text.setFixedHeight(80)
        alt_layout.addWidget(self.alt_text)
        self.c_layout.addWidget(self.alt_card)

        self.c_layout.addStretch()
        scroll.setWidget(container)
        layout.addWidget(scroll)

    def set_stage_evidence(
        self,
        stage_title: str,
        status: str,
        confidence_score: float,
        supporting: list,
        contradicting: list,
        alternatives: list
    ):
        """Updates panel contents with active field explainability metrics."""
        self.stage_title_lbl.setText(stage_title)
        self.status_badge.setText(status)
        self.status_badge.setStyleSheet(get_badge_stylesheet(status))
        self.confidence_lbl.setText(f"ASTRA Confidence: {confidence_score:.2f}")

        sup_str = "\n".join(f"+ {s}" for s in supporting) if supporting else "Awaiting stage validation evidence..."
        self.sup_text.setPlainText(sup_str)

        contra_str = "\n".join(f"- {c}" for c in contradicting) if contradicting else "None. Evidence mutually consistent."
        self.contra_text.setPlainText(contra_str)

        alt_str = "\n".join(f"• {a}" for a in alternatives) if alternatives else "Top candidate dominant."
        self.alt_text.setPlainText(alt_str)

    def update_recovered_params(self, params: Dict[str, Any]):
        """Updates the visible parameter values in the RECOVERED PARAMETERS card."""
        for k, v in params.items():
            if k in self.param_labels:
                self.param_labels[k].setText(str(v))

    def reset_params(self):
        """Resets all parameter rows to waiting placeholder."""
        for lbl in self.param_labels.values():
            lbl.setText("—")
