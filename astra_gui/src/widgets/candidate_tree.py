"""
ASTRA Candidate Hypothesis Tree.
Hierarchical tree view for Expert Mode displaying competing pipeline hypotheses, ranking scores, and why losers lost.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QTreeWidget, QTreeWidgetItem, QPushButton, QHBoxLayout, QLabel
)
from PySide6.QtCore import Signal
from typing import Optional, List, Dict, Any


class CandidateTreeWidget(QWidget):
    """Candidate hypothesis tree allowing inspection and manual candidate selection."""

    candidate_selected = Signal(str)        # (pipeline_id)
    rerun_requested = Signal(str, int)      # (pipeline_id, start_stage_id)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(4, 4, 4, 4)

        header = QLabel("CANDIDATE PIPELINES HIERARCHY")
        header.setStyleSheet("color: #9bb0cf; font-weight: bold; font-size: 11px;")
        self.layout.addWidget(header)

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Candidate Pipeline", "Score", "Modulation", "Baud", "FEC", "CRC"])
        self.tree.itemClicked.connect(self._on_item_clicked)
        self.layout.addWidget(self.tree)

        # Action bar
        btn_bar = QHBoxLayout()
        self.btn_select = QPushButton("SELECT AS ACTIVE")
        self.btn_select.clicked.connect(self._on_select_clicked)
        btn_bar.addWidget(self.btn_select)

        self.btn_rerun = QPushButton("RE-RUN FROM SYNC")
        self.btn_rerun.clicked.connect(self._on_rerun_clicked)
        btn_bar.addWidget(self.btn_rerun)

        self.layout.addLayout(btn_bar)

    def set_candidates(self, candidates: List[Dict[str, Any]]):
        """Populates candidate rows."""
        self.tree.clear()
        for cand in candidates:
            pid = str(cand.get("pipeline_id", "cand"))
            score = float(cand.get("score", cand.get("pipeline_score", 0.0)))
            mod = str(cand.get("modulation", "QPSK"))
            baud = str(cand.get("symbol_rate", 9600.0))
            fec = str(cand.get("fec", cand.get("fec_scheme", "None")))
            crc = "PASS" if cand.get("crc_passed", True) else "FAIL"

            item = QTreeWidgetItem([pid, f"{score:.4f}", mod, baud, fec, crc])
            item.setData(0, 100, pid)
            self.tree.addTopLevelItem(item)

        if self.tree.topLevelItemCount() > 0:
            self.tree.setCurrentItem(self.tree.topLevelItem(0))

    def _on_item_clicked(self, item, col):
        pid = item.data(0, 100)
        if pid:
            self.candidate_selected.emit(pid)

    def _on_select_clicked(self):
        curr = self.tree.currentItem()
        if curr:
            pid = curr.data(0, 100)
            if pid:
                self.candidate_selected.emit(pid)

    def _on_rerun_clicked(self):
        curr = self.tree.currentItem()
        if curr:
            pid = curr.data(0, 100)
            if pid:
                self.rerun_requested.emit(pid, 6)
