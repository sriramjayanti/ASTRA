"""
ASTRA Frame Structure Table.
Displays extracted frames with start offsets, lengths, sequence numbers, CRC validation status, and payload snippets.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QTableWidget, QTableWidgetItem, QHeaderView
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from typing import Optional, List, Dict, Any


class FrameTableWidget(QWidget):
    """Table view of recovered frames."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(2, 2, 2, 2)

        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels([
            "Frame #", "Start Bit", "Length", "CRC Status", "Seq", "Payload Preview"
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)

        self.layout.addWidget(self.table)

    def set_frames(self, frames: Any):
        """Populates frame entries."""
        if isinstance(frames, dict):
            frames = [frames]
        elif not isinstance(frames, list):
            frames = []

        self.table.setRowCount(len(frames))
        for row, f in enumerate(frames):
            f_idx = f.get("frame_idx", f.get("frame_id", f.get("index", row + 1)))
            f_start_bit = f.get("start_bit", f.get("start", f.get("offset", row * 512)))
            f_length = f.get("length_bits", f.get("frame_length", f.get("length", 512)))
            crc_valid = f.get("crc_valid", f.get("crc_passed", f.get("crc", True)))
            f_sequence = f.get("sequence", f.get("seq", row + 1))
            preview_text = str(f.get("payload_preview") or f.get("payload_hex") or f.get("payload") or "DATA FRAME")

            f_num = QTableWidgetItem(str(f_idx))
            f_start = QTableWidgetItem(str(f_start_bit))
            f_len = QTableWidgetItem(str(f_length))
            f_crc = QTableWidgetItem("PASS" if crc_valid else "FAIL")
            f_seq = QTableWidgetItem(str(f_sequence))
            f_preview = QTableWidgetItem(preview_text)

            f_num.setTextAlignment(Qt.AlignCenter)
            f_start.setTextAlignment(Qt.AlignCenter)
            f_len.setTextAlignment(Qt.AlignCenter)
            f_crc.setTextAlignment(Qt.AlignCenter)
            f_seq.setTextAlignment(Qt.AlignCenter)

            if crc_valid:
                f_crc.setForeground(QColor("#00f5d4"))
            else:
                f_crc.setForeground(QColor("#ff0055"))

            self.table.setItem(row, 0, f_num)
            self.table.setItem(row, 1, f_start)
            self.table.setItem(row, 2, f_len)
            self.table.setItem(row, 3, f_crc)
            self.table.setItem(row, 4, f_seq)
            self.table.setItem(row, 5, f_preview)
