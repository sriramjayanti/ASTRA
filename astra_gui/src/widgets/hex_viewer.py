"""
ASTRA Hex & ASCII Payload Viewer.
Professional hex dump display showing memory offset, hex byte octets, and printable ASCII translation.
"""

from PySide6.QtWidgets import QWidget, QVBoxLayout, QTextEdit, QLabel
from typing import Optional


class HexViewerWidget(QWidget):
    """Hex dump viewer for recovered payload bytes."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(4, 4, 4, 4)

        self.info_lbl = QLabel("Hex Payload Dump & ASCII Reconstruction:")
        self.info_lbl.setStyleSheet("color: #9bb0cf; font-size: 11px; font-weight: bold;")
        self.layout.addWidget(self.info_lbl)

        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        self.text_edit.setStyleSheet("""
            QTextEdit {
                background-color: #0a0c10;
                color: #00f59b;
                font-family: Consolas, monospace;
                font-size: 12px;
                border: 1px solid #263043;
                border-radius: 4px;
            }
        """)
        self.layout.addWidget(self.text_edit)

    def set_bytes(self, raw_bytes: bytes, max_len: int = 1024):
        """Formats bytes into canonical hex dump rows: OFFSET | 16 HEX BYTES | ASCII."""
        if not raw_bytes:
            self.text_edit.setPlainText("")
            return

        b = raw_bytes[:max_len]
        lines = []
        for offset in range(0, len(b), 16):
            chunk = b[offset:offset+16]
            hex_part = " ".join(f"{byte:02X}" for byte in chunk)
            ascii_part = "".join(chr(byte) if 32 <= byte <= 126 else "." for byte in chunk)
            lines.append(f"{offset:06X}  {hex_part:<48}  |{ascii_part}|")

        self.text_edit.setPlainText("\n".join(lines))
