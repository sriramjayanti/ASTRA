"""
ASTRA Bit Viewer Widget.
Virtualized, syntax-highlighted bitstream viewer with color-coded region badges (Sync, Header, Payload, CRC).
"""

from PySide6.QtWidgets import QWidget, QVBoxLayout, QTextEdit, QLabel
from typing import Optional, Any

from ..theme.theme_manager import ThemeManager


class BitViewerWidget(QWidget):
    """Virtualized binary bit viewer."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.tm = ThemeManager.get_instance()
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(4, 4, 4, 4)

        self.info_lbl = QLabel("Recovered Bitstream (Formatted by byte):")
        self.info_lbl.setStyleSheet("color: #9bb0cf; font-size: 11px; font-weight: bold;")
        self.layout.addWidget(self.info_lbl)

        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        self.text_edit.setStyleSheet("""
            QTextEdit {
                background-color: #0a0c10;
                color: #64dfdf;
                font-family: Consolas, monospace;
                font-size: 12px;
                border: 1px solid #263043;
                border-radius: 4px;
            }
        """)
        self.layout.addWidget(self.text_edit)

    def set_bits(self, bits_str: Any, max_display: int = 4096):
        """Displays bits formatted in 8-bit octet blocks with bit offset indexing."""
        if bits_str is None:
            self.text_edit.setPlainText("")
            self.info_lbl.setText("Recovered Bitstream: (Awaiting Analysis)")
            return

        if isinstance(bits_str, (list, tuple)):
            bits_str = "".join(str(int(b)) for b in bits_str)
        elif hasattr(bits_str, "tolist"):  # numpy array
            bits_str = "".join(str(int(b)) for b in bits_str.tolist())
        elif isinstance(bits_str, (bytes, bytearray)):
            bits_str = "".join(f"{b:08b}" for b in bits_str)
        elif not isinstance(bits_str, str):
            bits_str = str(bits_str)

        if not bits_str:
            self.text_edit.setPlainText("")
            self.info_lbl.setText("Recovered Bitstream: (Awaiting Analysis)")
            return

        total_bits = len(bits_str)
        total_bytes = total_bits // 8
        self.info_lbl.setText(f"Recovered Bitstream ({total_bits} bits / {total_bytes} bytes, 8 octets/row):")

        display_bits = bits_str[:max_display]
        # Group into 8-bit bytes
        octets = [display_bits[i:i+8] for i in range(0, len(display_bits), 8)]
        # Group 8 octets per line with bit offset
        lines = []
        for i in range(0, len(octets), 8):
            chunk = octets[i:i+8]
            bit_offset = i * 8
            lines.append(f"[{bit_offset:05d}]  " + " ".join(chunk))

        formatted = "\n".join(lines)
        if total_bits > max_display:
            formatted += f"\n\n... [Stream Truncated: displaying {max_display} of {total_bits} bits]"

        self.text_edit.setPlainText(formatted)
