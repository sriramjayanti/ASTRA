"""
ASTRA Bitstream Rolling Entropy Plot.
Visualizes entropy elevation profile over bitstream to identify structured headers vs payload.
"""

from PySide6.QtWidgets import QWidget, QVBoxLayout
import pyqtgraph as pg
import numpy as np
from typing import Optional

from ..theme.theme_manager import ThemeManager


class EntropyPlotWidget(QWidget):
    """Rolling bitstream entropy viewer."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.tm = ThemeManager.get_instance()
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(2, 2, 2, 2)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground(self.tm.get_color("background_panel"))
        self.plot_widget.showGrid(x=True, y=True, alpha=0.2)
        self.plot_widget.setLabel('bottom', "Bit Index", color=self.tm.get_color_hex("text_secondary"))
        self.plot_widget.setLabel('left', "Entropy (Bits)", color=self.tm.get_color_hex("text_secondary"))
        self.plot_widget.setYRange(0.0, 1.05)

        pen = pg.mkPen(color=self.tm.get_color_hex("secondary_violet"), width=2.0)
        self.curve = self.plot_widget.plot(name="Rolling Entropy", pen=pen)

        # Baseline threshold line at 0.5
        self.ref_line = pg.InfiniteLine(pos=0.5, angle=0, pen=pg.mkPen("#4a4e69", style=pg.QtCore.Qt.DashLine))
        self.plot_widget.addItem(self.ref_line)

        self.layout.addWidget(self.plot_widget)

    def set_entropy(self, entropy_values: np.ndarray, bit_indices: Optional[np.ndarray] = None):
        if entropy_values is None or len(entropy_values) == 0:
            self.curve.setData([], [])
            return

        x = bit_indices if bit_indices is not None else np.arange(len(entropy_values))
        self.curve.setData(x, entropy_values)
