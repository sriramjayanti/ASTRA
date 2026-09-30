"""
ASTRA Eye Diagram Plot.
Displays synchronized overlapping symbol trajectories illustrating timing jitter and symbol dispersion.
"""

from PySide6.QtWidgets import QWidget, QVBoxLayout
import pyqtgraph as pg
import numpy as np
from typing import Optional

from ..theme.theme_manager import ThemeManager


class EyeDiagramWidget(QWidget):
    """Eye diagram oscilloscope view for timing recovery evaluation."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.tm = ThemeManager.get_instance()
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(2, 2, 2, 2)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground(self.tm.get_color("background_panel"))
        self.plot_widget.showGrid(x=True, y=True, alpha=0.2)
        self.plot_widget.setLabel('bottom', "Symbol Period (T_sym)", color=self.tm.get_color_hex("text_secondary"))
        self.plot_widget.setLabel('left', "Signal Amplitude", color=self.tm.get_color_hex("text_secondary"))

        self.layout.addWidget(self.plot_widget)

    def set_data(self, real_samples: np.ndarray, sps: int = 20, num_traces: int = 64):
        """Generates 2-symbol windowed overlapping traces."""
        self.plot_widget.clear()
        if real_samples is None or len(real_samples) < sps * 4:
            return

        window_len = sps * 2
        t = np.linspace(-1.0, 1.0, window_len)
        pen = pg.mkPen(color=(0, 210, 255, 60), width=1.0)

        max_start = len(real_samples) - window_len
        step = max(1, max_start // num_traces)

        for i in range(0, min(max_start, step * num_traces), step):
            segment = real_samples[i:i + window_len]
            self.plot_widget.plot(t, segment, pen=pen)
