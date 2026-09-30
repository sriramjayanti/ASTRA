"""
ASTRA Time-Domain Plot.
PyQtGraph-based real-time time-series visualizer for I/Q waveforms and envelope.
"""

from PySide6.QtWidgets import QWidget, QVBoxLayout
import pyqtgraph as pg
import numpy as np
from typing import Optional

from ..theme.theme_manager import ThemeManager


class TimePlotWidget(QWidget):
    """High-performance 2D I/Q time-domain waveform viewer."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.tm = ThemeManager.get_instance()
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(2, 2, 2, 2)

        # Configure PyQtGraph widget
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground(self.tm.get_color("background_panel"))
        self.plot_widget.showGrid(x=True, y=True, alpha=0.2)
        self.plot_widget.setLabel('bottom', "Sample Index", color=self.tm.get_color_hex("text_secondary"))
        self.plot_widget.setLabel('left', "Amplitude", color=self.tm.get_color_hex("text_secondary"))

        # Curves
        pen_i = pg.mkPen(color=self.tm.get_color_hex("primary_cyan"), width=1.5)
        pen_q = pg.mkPen(color=self.tm.get_color_hex("secondary_violet"), width=1.5)
        self.curve_i = self.plot_widget.plot(name="In-Phase (I)", pen=pen_i)
        self.curve_q = self.plot_widget.plot(name="Quadrature (Q)", pen=pen_q)

        self.layout.addWidget(self.plot_widget)

    def set_data(self, iq_samples: np.ndarray, max_points: int = 4096):
        """Updates waveform display with automatic downsampling."""
        if iq_samples is None or len(iq_samples) == 0:
            self.curve_i.setData([], [])
            self.curve_q.setData([], [])
            return

        n = len(iq_samples)
        if n > max_points:
            step = n // max_points
            samples = iq_samples[::step]
            x = np.arange(0, n, step)
        else:
            samples = iq_samples
            x = np.arange(n)

        self.curve_i.setData(x, np.real(samples))
        self.curve_q.setData(x, np.imag(samples))
