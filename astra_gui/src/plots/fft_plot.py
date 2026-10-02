"""
ASTRA FFT and Power Spectral Density Plot.
PyQtGraph-based frequency spectrum analyzer with peak markers and carrier offset indicators.
"""

from PySide6.QtWidgets import QWidget, QVBoxLayout
import pyqtgraph as pg
import numpy as np
from typing import Optional

from ..theme.theme_manager import ThemeManager


class FFTPlotWidget(QWidget):
    """Scientific spectrum and PSD analyzer widget."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.tm = ThemeManager.get_instance()
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(2, 2, 2, 2)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground(self.tm.get_color("background_panel"))
        self.plot_widget.showGrid(x=True, y=True, alpha=0.2)
        self.plot_widget.setLabel('bottom', "Frequency (kHz)", color=self.tm.get_color_hex("text_secondary"))
        self.plot_widget.setLabel('left', "Power (dB)", color=self.tm.get_color_hex("text_secondary"))

        vb = self.plot_widget.getViewBox()
        if vb is not None:
            vb.setMouseMode(pg.ViewBox.PanMode)
            vb.wheelEvent = lambda ev, axis=None: ev.ignore()

        pen_spec = pg.mkPen(color=self.tm.get_color_hex("primary_cyan"), width=1.5)
        self.spec_curve = self.plot_widget.plot(name="Spectrum", pen=pen_spec)

        # Carrier marker
        self.carrier_line = pg.InfiniteLine(pos=0.0, angle=90, pen=pg.mkPen("#ff0055", width=1.5, style=pg.QtCore.Qt.DashLine))
        self.plot_widget.addItem(self.carrier_line)

        self.layout.addWidget(self.plot_widget)

    def set_data(self, iq_samples: np.ndarray, sample_rate: float = 192000.0, n_fft: int = 2048):
        """Computes FFT and updates PSD curve."""
        if iq_samples is None or len(iq_samples) < 128:
            self.spec_curve.setData([], [])
            return

        windowed = iq_samples[:min(len(iq_samples), n_fft)]
        window = np.hanning(len(windowed))
        fft_res = np.fft.fftshift(np.fft.fft(windowed * window, n=n_fft))
        mag_db = 20 * np.log10(np.abs(fft_res) + 1e-12)
        mag_db -= np.max(mag_db)  # Normalize peak to 0 dB

        freqs = np.fft.fftshift(np.fft.fftfreq(n_fft, d=1.0 / sample_rate)) / 1000.0  # kHz

        self.spec_curve.setData(freqs, mag_db)

    def set_carrier_marker(self, cfo_hz: float):
        """Positions carrier frequency marker."""
        self.carrier_line.setValue(cfo_hz / 1000.0)
