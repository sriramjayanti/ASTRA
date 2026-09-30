"""
ASTRA Waterfall Plot.
PyQtGraph-based 2D STFT spectrogram viewer with customizable scientific colormap.
"""

from PySide6.QtWidgets import QWidget, QVBoxLayout
import pyqtgraph as pg
import numpy as np
from scipy import signal
from typing import Optional

from ..theme.theme_manager import ThemeManager


class WaterfallPlotWidget(QWidget):
    """Zoomable 2D STFT spectral waterfall visualizer."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.tm = ThemeManager.get_instance()
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(2, 2, 2, 2)

        # Image view
        self.imv = pg.ImageView()
        self.imv.ui.roiBtn.hide()
        self.imv.ui.menuBtn.hide()

        # Scientific colormap: deep navy -> cyan -> violet -> bright white
        colors = [
            (10, 12, 16),
            (0, 75, 120),
            (0, 210, 255),
            (157, 78, 221),
            (255, 255, 255)
        ]
        pos = np.array([0.0, 0.25, 0.55, 0.85, 1.0])
        cmap = pg.ColorMap(pos, np.array(colors, dtype=np.uint8))
        self.imv.setColorMap(cmap)

        self.layout.addWidget(self.imv)

    def set_data(self, iq_samples: np.ndarray, sample_rate: float = 192000.0, nperseg: int = 256):
        """Computes STFT and renders 2D waterfall image."""
        if iq_samples is None or len(iq_samples) < nperseg:
            return

        f, t, zxx = signal.stft(iq_samples[:min(len(iq_samples), 16384)], fs=sample_rate, nperseg=nperseg)
        # Transpose so time is horizontal or vertical as preferred
        spec_mag = np.abs(np.fft.fftshift(zxx, axes=0))
        spec_db = 20 * np.log10(spec_mag + 1e-9)

        # Normalize [0, 1]
        s_min, s_max = np.min(spec_db), np.max(spec_db)
        if s_max > s_min:
            img = (spec_db - s_min) / (s_max - s_min)
        else:
            img = spec_db

        self.imv.setImage(img.T)
