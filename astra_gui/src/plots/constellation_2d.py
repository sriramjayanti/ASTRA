"""
ASTRA 2D Constellation Plot.
PyQtGraph scatter visualizer for I/Q symbols, ideal cluster centroids, and EVM rings.
"""

from PySide6.QtWidgets import QWidget, QVBoxLayout
import pyqtgraph as pg
import numpy as np
from typing import Optional, List

from ..theme.theme_manager import ThemeManager


class Constellation2DWidget(QWidget):
    """High-precision 2D I/Q constellation diagram."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.tm = ThemeManager.get_instance()
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(2, 2, 2, 2)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground(self.tm.get_color("background_panel"))
        self.plot_widget.showGrid(x=True, y=True, alpha=0.3)
        self.plot_widget.setAspectLocked(True)
        self.plot_widget.setLabel('bottom', "In-Phase (I)", color=self.tm.get_color_hex("text_secondary"))
        self.plot_widget.setLabel('left', "Quadrature (Q)", color=self.tm.get_color_hex("text_secondary"))
        self.plot_widget.setXRange(-2.5, 2.5)
        self.plot_widget.setYRange(-2.5, 2.5)

        vb = self.plot_widget.getViewBox()
        if vb is not None:
            vb.setMouseMode(pg.ViewBox.PanMode)
            vb.wheelEvent = lambda ev, axis=None: ev.ignore()

        # Received symbols scatter
        self.symbols_scatter = pg.ScatterPlotItem(
            size=5,
            pen=pg.mkPen(None),
            brush=pg.mkBrush(0, 210, 255, 140)
        )
        self.plot_widget.addItem(self.symbols_scatter)

        # Ideal reference symbols
        self.ideal_scatter = pg.ScatterPlotItem(
            size=12,
            pen=pg.mkPen("#ffffff", width=1.5),
            brush=pg.mkBrush(255, 0, 85, 200),
            symbol='+'
        )
        self.plot_widget.addItem(self.ideal_scatter)

        self.layout.addWidget(self.plot_widget)

    def set_symbols(self, symbols: np.ndarray, max_points: int = 4096):
        """Displays received symbols scatter plot with decimation."""
        if symbols is None or len(symbols) == 0:
            self.symbols_scatter.setData(pos=[])
            return

        if len(symbols) > max_points:
            indices = np.random.choice(len(symbols), max_points, replace=False)
            sub_symbols = symbols[indices]
        else:
            sub_symbols = symbols

        points = [{'pos': (float(np.real(s)), float(np.imag(s)))} for s in sub_symbols]
        self.symbols_scatter.setData(points)

    def set_ideal_points(self, ideal_points: List[complex]):
        """Renders ideal constellation reference grid (e.g. QPSK, 16QAM)."""
        if not ideal_points:
            self.ideal_scatter.setData(pos=[])
            return

        pts = [{'pos': (float(np.real(p)), float(np.imag(p)))} for p in ideal_points]
        self.ideal_scatter.setData(pts)
