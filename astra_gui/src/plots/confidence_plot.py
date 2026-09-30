"""
ASTRA Confidence Breakdown & Ranking Margin Plot.
PyQtGraph bar chart displaying decomposed confidence contributions and top-K pipeline margins.
"""

from PySide6.QtWidgets import QWidget, QVBoxLayout
import pyqtgraph as pg
import numpy as np
from typing import Dict, Any, Optional

from ..theme.theme_manager import ThemeManager


class ConfidencePlotWidget(QWidget):
    """Visualizer for Stage 15 multi-component confidence decomposition."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.tm = ThemeManager.get_instance()
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(2, 2, 2, 2)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground(self.tm.get_color("background_panel"))
        self.plot_widget.showGrid(x=False, y=True, alpha=0.2)
        self.plot_widget.setYRange(0.0, 1.05)
        self.plot_widget.setLabel('left', "Confidence Score", color=self.tm.get_color_hex("text_secondary"))

        self.bar_item = pg.BarGraphItem(x=[], height=[], width=0.6, brush=pg.mkBrush(self.tm.get_color_hex("primary_cyan")))
        self.plot_widget.addItem(self.bar_item)

        self.layout.addWidget(self.plot_widget)

    def set_breakdown(self, breakdown_dict: Dict[str, float]):
        """Sets components dictionary (e.g. {'Mod': 0.94, 'Sync': 0.93, 'FEC': 0.99, ...})."""
        if not breakdown_dict:
            self.bar_item.setOpts(x=[], height=[])
            return

        labels = list(breakdown_dict.keys())
        values = [float(breakdown_dict[k]) for k in labels]
        x = np.arange(len(labels))

        # Color based on value
        brushes = []
        for v in values:
            if v >= 0.85:
                brushes.append(pg.mkBrush("#00f59b"))
            elif v >= 0.60:
                brushes.append(pg.mkBrush("#00d2ff"))
            elif v >= 0.35:
                brushes.append(pg.mkBrush("#ffbe0b"))
            else:
                brushes.append(pg.mkBrush("#6c757d"))

        self.bar_item.setOpts(x=x, height=values, brushes=brushes)
        axis = self.plot_widget.getAxis('bottom')
        axis.setTicks([list(enumerate(labels))])
