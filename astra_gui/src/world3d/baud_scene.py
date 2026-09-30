"""
ASTRA Stage 4: Symbol Rate Pulsing Peaks Scene.
Visualizes candidate baud rate lanes (4800, 9600, 19200) with synchronized temporal pulses.
"""

import numpy as np
import pyqtgraph.opengl as gl
from typing import Dict, Any

from .base_scene import BaseScene
from .geometry import GeometryFactory
from .particles import ParticleSystem


class BaudScene(BaseScene):
    """Stage 4: 3D Baud Rate Candidate Lanes."""

    def __init__(self, view_widget: gl.GLViewWidget):
        super().__init__(view_widget, "BaudScene")
        self.lanes = []
        self.pulse_markers = None

    def initialize(self):
        grid = GeometryFactory.create_grid(size=70.0, spacing=7.0)
        self.items.append(grid)

        # 3 parallel lanes: Y = -12 (4800 baud), Y = 0 (9600 baud - Winner), Y = 12 (19200 baud)
        y_lanes = [-12.0, 0.0, 12.0]
        rates = [4800.0, 9600.0, 19200.0]

        pts_list = []
        colors = []
        for y, rate in zip(y_lanes, rates):
            x = np.linspace(-35.0, 35.0, 64)
            # Winner has higher peak
            z_amp = 12.0 if rate == 9600.0 else 4.0
            z = z_amp * np.sin(x / 4.0) ** 2
            pts = np.column_stack((x, np.full_like(x, y), z))
            col = (0.0, 0.95, 0.6, 0.9) if rate == 9600.0 else (0.4, 0.5, 0.7, 0.5)

            lane_strip = GeometryFactory.create_line_strip(pts, color=col, width=2.5)
            self.items.append(lane_strip)
            self.lanes.append(lane_strip)

            # Peak marker
            pts_list.append([0.0, y, z_amp])
            colors.append(col)

        self.pulse_markers = ParticleSystem.create_multi_colored_cloud(
            np.array(pts_list),
            np.array(colors),
            size=14.0
        )
        self.items.append(self.pulse_markers)

    def load_data(self, data: Dict[str, Any]):
        pass
