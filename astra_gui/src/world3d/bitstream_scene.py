"""
ASTRA Stage 12: Bitstream Periodicity & Frame Length Discovery Scene.
Visualizes continuous recovered bits as a 3D ribbon with recurring 512-bit autocorrelation pulses.
"""

import numpy as np
import pyqtgraph.opengl as gl
from typing import Dict, Any

from .base_scene import BaseScene
from .geometry import GeometryFactory
from .particles import ParticleSystem


class BitstreamScene(BaseScene):
    """Stage 12: 3D Bit Ribbon and Frame Boundary Discovery."""

    def __init__(self, view_widget: gl.GLViewWidget):
        super().__init__(view_widget, "BitstreamScene")
        self.bit_ribbon = None
        self.boundary_markers = []

    def initialize(self):
        grid = GeometryFactory.create_grid(size=70.0, spacing=7.0)
        self.items.append(grid)

        # 3D bit ribbon curving across the scene
        n_bits = 256
        x = np.linspace(-30.0, 30.0, n_bits)
        y = np.sin(x / 6.0) * 8.0
        z = np.zeros(n_bits) + 4.0
        pts = np.column_stack((x, y, z))

        colors = np.zeros((n_bits, 4))
        for i in range(n_bits):
            colors[i] = [0.0, 0.85, 1.0, 0.8] if (i % 2 == 0) else [0.6, 0.3, 0.9, 0.8]

        self.bit_ribbon = ParticleSystem.create_multi_colored_cloud(pts, colors, size=6.0)
        self.items.append(self.bit_ribbon)

        # 512-bit Periodic frame boundary planes (at x = -15, x = 0, x = 15)
        for bx in [-15.0, 0.0, 15.0]:
            b_line = np.array([[bx, -12.0, 0.0], [bx, 12.0, 0.0], [bx, 12.0, 10.0], [bx, -12.0, 10.0], [bx, -12.0, 0.0]])
            b_item = GeometryFactory.create_line_strip(b_line, color=(0.0, 0.95, 0.6, 0.7), width=1.5)
            self.items.append(b_item)
            self.boundary_markers.append(b_item)

    def load_data(self, data: Dict[str, Any]):
        pass
