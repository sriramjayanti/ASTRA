"""
ASTRA Stage 7: Demodulation & Bit Slicing Scene.
Visualizes constellation symbols passing through decision planes into binary bit tokens.
"""

import numpy as np
import pyqtgraph.opengl as gl
from typing import Dict, Any

from .base_scene import BaseScene
from .geometry import GeometryFactory
from .particles import ParticleSystem


class DemodulationScene(BaseScene):
    """Stage 7: Decision Planes & Bit Slicing Projection."""

    def __init__(self, view_widget: gl.GLViewWidget):
        super().__init__(view_widget, "DemodulationScene")
        self.plane_i = None
        self.plane_q = None
        self.bit_tokens = None

    def initialize(self):
        grid = GeometryFactory.create_grid(size=50.0, spacing=5.0)
        self.items.append(grid)

        # I=0 and Q=0 decision cross-planes
        axis_pts_i = np.array([[-25.0, 0.0, 0.0], [25.0, 0.0, 0.0]])
        axis_pts_q = np.array([[0.0, -25.0, 0.0], [0.0, 25.0, 0.0]])
        line_i = GeometryFactory.create_line_strip(axis_pts_i, color=(0.4, 0.6, 0.8, 0.5), width=2.0)
        line_q = GeometryFactory.create_line_strip(axis_pts_q, color=(0.4, 0.6, 0.8, 0.5), width=2.0)
        self.items.extend([line_i, line_q])

        # Bit stream emerging forward along +Z axis
        n_bits = 64
        z = np.linspace(0.0, 30.0, n_bits)
        x = np.sin(z / 2.0) * 4.0
        y = np.cos(z / 2.0) * 4.0
        pos = np.column_stack((x, y, z))
        colors = np.array([[0.0, 0.95, 0.6, 0.9] if (i % 2 == 0) else [0.0, 0.7, 1.0, 0.9] for i in range(n_bits)])

        self.bit_tokens = ParticleSystem.create_multi_colored_cloud(pos, colors, size=6.0)
        self.items.append(self.bit_tokens)

    def load_data(self, data: Dict[str, Any]):
        pass
