"""
ASTRA Stage 14: Payload Vault Scene.
Visualizes payload bits transforming into hex matrix (`68 69 20 68 65 6c 6c 6f`) and text representation ('hi hello').
"""

import numpy as np
import pyqtgraph.opengl as gl
from typing import Dict, Any

from .base_scene import BaseScene
from .geometry import GeometryFactory
from .particles import ParticleSystem


class PayloadScene(BaseScene):
    """Stage 14: Payload Byte Reconstruction & Hex Matrix."""

    def __init__(self, view_widget: gl.GLViewWidget):
        super().__init__(view_widget, "PayloadScene")
        self.byte_cubes = None

    def initialize(self):
        grid = GeometryFactory.create_grid(size=60.0, spacing=6.0)
        self.items.append(grid)

        # 8 recovered bytes: 'h', 'i', ' ', 'h', 'e', 'l', 'l', 'o'
        # [0x68, 0x69, 0x20, 0x68, 0x65, 0x6C, 0x6C, 0x6F]
        n_bytes = 8
        xs = np.linspace(-18.0, 18.0, n_bytes)
        ys = np.zeros(n_bytes)
        zs = np.zeros(n_bytes) + 8.0
        pos = np.column_stack((xs, ys, zs))

        colors = np.array([[0.0, 0.96, 0.6, 1.0]] * n_bytes)  # Confirmed emerald green
        self.byte_cubes = ParticleSystem.create_multi_colored_cloud(pos, colors, size=18.0)
        self.items.append(self.byte_cubes)

    def load_data(self, data: Dict[str, Any]):
        pass
