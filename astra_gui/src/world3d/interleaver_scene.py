"""
ASTRA Stage 8: Deinterleaving 3D Grid Permutation Scene.
Visualizes a 16x16 matrix of bit tiles scrambling and resolving into contiguous sequential order.
"""

import numpy as np
import pyqtgraph.opengl as gl
from typing import Dict, Any

from .base_scene import BaseScene
from .geometry import GeometryFactory
from .particles import ParticleSystem


class InterleaverScene(BaseScene):
    """Stage 8: 16x16 Block Deinterleaving Matrix Permutation."""

    def __init__(self, view_widget: gl.GLViewWidget):
        super().__init__(view_widget, "InterleaverScene")
        self.bit_grid = None
        self.rows = 16
        self.cols = 16
        self.is_permuting = False
        self.original_positions = None
        self.permuted_positions = None

    def initialize(self):
        grid = GeometryFactory.create_grid(size=60.0, spacing=6.0)
        self.items.append(grid)

        # 16x16 grid = 256 bits
        n_bits = self.rows * self.cols
        xs = np.linspace(-15.0, 15.0, self.cols)
        ys = np.linspace(-15.0, 15.0, self.rows)
        X, Y = np.meshgrid(xs, ys)

        # Initial deinterleaved target positions
        self.permuted_positions = np.column_stack((X.flatten(), Y.flatten(), np.zeros(n_bits)))

        # Interleaved/scrambled positions (shuffled columns and elevated Z)
        shuffled_idx = np.arange(n_bits).reshape((self.rows, self.cols)).T.flatten()
        self.original_positions = self.permuted_positions[shuffled_idx].copy()
        self.original_positions[:, 2] += np.random.uniform(2.0, 8.0, size=n_bits)

        colors = np.zeros((n_bits, 4))
        for i in range(n_bits):
            colors[i] = [0.0, 0.90, 1.0, 0.85] if (i % 2 == 0) else [0.6, 0.3, 0.9, 0.85]

        self.bit_grid = ParticleSystem.create_multi_colored_cloud(self.original_positions, colors, size=7.0)
        self.items.append(self.bit_grid)

    def load_data(self, data: Dict[str, Any]):
        pass

    def update(self, dt: float):
        super().update(dt)
        if self.is_active and self.bit_grid is not None:
            # Over 2.5 seconds, interpolate from original scrambled to deinterleaved flat matrix
            t_norm = min(1.0, self.animation_time / 2.5)
            # Smooth ease-in-out
            s = 3 * t_norm**2 - 2 * t_norm**3
            cur_pos = (1.0 - s) * self.original_positions + s * self.permuted_positions
            self.bit_grid.setData(pos=cur_pos)
