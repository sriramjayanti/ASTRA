"""
ASTRA Stage 9: FEC Decoding & Codeword Repair Scene.
Visualizes soft Viterbi trellis / codeword repair where corrupted positions turn from error-red to verified-green.
"""

import numpy as np
import pyqtgraph.opengl as gl
from typing import Dict, Any

from .base_scene import BaseScene
from .geometry import GeometryFactory
from .particles import ParticleSystem


class FECScene(BaseScene):
    """Stage 9: Viterbi Trellis & Bit Error Repair Animation."""

    def __init__(self, view_widget: gl.GLViewWidget):
        super().__init__(view_widget, "FECScene")
        self.codeword_bits = None
        self.trellis_lines = []
        self.n_bits = 64
        self.bit_positions = None
        self.corrupted_indices = [7, 18, 31, 45, 58]

    def initialize(self):
        grid = GeometryFactory.create_grid(size=60.0, spacing=6.0)
        self.items.append(grid)

        # Linear stream of bits along X axis
        x = np.linspace(-25.0, 25.0, self.n_bits)
        y = np.zeros(self.n_bits)
        z = np.zeros(self.n_bits) + 5.0
        self.bit_positions = np.column_stack((x, y, z))

        # Initial colors: most cyan, but corrupted indices are bright red
        self.current_colors = np.zeros((self.n_bits, 4))
        for i in range(self.n_bits):
            if i in self.corrupted_indices:
                self.current_colors[i] = [1.0, 0.0, 0.35, 1.0]  # Red error
            else:
                self.current_colors[i] = [0.0, 0.85, 1.0, 0.8]  # Clean cyan

        self.codeword_bits = ParticleSystem.create_multi_colored_cloud(
            self.bit_positions,
            self.current_colors,
            size=9.0
        )
        self.items.append(self.codeword_bits)

    def load_data(self, data: Dict[str, Any]):
        pass

    def update(self, dt: float):
        super().update(dt)
        if self.is_active and self.codeword_bits is not None:
            # At 1.5 seconds, Viterbi repairs corrupted bits to glowing emerald green
            if self.animation_time > 1.5:
                for idx in self.corrupted_indices:
                    self.current_colors[idx] = [0.0, 0.96, 0.6, 1.0]  # Green repaired
                self.codeword_bits.setData(color=self.current_colors)
