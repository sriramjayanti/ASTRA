"""
ASTRA Stage 5: Candidate Hypothesis Universe.
Renders candidate combinations (Modulation x Baud) in 3D orbit around the central signal axis.
"""

import numpy as np
import pyqtgraph.opengl as gl
from typing import Dict, Any

from .base_scene import BaseScene
from .geometry import GeometryFactory
from .particles import ParticleSystem


class CandidateScene(BaseScene):
    """Stage 5: Multi-hypothesis 3D orbit."""

    def __init__(self, view_widget: gl.GLViewWidget):
        super().__init__(view_widget, "CandidateScene")
        self.candidate_nodes = None
        self.orbit_rings = []

    def initialize(self):
        grid = GeometryFactory.create_grid(size=60.0, spacing=6.0)
        self.items.append(grid)

        # Orbit ring 1 (radius 18)
        theta = np.linspace(0, 2*np.pi, 64)
        r1 = 18.0
        ring1_pts = np.column_stack((r1 * np.cos(theta), r1 * np.sin(theta), np.zeros_like(theta)))
        ring1 = GeometryFactory.create_line_strip(ring1_pts, color=(0.15, 0.35, 0.55, 0.5), width=1.0)
        self.items.append(ring1)
        self.orbit_rings.append(ring1)

        # 4 Candidates placed on orbit:
        # Cand 1: QPSK / 9600 (Winner)
        # Cand 2: 8PSK / 9600
        # Cand 3: QPSK / 4800
        # Cand 4: 16QAM / 19200
        pos = np.array([
            [r1 * np.cos(0.0), r1 * np.sin(0.0), 3.0],
            [r1 * np.cos(np.pi/2), r1 * np.sin(np.pi/2), 0.0],
            [r1 * np.cos(np.pi), r1 * np.sin(np.pi), -2.0],
            [r1 * np.cos(3*np.pi/2), r1 * np.sin(3*np.pi/2), -4.0]
        ])
        colors = np.array([
            [0.0, 0.95, 0.6, 1.0],   # Bright Cyan/Green Winner
            [0.6, 0.3, 0.9, 0.7],   # Violet Runner-up
            [0.0, 0.7, 0.9, 0.5],   # Faded Cyan
            [0.4, 0.4, 0.5, 0.3]    # Dim
        ])
        self.candidate_nodes = ParticleSystem.create_multi_colored_cloud(pos, colors, size=16.0)
        self.items.append(self.candidate_nodes)

    def load_data(self, data: Dict[str, Any]):
        pass

    def update(self, dt: float):
        super().update(dt)
        # Slow orbit rotation
        if self.is_active and self.candidate_nodes is not None:
            # Rotate nodes
            pass
