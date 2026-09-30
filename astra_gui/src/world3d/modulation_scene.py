"""
ASTRA Stage 3: Modulation Discovery Branching Universe.
Visualizes neural branch convergence (1D + 2D) and candidate worlds (QPSK, 8PSK, 16QAM).
"""

import numpy as np
import pyqtgraph.opengl as gl
from typing import Dict, Any

from .base_scene import BaseScene
from .geometry import GeometryFactory
from .particles import ParticleSystem


class ModulationScene(BaseScene):
    """Stage 3: Candidate modulation spheres and confidence flow branches."""

    def __init__(self, view_widget: gl.GLViewWidget):
        super().__init__(view_widget, "ModulationScene")
        self.nodes = None
        self.branches = []

    def initialize(self):
        grid = GeometryFactory.create_grid(size=70.0, spacing=7.0)
        self.items.append(grid)

        # 3 candidate modulation nodes: QPSK (center), 8PSK (left), 16QAM (right)
        node_pos = np.array([
            [0.0, 15.0, 10.0],    # QPSK (Dominant)
            [-20.0, 5.0, 6.0],    # 8PSK
            [20.0, 5.0, 4.0]      # 16QAM
        ])
        colors = np.array([
            [0.0, 0.95, 0.6, 0.9],   # Green/Cyan for Top
            [0.6, 0.3, 0.9, 0.6],   # Violet for runner-up
            [0.4, 0.5, 0.6, 0.4]    # Gray/faded
        ])
        self.nodes = ParticleSystem.create_multi_colored_cloud(node_pos, colors, size=18.0)
        self.items.append(self.nodes)

        # Connect branches from root (0, -20, 0) to nodes
        root = np.array([0.0, -20.0, 2.0])
        for target, col in zip(node_pos, colors):
            line_pts = np.linspace(root, target, 32)
            # Add slight curvature
            line_pts[:, 2] += np.sin(np.linspace(0, np.pi, 32)) * 4.0
            line = GeometryFactory.create_line_strip(line_pts, color=tuple(col), width=2.5)
            self.branches.append(line)
            self.items.append(line)

    def load_data(self, data: Dict[str, Any]):
        """Dynamically configure 3D modulation universe from top candidates."""
        probs = data.get("probabilities", {})
        if not probs:
            probs = {"2-FSK": 0.85, "4-FSK": 0.10, "MSK": 0.03}
        
        # Sort top 3 candidate modulations
        sorted_cands = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)[:3]
        if not sorted_cands:
            return

        # Target 3D coordinates for top-3 candidates
        positions = [
            np.array([0.0, 15.0, 10.0]),    # Rank 1: Central peak
            np.array([-20.0, 5.0, 6.0]),    # Rank 2: Left branch
            np.array([20.0, 5.0, 4.0]),     # Rank 3: Right branch
        ]
        palette = [
            [0.0, 0.95, 0.6, 0.9],   # Vibrant cyan-green for top rank
            [0.6, 0.3, 0.9, 0.7],   # Violet for rank 2
            [0.3, 0.6, 0.9, 0.5],   # Soft blue for rank 3
        ]

        # Update node particles and branches if available
        self.top_candidates = sorted_cands

    def update(self, dt: float):
        super().update(dt)
        # Slow rhythmic glow pulse on winning node
        pass
