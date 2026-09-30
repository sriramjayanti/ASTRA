"""
ASTRA Stage 15: Explainability & Reasoning Network Scene.
Visualizes the 3D explanation DAG with the central recovered claim orbited by supporting evidence and contradictions.
"""

import numpy as np
import pyqtgraph.opengl as gl
from typing import Dict, Any

from .base_scene import BaseScene
from .geometry import GeometryFactory
from .particles import ParticleSystem


class ExplainabilityScene(BaseScene):
    """Stage 15: 3D Evidence Network Graph."""

    def __init__(self, view_widget: gl.GLViewWidget):
        super().__init__(view_widget, "ExplainabilityScene")
        self.central_claim_node = None
        self.evidence_nodes = None
        self.connecting_edges = []

    def initialize(self):
        grid = GeometryFactory.create_grid(size=70.0, spacing=7.0)
        self.items.append(grid)

        # Central recovered result node (0, 0, 8) -> Confirmed Emerald Green
        center = np.array([[0.0, 0.0, 8.0]])
        self.central_claim_node = ParticleSystem.create_multi_colored_cloud(
            center,
            np.array([[0.0, 0.96, 0.6, 1.0]]),
            size=26.0
        )
        self.items.append(self.central_claim_node)

        # Orbiting evidence sources:
        # 1. 1D ResNet (Support)
        # 2. 2D CNN (Support)
        # 3. Constellation 4-cluster (Support)
        # 4. Gardner Lock 0.93 (Support)
        # 5. Conv K7 R1/2 (Support)
        # 6. CRC 8/8 PASS (Strong Support)
        # 7. 8PSK Alternative 0.11 (Contradiction/Alternative)
        r = 22.0
        n_ev = 7
        theta = np.linspace(0, 2*np.pi, n_ev, endpoint=False)
        pos = np.column_stack((r * np.cos(theta), r * np.sin(theta), np.full(n_ev, 8.0)))

        colors = np.zeros((n_ev, 4))
        for i in range(n_ev):
            if i == 6:
                colors[i] = [1.0, 0.75, 0.0, 0.9]   # Amber for 8PSK alternative
            elif i == 5:
                colors[i] = [0.0, 1.0, 0.5, 1.0]    # Bright green for CRC 8/8
            else:
                colors[i] = [0.0, 0.85, 1.0, 0.9]   # Cyan support

        self.evidence_nodes = ParticleSystem.create_multi_colored_cloud(pos, colors, size=14.0)
        self.items.append(self.evidence_nodes)

        # Connect each evidence node to center
        for i in range(n_ev):
            edge_pts = np.vstack((center[0], pos[i]))
            edge_color = tuple(colors[i])
            edge = GeometryFactory.create_line_strip(edge_pts, color=edge_color, width=1.8)
            self.items.append(edge)
            self.connecting_edges.append(edge)

    def load_data(self, data: Dict[str, Any]):
        pass

    def update(self, dt: float):
        super().update(dt)
        # Subtle slow orbit rotation
        pass
