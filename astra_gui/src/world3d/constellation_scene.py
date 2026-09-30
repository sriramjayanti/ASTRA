"""
ASTRA Stage 7: 3D Constellation Space.
Visualizes locked 4-cluster QPSK point cloud in 3D orbit with cluster centers and EVM rings.
"""

import numpy as np
import pyqtgraph.opengl as gl
from typing import Dict, Any

from .base_scene import BaseScene
from .geometry import GeometryFactory
from .particles import ParticleSystem


class ConstellationScene(BaseScene):
    """Stage 7: 3D Constellation Cloud and Decision Geometry."""

    def __init__(self, view_widget: gl.GLViewWidget):
        super().__init__(view_widget, "ConstellationScene")
        self.points_item = None
        self.centroid_markers = None

    def initialize(self):
        grid = GeometryFactory.create_grid(size=50.0, spacing=5.0)
        self.items.append(grid)

        # Ideal QPSK centroids (+1+1j, -1+1j, -1-1j, +1-1j) * 8.0
        r = 9.0
        centroids = np.array([
            [r, r, 0.0],
            [-r, r, 0.0],
            [-r, -r, 0.0],
            [r, -r, 0.0]
        ])
        centroid_colors = np.array([[1.0, 0.0, 0.35, 1.0]] * 4)
        self.centroid_markers = ParticleSystem.create_multi_colored_cloud(centroids, centroid_colors, size=16.0)
        self.items.append(self.centroid_markers)

        # 1024 locked received symbol points around centroids
        n_pts = 1024
        quad_indices = np.random.choice([0, 1, 2, 3], size=n_pts)
        centers = centroids[quad_indices]
        # Gaussian cluster dispersion
        dispersion = np.random.randn(n_pts, 3) * 0.9
        dispersion[:, 2] *= 0.3  # Flatter in Z
        pts = centers + dispersion

        colors = np.zeros((n_pts, 4))
        for i in range(n_pts):
            colors[i] = [0.0, 0.85, 1.0, 0.75]

        self.points_item = ParticleSystem.create_multi_colored_cloud(pts, colors, size=5.0)
        self.items.append(self.points_item)

    def load_data(self, data: Dict[str, Any]):
        pass
