"""
ASTRA Particle System.
GPU-efficient 3D scatter clouds for complex RF samples, symbol clouds, and moving bits.
"""

import pyqtgraph.opengl as gl
import numpy as np


class ParticleSystem:
    """Manages GLScatterPlotItem clouds with dynamic colors and positions."""

    @staticmethod
    def create_point_cloud(
        positions: np.ndarray,
        color=(0.0, 0.85, 1.0, 0.7),
        size: float = 3.0
    ) -> gl.GLScatterPlotItem:
        """Positions: Nx3 numpy array."""
        if positions is None or len(positions) == 0:
            positions = np.zeros((1, 3))
        return gl.GLScatterPlotItem(
            pos=positions,
            color=color,
            size=size,
            pxMode=True
        )

    @staticmethod
    def create_multi_colored_cloud(
        positions: np.ndarray,
        colors: np.ndarray,
        size: float = 4.0
    ) -> gl.GLScatterPlotItem:
        """Positions: Nx3, Colors: Nx4 float RGBA in range [0, 1]."""
        return gl.GLScatterPlotItem(
            pos=positions,
            color=colors,
            size=size,
            pxMode=True
        )
