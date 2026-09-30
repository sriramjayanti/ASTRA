"""
ASTRA Stage 2: 3D Spectral Terrain & Preprocessing Scene.
Visualizes frequency x time x power surface landscape and DC/filtering transformation.
"""

import numpy as np
import pyqtgraph.opengl as gl
from typing import Dict, Any

from .base_scene import BaseScene
from .geometry import GeometryFactory


class DSPScene(BaseScene):
    """Stage 2: 3D Spectral Landscape Surface."""

    def __init__(self, view_widget: gl.GLViewWidget):
        super().__init__(view_widget, "DSPScene")
        self.surface_item = None
        self.filter_curve = None

    def initialize(self):
        grid = GeometryFactory.create_grid(size=80.0, spacing=8.0)
        self.items.append(grid)

        # 3D surface plot for spectral terrain
        nx, ny = 32, 32
        x = np.linspace(-30, 30, nx)
        y = np.linspace(-30, 30, ny)
        z = np.zeros((nx, ny))
        self.surface_item = gl.GLSurfacePlotItem(x=x, y=y, z=z, shader='shaded', color=(0.1, 0.4, 0.8, 0.8))
        self.items.append(self.surface_item)

        # Matched filter response curve
        pts = np.zeros((64, 3))
        self.filter_curve = GeometryFactory.create_line_strip(pts, color=(0.0, 1.0, 0.7, 0.9), width=3.0)
        self.items.append(self.filter_curve)

    def load_data(self, data: Dict[str, Any]):
        # Generate simulated 3D spectral terrain peaks
        nx, ny = 32, 32
        x = np.linspace(-30, 30, nx)
        y = np.linspace(-30, 30, ny)
        X, Y = np.meshgrid(x, y)
        R = np.sqrt(X**2 + Y**2)
        Z = 8.0 * np.exp(-R**2 / 120.0) + 1.2 * np.sin(X/3.0) * np.cos(Y/3.0)
        self.surface_item.setData(x=x, y=y, z=Z)

        # Filter response
        fx = np.linspace(-25, 25, 64)
        fy = 12.0 / (1.0 + (fx / 8.0)**4)
        fz = np.zeros(64) + 10.0
        self.filter_curve.setData(pos=np.column_stack((fx, fy, fz)))

    def update(self, dt: float):
        super().update(dt)
        if self.is_active and self.surface_item is not None:
            # Subtle spectral breathing
            scale = 1.0 + 0.05 * np.sin(self.animation_time * 2.0)
            self.surface_item.scale(1.0, 1.0, scale)
