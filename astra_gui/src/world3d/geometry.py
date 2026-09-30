"""
ASTRA 3D Geometry Utilities.
Provides standard 3D visual primitives (grid floors, axes, bounding boxes, ribbons).
"""

import pyqtgraph.opengl as gl
import numpy as np


class GeometryFactory:
    """Creates reusable PyQtGraph OpenGL geometry items."""

    @staticmethod
    def create_grid(size: float = 80.0, spacing: float = 5.0) -> gl.GLGridItem:
        grid = gl.GLGridItem()
        grid.setSize(size, size)
        grid.setSpacing(spacing, spacing)
        grid.setColor((0.15, 0.20, 0.28, 0.4))
        return grid

    @staticmethod
    def create_axes(size: float = 20.0) -> gl.GLAxisItem:
        axes = gl.GLAxisItem()
        axes.setSize(size, size, size)
        return axes

    @staticmethod
    def create_line_strip(points: np.ndarray, color=(0.0, 0.82, 1.0, 0.9), width: float = 2.0) -> gl.GLLinePlotItem:
        """Creates continuous 3D line from Nx3 array."""
        return gl.GLLinePlotItem(pos=points, color=color, width=width, antialias=True, mode='line_strip')

    @staticmethod
    def create_box(center, size, color=(0.6, 0.2, 0.8, 0.8)) -> gl.GLBoxItem:
        box = gl.GLBoxItem(color=color)
        box.setSize(*size)
        box.translate(center[0] - size[0]/2, center[1] - size[1]/2, center[2] - size[2]/2)
        return box
