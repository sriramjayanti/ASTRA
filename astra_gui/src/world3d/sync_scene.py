"""
ASTRA Stage 6: Synchronization Scene.
Visualizes rotating constellation entering CFO stabilization chamber and locking into place.
"""

import numpy as np
import pyqtgraph.opengl as gl
from typing import Dict, Any

from .base_scene import BaseScene
from .geometry import GeometryFactory
from .particles import ParticleSystem


class SyncScene(BaseScene):
    """Stage 6: Costas Loop & Gardner Timing Recovery stabilization chamber."""

    def __init__(self, view_widget: gl.GLViewWidget):
        super().__init__(view_widget, "SyncScene")
        self.points_item = None
        self.chamber_ring = None
        self.cfo_hz = 1186.4
        self.residual_hz = 12.1
        self.rotation_phase = 0.0

    def initialize(self):
        grid = GeometryFactory.create_grid(size=60.0, spacing=6.0)
        self.items.append(grid)

        # Stabilization chamber ring
        theta = np.linspace(0, 2*np.pi, 64)
        r = 15.0
        ring_pts = np.column_stack((r * np.cos(theta), r * np.sin(theta), np.zeros_like(theta)))
        self.chamber_ring = GeometryFactory.create_line_strip(ring_pts, color=(0.0, 0.95, 1.0, 0.7), width=2.0)
        self.items.append(self.chamber_ring)

        # QPSK 4 clusters rotating initially
        n_pts = 512
        quadrants = np.random.choice([1+1j, -1+1j, -1-1j, 1-1j], size=n_pts)
        noise = 0.15 * (np.random.randn(n_pts) + 1j * np.random.randn(n_pts))
        self.base_symbols = (quadrants + noise) * 8.0

        x = np.real(self.base_symbols)
        y = np.imag(self.base_symbols)
        z = np.zeros(n_pts)
        pts = np.column_stack((x, y, z))

        self.points_item = ParticleSystem.create_point_cloud(pts, color=(0.0, 0.85, 1.0, 0.8), size=5.0)
        self.items.append(self.points_item)

    def load_data(self, data: Dict[str, Any]):
        self.cfo_hz = data.get("before_hz", 1186.4)
        self.residual_hz = data.get("after_hz", 12.1)

    def update(self, dt: float):
        super().update(dt)
        if self.is_active and self.points_item is not None:
            # Over 4 seconds, rotation slows from initial speed to residual lock
            t_norm = min(1.0, self.animation_time / 3.0)
            current_speed = (1.0 - t_norm) * 4.0 + 0.05
            self.rotation_phase += current_speed * dt

            # Rotate symbols
            rot = np.exp(1j * self.rotation_phase)
            rot_syms = self.base_symbols * rot

            pts = np.column_stack((np.real(rot_syms), np.imag(rot_syms), np.zeros(len(rot_syms))))
            self.points_item.setData(pos=pts)
