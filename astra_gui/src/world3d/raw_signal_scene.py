"""
ASTRA Stage 1: Raw Signal 3D Waveform Tunnel.
Visualizes complex IQ samples as a 3D helical waveform tunnel and luminous sample stream.
"""

import numpy as np
import pyqtgraph.opengl as gl
from typing import Dict, Any

from .base_scene import BaseScene
from .geometry import GeometryFactory
from .particles import ParticleSystem


class RawSignalScene(BaseScene):
    """Stage 1: 3D IQ Waveform Tunnel (X: time, Y: I, Z: Q)."""

    def __init__(self, view_widget: gl.GLViewWidget):
        super().__init__(view_widget, "RawSignalScene")
        self.grid = None
        self.line_item = None
        self.particles = None
        self.raw_iq = None

    def initialize(self):
        self.grid = GeometryFactory.create_grid(size=100.0, spacing=10.0)
        self.items.append(self.grid)

        # Initial placeholder line
        pts = np.zeros((100, 3))
        self.line_item = GeometryFactory.create_line_strip(pts, color=(0.0, 0.82, 1.0, 0.9), width=2.0)
        self.items.append(self.line_item)

        # Luminous particle stream
        self.particles = ParticleSystem.create_point_cloud(pts, color=(0.0, 0.95, 1.0, 0.6), size=3.0)
        self.items.append(self.particles)

    def load_data(self, data: Dict[str, Any]):
        iq = data.get("raw_iq")
        if iq is not None and len(iq) > 0:
            self.raw_iq = iq
            # Downsample to 2048 points for high-FPS 3D rendering
            n = min(len(iq), 2048)
            step = len(iq) // n
            samples = iq[::step][:n]

            # X: time spread (-40 to +40), Y: In-Phase, Z: Quadrature
            x = np.linspace(-40.0, 40.0, len(samples))
            scale = 10.0 / (np.max(np.abs(samples)) + 1e-6)
            y = np.real(samples) * scale
            z = np.imag(samples) * scale

            pts = np.column_stack((x, y, z))
            self.line_item.setData(pos=pts)
            self.particles.setData(pos=pts)

    def update(self, dt: float):
        super().update(dt)
        # Slow organic pulse along time axis
        if self.is_active and self.line_item is not None:
            shift = np.sin(self.animation_time * 1.5) * 0.5
            self.line_item.translate(0, 0, shift * 0.05)
