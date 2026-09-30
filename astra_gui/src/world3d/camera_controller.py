"""
ASTRA 3D Camera Controller.
Manages interactive perspective, smooth cinematic transitions, and preset viewpoints.
"""

from typing import Tuple, Optional
import numpy as np
from PySide6.QtGui import QVector3D


class CameraController:
    """Controls GLViewWidget camera angle, distance, center, and smooth fly-throughs."""

    def __init__(self, gl_widget):
        self.widget = gl_widget
        self.default_distance = 60.0
        self.default_elevation = 25.0
        self.default_azimuth = 45.0
        self.default_center = QVector3D(0.0, 0.0, 0.0)

    def reset_view(self):
        """Restores default perspective."""
        if hasattr(self.widget, "setCameraPosition"):
            self.widget.setCameraPosition(
                pos=self.default_center,
                distance=self.default_distance,
                elevation=self.default_elevation,
                azimuth=self.default_azimuth
            )

    def set_top_view(self):
        """Top-down orthographic/planar perspective."""
        if hasattr(self.widget, "setCameraPosition"):
            self.widget.setCameraPosition(distance=50.0, elevation=90.0, azimuth=0.0)

    def set_front_view(self):
        """Front planar view."""
        if hasattr(self.widget, "setCameraPosition"):
            self.widget.setCameraPosition(distance=50.0, elevation=0.0, azimuth=0.0)

    def set_tunnel_view(self):
        """Deep perspective looking down the signal tunnel."""
        if hasattr(self.widget, "setCameraPosition"):
            self.widget.setCameraPosition(pos=QVector3D(0.0, 0.0, 50.0), distance=40.0, elevation=10.0, azimuth=0.0)

    def cinematic_step(self, t: float):
        """Smoothly rotates camera during cinematic replay."""
        if hasattr(self.widget, "orbit"):
            self.widget.orbit(0.3, 0.0)
