"""
ASTRA Base 3D Scene.
Standard lifecycle and OpenGL item management for all 3D Signal World scenes.
"""

from typing import List, Dict, Any, Optional
import pyqtgraph.opengl as gl


class BaseScene:
    """Abstract base class for all 3D Signal World visualizations."""

    def __init__(self, view_widget: gl.GLViewWidget, name: str = "BaseScene"):
        self.view = view_widget
        self.name = name
        self.items: List[Any] = []
        self.is_active: bool = False
        self.animation_time: float = 0.0

    def initialize(self):
        """Constructs OpenGL items once upon scene creation."""
        pass

    def load_data(self, data: Dict[str, Any]):
        """Consumes real pipeline metrics and sample buffers."""
        pass

    def enter(self):
        """Adds scene items to GLViewWidget when stage becomes active."""
        if not self.is_active:
            for it in self.items:
                self.view.addItem(it)
            self.is_active = True
            self.animation_time = 0.0

    def update(self, dt: float):
        """Advances scene animations."""
        self.animation_time += dt

    def pause(self):
        pass

    def resume(self):
        pass

    def exit(self):
        """Removes items from view without destroying memory."""
        if self.is_active:
            for it in self.items:
                try:
                    self.view.removeItem(it)
                except Exception:
                    pass
            self.is_active = False

    def dispose(self):
        """Fully releases GPU buffers."""
        self.exit()
        self.items.clear()
