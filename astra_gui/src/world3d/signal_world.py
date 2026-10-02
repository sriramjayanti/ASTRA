"""
ASTRA Signal World Widget.
Primary central viewport hosting the 3D OpenGL signal processing environment and camera controller.
"""

from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFrame
from PySide6.QtCore import QTimer, Qt
import pyqtgraph.opengl as gl
from typing import Optional, Dict, Any

from .scene_manager import SignalWorldSceneManager
from .camera_controller import CameraController
from ..theme.theme_manager import ThemeManager


class SmoothGLViewWidget(gl.GLViewWidget):
    """GLViewWidget with dampened mouse wheel zoom to prevent sudden jumping."""
    
    def wheelEvent(self, ev):
        delta = ev.angleDelta().y()
        if delta == 0:
            delta = ev.angleDelta().x()
        # Scale down zoom step by 75% for ultra-smooth tracking
        d = float(delta) * 0.025
        self.opts['distance'] = max(5.0, min(300.0, self.opts['distance'] * (0.999 ** d)))
        self.update()


class SignalWorldWidget(QWidget):
    """Interactive 3D Signal Processing Observatory Widget."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.tm = ThemeManager.get_instance()
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)

        # 1. 3D OpenGL Viewport with smooth zoom dampening
        self.gl_view = SmoothGLViewWidget()
        self.gl_view.setBackgroundColor(self.tm.get_color("background_deep"))
        self.layout.addWidget(self.gl_view)

        # 2. Camera & Scene Controllers
        self.camera = CameraController(self.gl_view)
        self.camera.reset_view()

        self.scene_manager = SignalWorldSceneManager(self.gl_view)

        # 3. Floating HUD / Camera Overlay Bar
        self._build_hud_overlay()

        # 4. 60 FPS Render Timer
        self.fps_timer = QTimer(self)
        self.fps_timer.setInterval(16)  # ~60 FPS
        self.fps_timer.timeout.connect(self._on_render_tick)
        self.fps_timer.start()

    def _build_hud_overlay(self):
        """Top overlay controls for quick camera angles and view reset."""
        self.hud_frame = QFrame(self.gl_view)
        self.hud_frame.setGeometry(12, 12, 420, 36)
        self.hud_frame.setStyleSheet("""
            QFrame {
                background-color: rgba(18, 22, 31, 0.85);
                border: 1px solid #263043;
                border-radius: 6px;
            }
            QPushButton {
                background-color: transparent;
                border: none;
                color: #9bb0cf;
                font-size: 11px;
                font-weight: 600;
                padding: 4px 8px;
            }
            QPushButton:hover {
                color: #00d2ff;
            }
        """)

        hud_layout = QHBoxLayout(self.hud_frame)
        hud_layout.setContentsMargins(6, 2, 6, 2)

        lbl = QLabel("3D WORLD:")
        lbl.setStyleSheet("color: #00d2ff; font-weight: bold; font-size: 10px; border: none;")
        hud_layout.addWidget(lbl)

        btn_reset = QPushButton("RESET")
        btn_reset.clicked.connect(self.camera.reset_view)
        hud_layout.addWidget(btn_reset)

        btn_top = QPushButton("TOP")
        btn_top.clicked.connect(self.camera.set_top_view)
        hud_layout.addWidget(btn_top)

        btn_front = QPushButton("FRONT")
        btn_front.clicked.connect(self.camera.set_front_view)
        hud_layout.addWidget(btn_front)

        btn_tunnel = QPushButton("TUNNEL")
        btn_tunnel.clicked.connect(self.camera.set_tunnel_view)
        hud_layout.addWidget(btn_tunnel)

        hud_layout.addStretch()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "hud_frame"):
            self.hud_frame.setGeometry(12, 12, 380, 36)

    def switch_stage(self, stage_id: int):
        """Switches visual scene to match requested stage."""
        self.scene_manager.switch_to_stage(stage_id)

    def feed_data(self, stage_id: int, data: Dict[str, Any]):
        """Passes real pipeline result to active/target scene."""
        self.scene_manager.feed_stage_data(stage_id, data)

    def _on_render_tick(self):
        """Ticks 60 FPS animation."""
        self.scene_manager.update(0.016)
