"""
Unit tests for ASTRA 3D Signal World and SceneManager.
Tests:
- TEST 13: Scene transition across Stages 1 to 15
- TEST 14: OpenGL scene cleanup without memory leaks
- TEST 15: Camera reset and viewpoint presets
"""

import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import pytest
import pyqtgraph.opengl as gl
from PySide6.QtWidgets import QApplication

from astra_gui.src.world3d.scene_manager import SignalWorldSceneManager
from astra_gui.src.world3d.camera_controller import CameraController


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_13_and_14_scene_transitions_and_cleanup(qapp):
    gl_view = gl.GLViewWidget()
    sm = SignalWorldSceneManager(gl_view)

    # Initial state should be Stage 1 (Raw Signal Scene)
    assert sm.active_stage_id == 1
    assert sm.active_scene.name == "RawSignalScene"
    assert sm.active_scene.is_active is True

    # Transition sequentially through Stages 1 to 15
    for stage_id in range(1, 16):
        sm.switch_to_stage(stage_id)
        assert sm.active_stage_id == stage_id
        assert sm.active_scene is not None
        assert sm.active_scene.is_active is True

    # Test update tick
    sm.update(0.016)

    # Cleanup / dispose
    for scene in sm.scenes.values():
        scene.dispose()
        assert len(scene.items) == 0


def test_15_camera_controller(qapp):
    gl_view = gl.GLViewWidget()
    cam = CameraController(gl_view)

    cam.reset_view()
    cam.set_top_view()
    cam.set_front_view()
    cam.set_tunnel_view()
    cam.cinematic_step(0.016)
