"""
Unit tests for ASTRA GUIStateStore.
Tests:
- TEST 4: State store update and notifications
- TEST 20: AUTO mode handling
- TEST 21: EXPERT mode handling and user overrides
- TEST 29: Reduced motion mode toggle
"""

import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import pytest
import numpy as np
from PySide6.QtWidgets import QApplication

from astra_gui.src.app.state_store import GUIStateStore


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_4_state_store_capture_and_stages(qapp):
    store = GUIStateStore.get_instance()

    # Test loading capture
    samples = (np.random.randn(2048) + 1j * np.random.randn(2048)).astype(np.complex64)
    store.set_capture("test_signal.iq", samples, sample_rate=192000.0)

    assert store.capture_name == "test_signal.iq"
    assert store.sample_count == 2048
    assert store.duration_s > 0.0
    assert store.decimated_iq is not None

    # Test stage selection
    store.set_active_stage(6)
    assert store.active_stage_id == 6
    assert store.timeline_stage == 6


def test_20_and_21_auto_and_expert_modes(qapp):
    store = GUIStateStore.get_instance()

    # AUTO mode
    store.set_mode("AUTO")
    assert store.mode == "AUTO"

    # EXPERT mode
    store.set_mode("EXPERT")
    assert store.mode == "EXPERT"

    # User override in expert mode
    store.set_user_override("modulation", "16QAM")
    assert store.user_overrides["modulation"] == "16QAM"


def test_29_reduced_motion(qapp):
    store = GUIStateStore.get_instance()
    store.reduced_motion = True
    assert store.reduced_motion is True
    store.reduced_motion = False
    assert store.reduced_motion is False
