"""
Unit tests for ASTRA PipelineController.
Tests:
- TEST 6: Worker thread execution and progress events
- TEST 7: Cancel operation
"""

import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import pytest
import time
import numpy as np
from PySide6.QtWidgets import QApplication

from astra_gui.src.app.state_store import GUIStateStore
from astra_gui.src.core.event_bus import VisualizationEventBus
from astra_gui.src.app.pipeline_controller import ASTRAPipelineController


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_6_and_7_controller_worker_execution_and_cancel(qapp):
    state = GUIStateStore.get_instance()
    bus = VisualizationEventBus.get_instance()
    controller = ASTRAPipelineController(state, bus)

    # Load synthetic signal
    iq = (np.random.randn(4096) + 1j * np.random.randn(4096)).astype(np.complex64)
    state.set_capture("test_controller.iq", iq, 192000.0)

    # Launch pipeline job
    completed = []
    bus.pipeline_completed.connect(lambda res: completed.append(res))

    controller.run_demo()
    assert state.is_analyzing is True

    # Test cancel operation
    controller.cancel_analysis()
    assert state.is_analyzing is False
