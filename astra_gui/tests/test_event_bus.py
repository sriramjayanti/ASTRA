"""
Unit tests for ASTRA VisualizationEventBus.
Tests TEST 5: Event emission and Qt signal delivery.
"""

import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import pytest
from PySide6.QtWidgets import QApplication

from astra_gui.src.core.event_bus import VisualizationEventBus
from astra_gui.src.core.visualization_events import VisualizationEvent, EventType


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_5_event_bus_signals(qapp):
    bus = VisualizationEventBus.get_instance()

    received_events = []
    received_stages = []

    def on_event(evt):
        received_events.append(evt)

    def on_stage_start(sid, sname):
        received_stages.append((sid, sname))

    bus.visualization_event.connect(on_event)
    bus.stage_started.connect(on_stage_start)

    # Emit stage start
    bus.stage_started.emit(6, "Stage 6: Synchronization")
    assert len(received_stages) == 1
    assert received_stages[0] == (6, "Stage 6: Synchronization")

    # Emit visualization event
    evt = VisualizationEvent(
        stage_id=6,
        event_type=EventType.CFO_CORRECTED,
        payload={"before_hz": 1186.4, "after_hz": 12.1}
    )
    bus.emit_event(evt)
    assert len(received_events) == 1
    assert received_events[0].event_type == EventType.CFO_CORRECTED
    assert received_events[0].payload["after_hz"] == 12.1
