"""
Unit tests for ASTRA SignalJourneyTimeline.
Tests:
- TEST 8: Stage navigation
- TEST 9: Timeline playback
- TEST 10: Timeline pause
- TEST 11: Timeline replay
- TEST 12: Speed controls
"""

import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import pytest
from PySide6.QtWidgets import QApplication

from astra_gui.src.widgets.timeline import SignalJourneyTimeline


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_8_to_12_timeline_playback_controls(qapp):
    timeline = SignalJourneyTimeline()

    # TEST 8: Stage navigation
    nav_stages = []
    timeline.stage_changed.connect(lambda sid: nav_stages.append(sid))

    timeline.next_stage()
    assert timeline.current_stage == 2
    assert 2 in nav_stages

    timeline.prev_stage()
    assert timeline.current_stage == 1

    # TEST 9 & 10: Timeline playback & pause
    assert timeline.is_playing is False
    timeline.toggle_playback()
    assert timeline.is_playing is True
    assert timeline.play_timer.isActive() is True

    timeline.toggle_playback()
    assert timeline.is_playing is False
    assert timeline.play_timer.isActive() is False

    # TEST 11: Replay
    timeline.set_stage(8)
    assert timeline.current_stage == 8
    timeline.replay()
    assert timeline.current_stage == 1

    # TEST 12: Speed controls
    timeline._on_speed_changed("2.0x")
    assert timeline.playback_speed == 2.0
    timeline._on_speed_changed("0.5x")
    assert timeline.playback_speed == 0.5
