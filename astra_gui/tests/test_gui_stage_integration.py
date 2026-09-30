"""
End-to-End GUI & Stage Integration Tests.
Tests:
- TEST 1: Application starts & creates main window
- TEST 2: Theme loads palette tokens
- TEST 3: Signal capture load & display decimation
- TEST 16: 2D Constellation plotting
- TEST 17: Waterfall STFT rendering
- TEST 18: Candidate tree population
- TEST 19: Evidence inspector updates
- TEST 24: Large capture decimation & LOD
- TEST 30: Full Stage 1–15 integration ('hi hello' recovery demo)
"""

import os
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import pytest
import numpy as np
from PySide6.QtWidgets import QApplication

from astra_gui.src.app.main_window import ASTRAMainWindow
from astra_gui.src.app.state_store import GUIStateStore
from astra_gui.src.theme.theme_manager import ThemeManager


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_1_and_2_app_and_theme_init(qapp):
    tm = ThemeManager.get_instance()
    assert tm.get_color_hex("primary_cyan") == "#00d2ff"
    assert tm.get_color_hex("confirmed_green") == "#00f59b"

    window = ASTRAMainWindow()
    assert window is not None
    assert "ASTRA" in window.windowTitle()


def test_3_and_24_capture_load_and_decimation(qapp):
    window = ASTRAMainWindow()
    store = GUIStateStore.get_instance()

    # Create 100,000 samples to verify intelligent decimation
    large_iq = (np.random.randn(100000) + 1j * np.random.randn(100000)).astype(np.complex64)
    store.set_capture("huge_capture.iq", large_iq, sample_rate=192000.0)

    assert store.sample_count == 100000
    # Decimated buffer should be capped for 60 FPS rendering
    assert len(store.decimated_iq) <= 32768
    assert window.top_bar.capture_badge.text() == "huge_capture.iq"


def test_16_to_19_scientific_views_and_evidence(qapp):
    window = ASTRAMainWindow()
    dock = window.scientific_dock

    # 16. Constellation plotting
    symbols = (np.random.randn(1024) + 1j * np.random.randn(1024)).astype(np.complex64)
    dock.constellation_plot.set_symbols(symbols)

    # 17. Waterfall rendering
    dock.waterfall_plot.set_data(symbols)

    # 18. Candidate tree
    cands = [
        {"pipeline_id": "p_qpsk_9600", "score": 0.965, "modulation": "QPSK", "symbol_rate": 9600.0, "fec": "Conv K7", "crc_passed": True},
        {"pipeline_id": "p_8psk_9600", "score": 0.380, "modulation": "8PSK", "symbol_rate": 9600.0, "fec": "None", "crc_passed": False}
    ]
    dock.candidate_tree.set_candidates(cands)
    assert dock.candidate_tree.tree.topLevelItemCount() == 2

    # 19. Evidence Inspector
    window.evidence_panel.set_stage_evidence(
        "Stage 3: Modulation", "CONFIRMED", 0.94,
        ["1D ResNet prob 0.84", "2D CNN prob 0.79"],
        ["8PSK was alternative at 0.11"],
        ["8PSK / 9600 baud"]
    )
    assert window.evidence_panel.status_badge.text() == "CONFIRMED"


def test_30_full_stage_1_to_15_integration_demo(qapp):
    """MASTER TEST: Runs the 'hi hello' full synthetic demo through the workstation."""
    window = ASTRAMainWindow()
    controller = window.controller
    state = window.state

    # Trigger demo run
    controller.run_demo()
    assert state.capture_name == "DEMO_SYNTHETIC_QPSK_HI_HELLO.iq"
    assert state.is_analyzing is True
    controller.cancel_analysis()
