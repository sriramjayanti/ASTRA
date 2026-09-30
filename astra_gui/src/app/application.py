"""
ASTRA Desktop Application Launcher.
Initializes Qt application context, OpenGL surface format, high-DPI scaling, and shows MainWindow.
"""

import sys
import os
from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
from PySide6.QtGui import QSurfaceFormat

from .main_window import ASTRAMainWindow


def create_application() -> tuple[QApplication, ASTRAMainWindow]:
    """Initializes PySide6 application with high-DPI and OpenGL settings."""
    # Enable High DPI
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    # Allow pyqtgraph GLViewWidget to manage its optimal OpenGL surface format

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    app.setApplicationName("ASTRA Signal Workstation")
    app.setOrganizationName("ASTRA Team")

    window = ASTRAMainWindow()
    return app, window
