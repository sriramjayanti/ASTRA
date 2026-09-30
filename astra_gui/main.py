"""
ASTRA Desktop Workstation Entry Point.
Run:
    python -m astra_gui.main
or:
    python astra_gui/main.py
"""

import sys
import os

# Add root project path to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from astra_gui.src.app.application import create_application


def main():
    app, window = create_application()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
