"""
run_tests.py
Test runner script for ASTRA Stage 12 Bitstream Intelligence Engine.
"""

import sys
import os
import pytest

if __name__ == "__main__":
    # Ensure current workspace root is in sys.path
    workspace_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if workspace_dir not in sys.path:
        sys.path.insert(0, workspace_dir)

    print("Running Stage 12 Bitstream Intelligence test suite...")
    test_dir = os.path.join(os.path.dirname(__file__), "tests")
    exit_code = pytest.main(["-v", test_dir])
    sys.exit(exit_code)
