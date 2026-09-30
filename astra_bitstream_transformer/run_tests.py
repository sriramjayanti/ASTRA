"""
run_tests.py
Test runner script for ASTRA Stage 13 1D CNN + Transformer Bitstream Structure Model.
"""

import sys
import os
import pytest

if __name__ == "__main__":
    workspace_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if workspace_dir not in sys.path:
        sys.path.insert(0, workspace_dir)

    print("Running Stage 13 Bitstream Structure Model test suite...")
    test_dir = os.path.join(os.path.dirname(__file__), "tests")
    exit_code = pytest.main(["-v", test_dir])
    sys.exit(exit_code)
