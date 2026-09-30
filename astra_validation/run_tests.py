"""
run_tests.py
Standalone test runner for Stage 10 Validation Engine.
"""

import sys
import pytest

if __name__ == "__main__":
    retcode = pytest.main(["-v", "astra_validation/tests"])
    sys.exit(retcode)
