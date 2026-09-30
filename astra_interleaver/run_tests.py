"""
run_tests.py
Test runner for ASTRA Stage 8 — Interleaver Candidate Testing Engine.
Runs all unit tests and reports pass/fail status and execution metrics.
"""

import unittest
import sys
import os

# Add root workspace to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = loader.discover(
        start_dir=os.path.join(os.path.dirname(__file__), "tests"),
        pattern="test_*.py"
    )
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
