"""
run_tests.py
Standalone test runner for Stage 11 Pipeline Scoring Model.
"""

import sys
import pytest

if __name__ == "__main__":
    retcode = pytest.main(["-v", "astra_pipeline_scorer/tests"])
    sys.exit(retcode)
