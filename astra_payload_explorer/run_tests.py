#!/usr/bin/env python3
"""
Test runner for ASTRA Stage 14 — Header / Payload Explorer.
"""

import sys
import subprocess
from pathlib import Path

def run_tests():
    project_root = Path(__file__).resolve().parent.parent
    tests_dir = Path(__file__).resolve().parent / "tests"

    print("==================================================")
    print("RUNNING ASTRA STAGE 14 (PAYLOAD EXPLORER) TESTS")
    print("==================================================")
    print(f"Directory: {tests_dir}\n")

    cmd = [
        sys.executable,
        "-m",
        "pytest",
        str(tests_dir),
        "-v",
        "--tb=short"
    ]

    result = subprocess.run(cmd, cwd=str(project_root))
    if result.returncode == 0:
        print("\n>>> ALL STAGE 14 TESTS PASSED SUCCESSFULLY! <<<")
    else:
        print(f"\n>>> TESTS FAILED WITH EXIT CODE {result.returncode} <<<")
    return result.returncode

if __name__ == "__main__":
    sys.exit(run_tests())
