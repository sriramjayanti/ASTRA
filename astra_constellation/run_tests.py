"""
ASTRA Constellation Analysis Engine Test Runner.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print("=" * 70)
    print("ASTRA Constellation Analysis Engine - Unit Test Runner")
    print("=" * 70)

    loader = unittest.TestLoader()
    suite = loader.discover(
        start_dir=str(Path(__file__).resolve().parent / "tests"),
        pattern="test_*.py"
    )

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "=" * 70)
    print(f"Tests run: {result.testsRun}, Failures: {len(result.failures)}, Errors: {len(result.errors)}")
    print("=" * 70)

    sys.exit(0 if result.wasSuccessful() else 1)
