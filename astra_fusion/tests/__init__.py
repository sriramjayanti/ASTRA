"""
ASTRA Fusion Tests Package.
"""
import os
import sys

# Ensure root workspace is in sys.path
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
