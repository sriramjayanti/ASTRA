"""
Test suite for physical source signal and window alignment validation.
"""

import pytest
from astra_fusion.src.models import SourceAlignmentError
from astra_fusion.src.validation import validate_source_alignment


def test_3_source_alignment_works():
    """TEST 3: Matching source IDs and window ranges pass validation."""
    validate_source_alignment(
        signal_id_1d="sig_001", start_1d=0, end_1d=2048,
        signal_id_2d="sig_001", start_2d=0, end_2d=2048,
    )
    # None parameters should also pass cleanly
    validate_source_alignment(
        signal_id_1d=None, start_1d=None, end_1d=None,
        signal_id_2d=None, start_2d=None, end_2d=None,
    )


def test_4_unrelated_source_windows_rejected():
    """TEST 4: Mismatched signal IDs or offsets raise SourceAlignmentError."""
    # Signal ID mismatch
    with pytest.raises(SourceAlignmentError):
        validate_source_alignment(
            signal_id_1d="sig_001", start_1d=0, end_1d=2048,
            signal_id_2d="sig_002", start_2d=0, end_2d=2048,
        )

    # Start window mismatch
    with pytest.raises(SourceAlignmentError):
        validate_source_alignment(
            signal_id_1d="sig_001", start_1d=0, end_1d=2048,
            signal_id_2d="sig_001", start_2d=1024, end_2d=2048,
        )

    # End window mismatch
    with pytest.raises(SourceAlignmentError):
        validate_source_alignment(
            signal_id_1d="sig_001", start_1d=0, end_1d=2048,
            signal_id_2d="sig_001", start_2d=0, end_2d=4096,
        )
