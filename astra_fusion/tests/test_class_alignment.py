"""
Test suite for class mapping validation and alignment.
"""

import pytest
from astra_fusion.src.models import ClassMappingMismatchError
from astra_fusion.src.validation import validate_class_alignment


def test_1_class_mapping_equality():
    """TEST 1: Identical class mapping passes validation."""
    classes_1d = ["2-FSK", "4-FSK", "BPSK", "QPSK", "8PSK", "16-QAM", "64-QAM", "Unknown"]
    classes_2d = ["2-FSK", "4-FSK", "BPSK", "QPSK", "8PSK", "16-QAM", "64-QAM", "Unknown"]
    # Should not raise
    validate_class_alignment(classes_1d, classes_2d)


def test_2_mismatched_mapping_raises_error():
    """TEST 2: Mismatched class count, order, or name raises ClassMappingMismatchError."""
    classes_1d = ["2-FSK", "4-FSK", "BPSK", "QPSK"]
    
    # Mismatched length
    with pytest.raises(ClassMappingMismatchError):
        validate_class_alignment(classes_1d, ["2-FSK", "4-FSK", "BPSK"])

    # Mismatched order
    with pytest.raises(ClassMappingMismatchError):
        validate_class_alignment(classes_1d, ["4-FSK", "2-FSK", "BPSK", "QPSK"])

    # Mismatched name
    with pytest.raises(ClassMappingMismatchError):
        validate_class_alignment(classes_1d, ["2-FSK", "4-FSK", "BPSK", "8PSK"])
