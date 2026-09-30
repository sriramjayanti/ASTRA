"""
headers.py
Header consistency validation for ASTRA Stage 10.
Supports schema-driven validation for constant fields, sequence counters,
and bounded integer rules across segmented frames.
"""

from typing import Any, Dict, List, Optional, Union
import numpy as np
from .models import HeaderConsistencyResult, EvidenceCheckState


def extract_field_uint(bits: np.ndarray, offset: int, length: int) -> int:
    """Extract unsigned integer from bit array at [offset : offset + length]."""
    if offset + length > len(bits):
        return 0
    val = 0
    for b in bits[offset:offset + length]:
        val = (val << 1) | int(b)
    return val


def validate_constant_field(
    frames: List[np.ndarray],
    field_name: str,
    offset_bits: int,
    length_bits: int,
    expected_value: Optional[int] = None,
) -> HeaderConsistencyResult:
    """Validate that a field maintains a constant value across all frames."""
    if not frames:
        return HeaderConsistencyResult(
            field_name=field_name,
            rule_type="constant",
            check_state=EvidenceCheckState.NOT_TESTED,
        )

    values = [extract_field_uint(f, offset_bits, length_bits) for f in frames if len(f) >= offset_bits + length_bits]
    if not values:
        return HeaderConsistencyResult(
            field_name=field_name,
            rule_type="constant",
            check_state=EvidenceCheckState.NOT_TESTED,
        )

    if expected_value is not None:
        valid_count = sum(1 for v in values if v == expected_value)
    else:
        # Check if all match the first frame's value
        valid_count = sum(1 for v in values if v == values[0])

    consistency = valid_count / len(values)
    check_state = EvidenceCheckState.PASS if consistency >= 0.90 else (
        EvidenceCheckState.FAIL if consistency <= 0.40 else EvidenceCheckState.NOT_TESTED
    )

    return HeaderConsistencyResult(
        field_name=field_name,
        rule_type="constant",
        frames_tested=len(values),
        frames_valid=valid_count,
        consistency_score=float(consistency),
        check_state=check_state,
    )


def validate_monotonic_counter(
    frames: List[np.ndarray],
    field_name: str,
    offset_bits: int,
    length_bits: int,
    wrap_limit: Optional[int] = None,
    max_gap: int = 4,
) -> HeaderConsistencyResult:
    """
    Validate that a sequence counter increments monotonically (+1 progression,
    allowing small gaps or wrap-around).
    """
    if len(frames) < 2:
        return HeaderConsistencyResult(
            field_name=field_name,
            rule_type="monotonic_counter",
            check_state=EvidenceCheckState.NOT_TESTED,
        )

    values = [extract_field_uint(f, offset_bits, length_bits) for f in frames if len(f) >= offset_bits + length_bits]
    if len(values) < 2:
        return HeaderConsistencyResult(
            field_name=field_name,
            rule_type="monotonic_counter",
            check_state=EvidenceCheckState.NOT_TESTED,
        )

    wrap = wrap_limit if wrap_limit is not None else (1 << length_bits)
    valid_transitions = 0
    total_transitions = len(values) - 1

    for i in range(total_transitions):
        v1 = values[i]
        v2 = values[i + 1]
        diff = (v2 - v1) % wrap
        if 1 <= diff <= max_gap:
            valid_transitions += 1

    score = valid_transitions / total_transitions
    check_state = EvidenceCheckState.PASS if score >= 0.80 else (
        EvidenceCheckState.FAIL if score <= 0.30 else EvidenceCheckState.NOT_TESTED
    )

    return HeaderConsistencyResult(
        field_name=field_name,
        rule_type="monotonic_counter",
        frames_tested=len(values),
        frames_valid=valid_transitions + 1 if score >= 0.80 else 0,
        consistency_score=float(score),
        check_state=check_state,
    )


def validate_header_schema(
    frames: List[np.ndarray],
    schema: Dict[str, Any],
) -> List[HeaderConsistencyResult]:
    """
    Validate a collection of frames against a configured header schema dict.
    """
    results = []
    if not frames:
        return results

    for field_name, field_spec in schema.items():
        offset = field_spec.get("offset_bits", 0)
        length = field_spec.get("length_bits", 8)
        rule = field_spec.get("rule", "constant")

        if rule == "constant":
            exp_val = field_spec.get("expected_value")
            res = validate_constant_field(frames, field_name, offset, length, exp_val)
            results.append(res)
        elif rule == "monotonic_counter":
            wrap = field_spec.get("wrap_limit")
            res = validate_monotonic_counter(frames, field_name, offset, length, wrap)
            results.append(res)
        elif rule == "bounded_integer":
            min_v = field_spec.get("min_value", 0)
            max_v = field_spec.get("max_value", (1 << length) - 1)
            values = [extract_field_uint(f, offset, length) for f in frames if len(f) >= offset + length]
            valid_count = sum(1 for v in values if min_v <= v <= max_v)
            score = valid_count / len(values) if values else 0.0
            results.append(
                HeaderConsistencyResult(
                    field_name=field_name,
                    rule_type="bounded_integer",
                    frames_tested=len(values),
                    frames_valid=valid_count,
                    consistency_score=score,
                    check_state=EvidenceCheckState.PASS if score >= 0.95 else EvidenceCheckState.FAIL,
                )
            )

    return results
