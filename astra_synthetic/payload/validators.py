"""
Validation utilities for ASTRA Payload Engine.
Strictly validates inputs, configurations, and generated records.
"""

from __future__ import annotations

import json
import hashlib
from typing import Any
import numpy as np

from .models import PayloadRecord


class ValidationError(ValueError):
    """Raised when payload data, configuration, or generated record fails validation."""
    pass


def validate_payload_bits(bits: np.ndarray, min_bits: int = 1) -> None:
    """Validate that the given array is a valid 1D binary bit array.

    Args:
        bits: NumPy array to validate.
        min_bits: Minimum allowed bit count (default: 1).

    Raises:
        ValidationError: If validation fails.
    """
    if not isinstance(bits, np.ndarray):
        raise ValidationError(f"Payload bits must be a NumPy array, got {type(bits).__name__}")
    
    if bits.ndim != 1:
        raise ValidationError(f"Payload bits must be 1D, got ndim={bits.ndim} (shape={bits.shape})")
    
    if len(bits) < min_bits:
        raise ValidationError(f"Payload bits length ({len(bits)}) is less than minimum allowed ({min_bits})")
    
    # Check that bits contain only 0 and 1
    # For performance on large arrays, use fast numpy checks
    if bits.size > 0:
        unique_vals = np.unique(bits)
        invalid_vals = np.setdiff1d(unique_vals, [0, 1])
        if len(invalid_vals) > 0:
            raise ValidationError(
                f"Payload bits contain invalid values: {invalid_vals.tolist()}. Only 0 and 1 are permitted."
            )


def validate_pattern(pattern: str) -> None:
    """Validate that pattern is a non-empty string consisting only of '0' and '1'.

    Args:
        pattern: Bit pattern string.

    Raises:
        ValidationError: If pattern is empty or contains non-binary characters.
    """
    if not isinstance(pattern, str):
        raise ValidationError(f"Pattern must be a string, got {type(pattern).__name__}")
    
    if len(pattern) == 0:
        raise ValidationError("Pattern string cannot be empty.")
    
    invalid_chars = set(pattern) - {"0", "1"}
    if invalid_chars:
        raise ValidationError(
            f"Pattern '{pattern}' contains invalid characters: {invalid_chars}. Only '0' and '1' allowed."
        )


def validate_probability(prob: float, name: str = "probability") -> None:
    """Validate that a probability value is in [0.0, 1.0].

    Args:
        prob: Probability value.
        name: Name of parameter for error message.

    Raises:
        ValidationError: If prob is not in [0.0, 1.0] or not numeric.
    """
    if not isinstance(prob, (int, float)):
        raise ValidationError(f"{name} must be numeric, got {type(prob).__name__}")
    
    if not (0.0 <= prob <= 1.0):
        raise ValidationError(f"{name} must be between 0.0 and 1.0, got {prob}")


def validate_hex_string(hex_string: str) -> None:
    """Validate that hex_string is a valid non-empty hexadecimal string.

    Args:
        hex_string: Hexadecimal string.

    Raises:
        ValidationError: If hex string is invalid or odd-length.
    """
    if not isinstance(hex_string, str):
        raise ValidationError(f"Hex input must be a string, got {type(hex_string).__name__}")
    
    cleaned = hex_string.strip()
    if cleaned.lower().startswith("0x"):
        cleaned = cleaned[2:]
    cleaned = "".join(cleaned.split())
    
    if len(cleaned) == 0:
        raise ValidationError("Hex string cannot be empty.")
    
    if len(cleaned) % 2 != 0:
        raise ValidationError(f"Hex string length must be even, got {len(cleaned)} characters: '{cleaned}'")
    
    try:
        bytes.fromhex(cleaned)
    except ValueError as e:
        raise ValidationError(f"Invalid hexadecimal string '{hex_string}': {e}") from e


def validate_payload_record(record: PayloadRecord) -> None:
    """Perform comprehensive validation on a generated PayloadRecord.

    Checks:
    - payload_bits contains only 0 and 1
    - bit_length matches len(payload_bits)
    - byte_length matches len(payload_bytes)
    - sha256 matches SHA-256(payload_bytes)
    - metadata is JSON serializable

    Args:
        record: PayloadRecord instance.

    Raises:
        ValidationError: If any consistency check fails.
    """
    if not isinstance(record, PayloadRecord):
        raise ValidationError(f"Expected PayloadRecord, got {type(record).__name__}")
    
    # 1. Validate payload_id
    if not record.payload_id or not isinstance(record.payload_id, str):
        raise ValidationError(f"Invalid payload_id: {record.payload_id}")
    
    # 2. Validate bits
    validate_payload_bits(record.payload_bits)
    if len(record.payload_bits) != record.bit_length:
        raise ValidationError(
            f"Bit length mismatch: record.bit_length={record.bit_length} vs len(payload_bits)={len(record.payload_bits)}"
        )
    
    # 3. Validate bytes length
    if len(record.payload_bytes) != record.byte_length:
        raise ValidationError(
            f"Byte length mismatch: record.byte_length={record.byte_length} vs len(payload_bytes)={len(record.payload_bytes)}"
        )
    
    # 4. Validate SHA-256
    expected_sha256 = hashlib.sha256(record.payload_bytes).hexdigest()
    if record.sha256 != expected_sha256:
        raise ValidationError(
            f"SHA-256 mismatch: recorded {record.sha256} vs computed {expected_sha256}"
        )
    
    # 5. Validate entropy range
    if not (0.0 <= record.entropy_estimate <= 1.0):
        raise ValidationError(
            f"Entropy estimate out of range [0, 1]: {record.entropy_estimate}"
        )
    
    # 6. Validate metadata JSON serializability
    try:
        json.dumps(record.metadata)
    except Exception as e:
        raise ValidationError(f"Metadata is not JSON-serializable: {e}") from e


def validate_config(config: dict[str, Any]) -> None:
    """Validate payload generator configuration dictionary.

    Args:
        config: Configuration dictionary.

    Raises:
        ValidationError: If config structure, length bounds, or probabilities are invalid.
    """
    if not isinstance(config, dict):
        raise ValidationError(f"Configuration must be a dictionary, got {type(config).__name__}")
    
    pg_cfg = config.get("payload_generator", config)
    
    # Validate length settings if present
    length_cfg = pg_cfg.get("length")
    if length_cfg:
        min_bits = length_cfg.get("min_bits", 64)
        max_bits = length_cfg.get("max_bits", 4096)
        if min_bits < 1:
            raise ValidationError(f"min_bits must be >= 1, got {min_bits}")
        if max_bits < min_bits:
            raise ValidationError(f"max_bits ({max_bits}) must be >= min_bits ({min_bits})")
    
    # Validate types distribution if present
    types_cfg = pg_cfg.get("types")
    if types_cfg:
        total_prob = 0.0
        for ptype, pinfo in types_cfg.items():
            if not isinstance(pinfo, dict):
                continue
            if pinfo.get("enabled", True):
                prob = pinfo.get("probability", 0.0)
                validate_probability(prob, f"Type '{ptype}' probability")
                total_prob += prob
                
        if total_prob > 0 and abs(total_prob - 1.0) > 1e-3:
            raise ValidationError(
                f"Enabled payload type probabilities must sum to 1.0 (got {total_prob:.4f})"
            )
