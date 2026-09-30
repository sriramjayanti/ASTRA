"""
Header Generator and Field Serializer for ASTRA Framing Engine.
Builds deterministic, structured header bitstreams with exact field-level ground truth tracking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import numpy as np

from ..payload.models import PayloadRecord


def int_to_bits(value: int, width: int, bit_order: str = "msb_first") -> np.ndarray:
    """Convert an unsigned integer into a 1D NumPy uint8 bit array of exact bit width.

    Args:
        value: Unsigned integer to convert.
        width: Exact bit width for the field.
        bit_order: Bit ordering convention (only 'msb_first' supported).

    Returns:
        1D np.ndarray of uint8 bits (0 and 1).

    Raises:
        ValueError: If value is negative or cannot fit in `width` bits.
    """
    if bit_order != "msb_first":
        raise ValueError(f"Unsupported bit order '{bit_order}'. ASTRA requires 'msb_first'.")
    if not isinstance(value, int):
        raise TypeError(f"Value must be an integer, got {type(value).__name__}")
    if width < 1:
        raise ValueError(f"Bit width must be >= 1, got {width}")
    
    max_val = (1 << width) - 1
    if not (0 <= value <= max_val):
        raise ValueError(
            f"Value {value} cannot fit in {width} bits (valid unsigned range: 0 to {max_val})"
        )

    return np.array(
        [(value >> (width - 1 - i)) & 1 for i in range(width)],
        dtype=np.uint8,
    )


def bits_to_int(bits: np.ndarray, bit_order: str = "msb_first") -> int:
    """Convert a 1D NumPy bit array to an unsigned integer.

    Args:
        bits: 1D NumPy array of uint8 bits (0 and 1).
        bit_order: Bit ordering convention (only 'msb_first' supported).

    Returns:
        Unsigned integer value.
    """
    if bit_order != "msb_first":
        raise ValueError(f"Unsupported bit order '{bit_order}'. ASTRA requires 'msb_first'.")
    if not isinstance(bits, np.ndarray):
        bits = np.asarray(bits, dtype=np.uint8)
    if bits.ndim != 1:
        raise ValueError(f"Bits array must be 1D, got shape {bits.shape}")
    
    val = 0
    for b in bits:
        val = (val << 1) | (int(b) & 1)
    return val


@dataclass
class HeaderFieldDef:
    """Definition of a single header field."""
    name: str
    bits: int
    value: int | None = None
    source: str | None = None    # 'auto_increment', 'payload_length', 'constant', etc.
    unit: str = "bits"           # For payload_length: 'bits' or 'bytes'


DEFAULT_HEADER_SCHEMA: dict[str, Any] = {
    "schema_name": "astra_default_v1",
    "fields": {
        "version": {
            "bits": 4,
            "value": 1,
        },
        "frame_type": {
            "bits": 4,
            "value": 0,
        },
        "sequence_number": {
            "bits": 16,
            "source": "auto_increment",
        },
        "payload_length": {
            "bits": 24,
            "source": "payload_length",
            "unit": "bits",
        },
        "flags": {
            "bits": 8,
            "value": 0,
        },
    },
}


class HeaderGenerator:
    """Constructs structured binary headers from schema definitions."""

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize HeaderGenerator with schema configuration."""
        self.config = config or DEFAULT_HEADER_SCHEMA
        self.schema_name = self.config.get("schema_name", "astra_default_v1")
        self.fields_cfg = self.config.get("fields", DEFAULT_HEADER_SCHEMA["fields"])
        
        # Parse fields
        self.fields: list[HeaderFieldDef] = []
        for fname, fprops in self.fields_cfg.items():
            fbits = int(fprops.get("bits", 8))
            fval = fprops.get("value")
            fsrc = fprops.get("source")
            funit = fprops.get("unit", "bits")
            self.fields.append(
                HeaderFieldDef(
                    name=fname,
                    bits=fbits,
                    value=fval,
                    source=fsrc,
                    unit=funit,
                )
            )

    @property
    def total_header_bits(self) -> int:
        """Total bit length of the header."""
        return sum(f.bits for f in self.fields)

    def build_header(
        self,
        payload_record: PayloadRecord,
        sequence_number: int,
        custom_field_values: dict[str, int] | None = None,
    ) -> tuple[np.ndarray, dict[str, int], dict[str, dict[str, int]]]:
        """Construct the header bitstream for a payload and sequence number.

        Args:
            payload_record: Source PayloadRecord to extract metadata from.
            sequence_number: Frame sequence number.
            custom_field_values: Optional explicit overrides for field values.

        Returns:
            tuple (header_bits, resolved_field_values, field_offsets)
        """
        custom_vals = custom_field_values or {}
        bit_chunks: list[np.ndarray] = []
        resolved_fields: dict[str, int] = {}
        field_offsets: dict[str, dict[str, int]] = {}

        current_offset = 0

        for fdef in self.fields:
            fname = fdef.name
            fwidth = fdef.bits

            # Resolve field value
            if fname in custom_vals:
                val = custom_vals[fname]
            elif fdef.source == "auto_increment" or fname == "sequence_number":
                # Handle sequence wrap-around
                max_seq = (1 << fwidth)
                val = sequence_number % max_seq
            elif fdef.source == "payload_length" or fname == "payload_length":
                if fdef.unit == "bytes":
                    val = payload_record.byte_length
                else:
                    val = payload_record.bit_length
            elif fdef.value is not None:
                val = int(fdef.value)
            else:
                val = 0

            # Convert to bits and validate range
            field_bits = int_to_bits(val, width=fwidth, bit_order="msb_first")
            bit_chunks.append(field_bits)
            resolved_fields[fname] = val

            field_offsets[fname] = {
                "start": current_offset,
                "length": fwidth,
                "value": val,
            }
            current_offset += fwidth

        header_bits = np.concatenate(bit_chunks) if bit_chunks else np.empty(0, dtype=np.uint8)
        return header_bits, resolved_fields, field_offsets
