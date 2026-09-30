"""
header_parser.py
Profile-aware header schema parsing into typed structured fields.
"""

from typing import List, Dict, Optional, Any
import numpy as np

from .models import ParsedField, FieldDefinition, ProtocolProfile, EvidenceLevel
from .bit_order import bits_to_uint, bits_to_int
from .endian import decode_integer_with_endian, decode_fixed_point
from .byte_alignment import bits_to_hex_str


def parse_field_value(
    header_bits: np.ndarray,
    field_def: FieldDefinition,
    profile_id: str = "known_profile"
) -> ParsedField:
    """Parse an individual field definition from header bits."""
    off = field_def.offset_bits
    width = field_def.width_bits
    n_bits = len(header_bits)

    if off + width > n_bits:
        return ParsedField(
            field_name=field_def.name,
            offset_bits=off,
            width_bits=width,
            raw_bits="",
            raw_hex="",
            decoded_value=None,
            evidence_level=EvidenceLevel.KNOWN,
            source=f"profile_{profile_id}",
            description=f"{field_def.description or ''} (TRUNCATED)",
            valid_constraint=False,
            confidence=0.0
        )

    f_bits = header_bits[off:off + width]
    raw_b_str = "".join(str(b) for b in f_bits)
    raw_hex = bits_to_hex_str(f_bits)
    raw_uint = bits_to_uint(f_bits)

    f_type = field_def.field_type.lower()
    endian = field_def.endianness.lower()
    decoded_val = None
    is_valid = True

    if f_type == "uint":
        decoded_val = decode_integer_with_endian(f_bits, endian=endian, signed=False)
        if field_def.expected_value is not None and decoded_val != field_def.expected_value:
            is_valid = False

    elif f_type == "int":
        decoded_val = decode_integer_with_endian(f_bits, endian=endian, signed=True)

    elif f_type == "boolean":
        decoded_val = bool(f_bits[0] == 1)

    elif f_type == "bitfield":
        sub_flags = {}
        if field_def.subfields:
            for s_name, s_def in field_def.subfields.items():
                s_off = s_def.offset_bits
                s_width = s_def.width_bits
                if s_off + s_width <= width:
                    s_bits = f_bits[s_off:s_off+s_width]
                    if s_def.field_type == "boolean" or s_width == 1:
                        sub_flags[s_name] = bool(s_bits[0] == 1)
                    else:
                        sub_flags[s_name] = bits_to_uint(s_bits)
        else:
            for b_idx in range(width):
                sub_flags[f"FLAG_{b_idx}"] = bool(f_bits[b_idx] == 1)
        decoded_val = sub_flags

    elif f_type == "enum":
        enum_map = field_def.enum_values
        decoded_val = enum_map.get(raw_uint, enum_map.get(str(raw_uint), f"UNKNOWN_VALUE_{raw_uint}"))

    elif f_type == "fixed_point":
        decoded_val = decode_fixed_point(
            f_bits,
            scale=field_def.scale,
            offset=field_def.offset,
            signed=(f_type == "fixed_point_signed"),
            endian=endian
        )

    else:
        decoded_val = raw_hex

    return ParsedField(
        field_name=field_def.name,
        offset_bits=off,
        width_bits=width,
        raw_bits=raw_b_str,
        raw_hex=raw_hex,
        decoded_value=decoded_val,
        raw_unsigned=raw_uint,
        evidence_level=EvidenceLevel.KNOWN,
        source=f"profile_{profile_id}",
        unit=field_def.unit,
        description=field_def.description,
        valid_constraint=is_valid,
        confidence=1.0 if is_valid else 0.5
    )


def parse_header(
    header_bits: np.ndarray,
    profile: ProtocolProfile
) -> Dict[str, ParsedField]:
    """Parse all defined header fields from header bits according to the protocol profile."""
    parsed_dict: Dict[str, ParsedField] = {}
    profile_id = getattr(profile, "profile_id", "known_profile")
    fields = getattr(profile, "header_fields", {})

    if isinstance(fields, dict):
        for name, f_def in fields.items():
            parsed_dict[name] = parse_field_value(header_bits, f_def, profile_id)
    elif isinstance(fields, list):
        for f_def in fields:
            if hasattr(f_def, "name"):
                parsed_dict[f_def.name] = parse_field_value(header_bits, f_def, profile_id)

    return parsed_dict
