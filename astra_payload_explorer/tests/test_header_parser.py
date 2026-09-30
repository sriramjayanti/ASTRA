import pytest
import numpy as np
from astra_payload_explorer.src.models import ProtocolProfile, FieldDefinition, EvidenceLevel
from astra_payload_explorer.src.header_parser import parse_header, parse_field_value
from astra_payload_explorer.src.byte_alignment import bytes_to_bits, bits_to_bytes

def test_uint_and_int_field_parsing():
    # 32-bit header: 8-bit uint (value 42), 8-bit signed int (value -5), 16-bit uint LE (value 1000)
    # 42 = 0x2A = 00101010
    # -5 (two's complement 8-bit) = 0xFB = 11111011
    # 1000 = 0x03E8 -> LE is E8 03 = 11101000 00000011
    fields_def = {
        "version": FieldDefinition(name="version", offset_bits=0, width_bits=8, field_type="uint", endianness="big"),
        "temp_signed": FieldDefinition(name="temp_signed", offset_bits=8, width_bits=8, field_type="int", endianness="big"),
        "seq_le": FieldDefinition(name="seq_le", offset_bits=16, width_bits=16, field_type="uint", endianness="little")
    }
    profile = ProtocolProfile(
        profile_id="test_profile",
        frame_length_bits=64,
        sync_length_bits=0,
        header_length_bits=32,
        header_fields=fields_def
    )

    header_bytes = bytes([0x2A, 0xFB, 0xE8, 0x03])
    header_bits = bytes_to_bits(header_bytes)

    parsed = parse_header(header_bits, profile)
    assert len(parsed) == 3
    assert parsed["version"].decoded_value == 42
    assert parsed["version"].evidence_level == EvidenceLevel.KNOWN
    assert parsed["temp_signed"].decoded_value == -5
    assert parsed["seq_le"].decoded_value == 1000

def test_fixed_point_and_enum_parsing():
    # 16-bit fixed point: scale=0.1, offset=-40.0, raw uint=650 -> decoded = 650 * 0.1 - 40 = 25.0
    # 8-bit enum: 0->telemetry, 1->command, 2->ack
    fields_def = {
        "temperature": FieldDefinition(
            name="temperature",
            offset_bits=0,
            width_bits=16,
            field_type="fixed_point",
            scale=0.1,
            offset=-40.0,
            unit="C"
        ),
        "frame_type": FieldDefinition(
            name="frame_type",
            offset_bits=16,
            width_bits=8,
            field_type="enum",
            enum_values={0: "telemetry", 1: "command", 2: "ack"}
        )
    }
    profile = ProtocolProfile(
        profile_id="test_sensor",
        frame_length_bits=64,
        header_length_bits=24,
        header_fields=fields_def
    )

    # 650 = 0x028A. enum=1 -> 0x01
    header_bytes = bytes([0x02, 0x8A, 0x01])
    header_bits = bytes_to_bits(header_bytes)

    parsed = parse_header(header_bits, profile)
    assert pytest.approx(parsed["temperature"].decoded_value, 0.01) == 25.0
    assert parsed["frame_type"].decoded_value == "command"

def test_unknown_enum_value_handling():
    # Unknown enum value should not crash; returns UNKNOWN_VALUE_5
    field_def = FieldDefinition(
        name="frame_type",
        offset_bits=0,
        width_bits=8,
        field_type="enum",
        enum_values={0: "telemetry", 1: "command"}
    )
    header_bits = np.array([0, 0, 0, 0, 0, 1, 0, 1], dtype=np.uint8) # 5
    field = parse_field_value(header_bits, field_def, "test_prof")
    assert field.decoded_value == "UNKNOWN_VALUE_5"
    assert field.raw_unsigned == 5

def test_bitfield_parsing():
    flags_sub = {
        "ACK_REQUIRED": FieldDefinition(name="ACK_REQUIRED", offset_bits=0, width_bits=1, field_type="boolean"),
        "ENCRYPTED": FieldDefinition(name="ENCRYPTED", offset_bits=1, width_bits=1, field_type="boolean"),
        "COMPRESSED": FieldDefinition(name="COMPRESSED", offset_bits=2, width_bits=1, field_type="boolean"),
        "RESERVED": FieldDefinition(name="RESERVED", offset_bits=3, width_bits=5, field_type="uint")
    }
    field_def = FieldDefinition(
        name="flags",
        offset_bits=0,
        width_bits=8,
        field_type="bitfield",
        subfields=flags_sub
    )
    # Flags = 1 0 1 00000 = 0xA0
    header_bits = np.array([1, 0, 1, 0, 0, 0, 0, 0], dtype=np.uint8)
    field = parse_field_value(header_bits, field_def, "test_flags")
    assert field.decoded_value["ACK_REQUIRED"] is True
    assert field.decoded_value["ENCRYPTED"] is False
    assert field.decoded_value["COMPRESSED"] is True
    assert field.decoded_value["RESERVED"] == 0
