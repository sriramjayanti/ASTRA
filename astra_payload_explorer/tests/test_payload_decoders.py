import pytest
import numpy as np
from astra_payload_explorer.src.payload_decoders import (
    decode_payload_views,
    decode_text_representations,
    detect_file_signatures,
    decode_base64
)
from astra_payload_explorer.src.byte_alignment import bytes_to_bits

def test_hi_hello_utf8_decoding():
    # "hi hello" in UTF-8
    text = "hi hello"
    payload_bytes = text.encode("utf-8") # b'hi hello' -> 8 bytes = 64 bits
    payload_bits = bytes_to_bits(payload_bytes)

    views = decode_payload_views(payload_bits)

    assert views.byte_aligned is True
    assert views.length_bits == 64
    assert views.length_bytes == 8
    assert views.hex_str == "68692068656c6c6f"
    assert views.byte_array == list(payload_bytes)

    # Check representations
    utf8_rep = views.representations.get("utf8")
    assert utf8_rep is not None
    assert utf8_rep["valid"] is True
    assert utf8_rep["value"] == "hi hello"
    assert utf8_rep["printable_fraction"] == 1.0

    ascii_rep = views.representations.get("ascii")
    assert ascii_rep is not None
    assert ascii_rep["valid"] is True
    assert ascii_rep["value"] == "hi hello"

def test_binary_non_text_payload_preserved():
    # Random binary bytes that are not valid ASCII text
    raw_b = bytes([0x00, 0xFF, 0x88, 0x12, 0xFE, 0xAA])
    payload_bits = bytes_to_bits(raw_b)

    views = decode_payload_views(payload_bits)

    assert views.byte_aligned is True
    assert views.length_bits == 48
    assert views.hex_str == "00ff8812feaa"
    assert views.byte_array == [0x00, 0xFF, 0x88, 0x12, 0xFE, 0xAA]

    # UTF-8 decoding should be marked as not valid or have replacement chars, but must NOT crash or destroy raw bits
    utf8_rep = views.representations.get("utf8")
    assert utf8_rep is not None
    assert len(views.raw_bits) == 48

def test_non_byte_aligned_payload():
    # 13 bits: 1 0 1 0 1 0 1 0 1 1 1 1 0
    bits = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0], dtype=np.uint8)
    views = decode_payload_views(bits)

    assert views.length_bits == 13
    assert views.byte_aligned is False
    assert views.length_bytes == 1 # 13 // 8
    assert views.remainder_bits == 5 # 13 % 8
    assert len(views.raw_bits) == 13
    # byte array contains packed bytes (padded MSB-first)
    assert len(views.byte_array) == 2

def test_detect_file_signatures():
    # PNG Magic Header: 89 50 4E 47 0D 0A 1A 0A
    png_header = bytes([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00])
    sigs = detect_file_signatures(png_header)
    assert "PNG" in sigs
    assert sigs["PNG"]["type"] == "image/png"

    # PDF Magic: 25 50 44 46 (%PDF)
    pdf_header = b"%PDF-1.4 header content"
    sigs_pdf = detect_file_signatures(pdf_header)
    assert "PDF" in sigs_pdf

def test_decode_base64():
    raw_b = b"ASTRA_TEST"
    b64_rep = decode_base64(raw_b)
    assert b64_rep["valid"] is True
    assert b64_rep["value"] == "QVNUUkFfVEVTVA=="
