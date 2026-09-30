"""
payload_decoders.py
Multi-representation payload decoders: raw bits, hex, bytes, UTF-8, ASCII, Base64, and file magic byte signatures.
"""

from typing import Dict, List, Optional, Tuple, Any
import base64
import numpy as np

from .models import PayloadViews
from .byte_alignment import bits_to_bytes, bits_to_hex_str


FILE_MAGIC_SIGNATURES: Dict[str, Dict[str, Any]] = {
    "PNG": {"magic": bytes.fromhex("89504E470D0A1A0A"), "type": "image/png"},
    "JPEG": {"magic": bytes.fromhex("FFD8FFE0"), "type": "image/jpeg"},
    "PDF": {"magic": bytes.fromhex("25504446"), "type": "application/pdf"},
    "ZIP": {"magic": bytes.fromhex("504B0304"), "type": "application/zip"},
    "GZIP": {"magic": bytes.fromhex("1F8B"), "type": "application/gzip"},
    "ELF": {"magic": bytes.fromhex("7F454C46"), "type": "application/x-executable"},
}


def detect_file_signatures(data_bytes: bytes) -> Dict[str, Dict[str, Any]]:
    """Check whether byte array matches any known file header signatures."""
    matched = {}
    for sig_name, sig_info in FILE_MAGIC_SIGNATURES.items():
        if data_bytes.startswith(sig_info["magic"]):
            matched[sig_name] = sig_info
    return matched


def decode_base64(data_bytes: bytes) -> Dict[str, Any]:
    """Return base64 representation of raw bytes."""
    try:
        b64_str = base64.b64encode(data_bytes).decode("ascii")
        return {"valid": True, "value": b64_str}
    except Exception as e:
        return {"valid": False, "value": None, "error": str(e)}


def decode_text_representations(data_bytes: bytes) -> Dict[str, Any]:
    """
    Attempt UTF-8 and ASCII decoding on raw bytes.
    Computes printable characters fraction and valid flags.
    """
    reps = {}

    # UTF-8
    try:
        utf8_str = data_bytes.decode("utf-8")
        printable_chars = set(range(0x20, 0x7F)) | {0x09, 0x0A, 0x0D}
        p_count = sum(1 for b in data_bytes if b in printable_chars)
        p_frac = float(p_count / max(1, len(data_bytes)))
        reps["utf8"] = {
            "valid": True,
            "value": utf8_str,
            "printable_fraction": p_frac,
            "replacement_character_count": 0
        }
    except UnicodeDecodeError:
        try:
            lossy = data_bytes.decode("utf-8", errors="replace")
            rep_count = lossy.count("\ufffd")
            reps["utf8"] = {
                "valid": False,
                "value": lossy,
                "printable_fraction": 0.0,
                "replacement_character_count": rep_count
            }
        except Exception:
            reps["utf8"] = {"valid": False, "value": None}

    # ASCII
    try:
        ascii_str = data_bytes.decode("ascii")
        reps["ascii"] = {
            "valid": True,
            "value": ascii_str,
            "printable_fraction": 1.0
        }
    except UnicodeDecodeError:
        printable_chars = set(range(0x20, 0x7F)) | {0x09, 0x0A, 0x0D}
        ascii_lossy = "".join(chr(b) if b in printable_chars else "." for b in data_bytes)
        reps["ascii"] = {
            "valid": False,
            "value": ascii_lossy,
            "printable_fraction": float(sum(1 for b in data_bytes if b in printable_chars) / max(1, len(data_bytes)))
        }

    return reps


def decode_payload_views(payload_bits: np.ndarray) -> PayloadViews:
    """
    Construct multi-view representations of raw payload bits without modifying the original data.
    """
    n_bits = len(payload_bits)
    if n_bits == 0:
        return PayloadViews(
            length_bits=0,
            length_bytes=0,
            hex_str="",
            raw_bits="",
            byte_array=[],
            byte_aligned=True,
            remainder_bits=0,
            representations={"utf8": {"valid": True, "value": ""}, "ascii": {"valid": True, "value": ""}}
        )

    is_aligned = (n_bits % 8 == 0)
    raw_b = bits_to_bytes(payload_bits)
    hex_str = bits_to_hex_str(payload_bits)
    raw_bits_str = "".join(str(b) for b in payload_bits)
    b_array = list(raw_b)

    # Decode representations
    reps = decode_text_representations(raw_b)
    reps["base64"] = decode_base64(raw_b)
    file_sigs = detect_file_signatures(raw_b)

    # Shannon Entropy calculation on bytes
    entropy = 0.0
    if len(b_array) > 0:
        counts = np.bincount(b_array, minlength=256)
        probs = counts[counts > 0] / len(b_array)
        entropy = -float(np.sum(probs * np.log2(probs)))

    return PayloadViews(
        length_bits=n_bits,
        length_bytes=n_bits // 8,
        hex_str=hex_str,
        raw_bits=raw_bits_str,
        byte_array=b_array,
        byte_aligned=is_aligned,
        remainder_bits=n_bits % 8,
        representations=reps,
        entropy_shannon=entropy,
        file_signatures=file_sigs
    )
