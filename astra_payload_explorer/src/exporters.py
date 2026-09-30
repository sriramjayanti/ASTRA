"""
exporters.py
Machine-readable exports (JSON, CSV, Hex Dumps, and Binary Files).
"""

from typing import List, Dict, Any, Optional, Union
import os
import json
import csv
import io

from .models import FrameRecord, PayloadExplorerResult, EvidenceLevel
from .byte_alignment import bits_to_bytes


def generate_hex_dump(data_bytes: Union[bytes, bytearray, List[int]], bytes_per_line: int = 16) -> str:
    """
    Format bytes into standard canonical hex dump with ASCII sidebar:
    0000: 68 69 20 68 65 6c 6c 6f  hi hello
    """
    if isinstance(data_bytes, list):
        data_bytes = bytes(data_bytes)
    elif isinstance(data_bytes, bytearray):
        data_bytes = bytes(data_bytes)

    lines = []
    for i in range(0, len(data_bytes), bytes_per_line):
        chunk = data_bytes[i:i + bytes_per_line]
        hex_str = " ".join(f"{b:02x}" for b in chunk)
        hex_pad = hex_str.ljust(bytes_per_line * 3 - 1)
        ascii_str = "".join(chr(b) if (0x20 <= b <= 0x7E) else "." for b in chunk)
        lines.append(f"{i:04x}: {hex_pad}  |{ascii_str}|")

    return "\n".join(lines)


format_hex_dump = generate_hex_dump  # Alias


def export_to_json(result: PayloadExplorerResult, filepath: Optional[str] = None) -> str:
    """Export PayloadExplorerResult to JSON string or file."""
    json_str = json.dumps(result.to_dict(include_frame_details=True), indent=2)
    if filepath:
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(json_str)
    return json_str


export_result_to_json = export_to_json  # Alias


def export_to_csv(frames: List[FrameRecord], filepath: Optional[str] = None) -> str:
    """Export tabular header fields across frames to CSV string or file."""
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["frame_index", "field_name", "offset_bits", "width_bits", "raw_hex", "decoded_value", "evidence_level", "source"])

    for f_rec in frames:
        fields = f_rec.header_fields if isinstance(f_rec.header_fields, dict) else {f.field_name: f for f in f_rec.header_fields}
        for name, fld in fields.items():
            ev = fld.evidence_level.value if isinstance(fld.evidence_level, EvidenceLevel) else str(fld.evidence_level)
            writer.writerow([
                f_rec.frame_index,
                fld.field_name,
                fld.offset_bits,
                fld.width_bits,
                fld.raw_hex,
                str(fld.decoded_value),
                ev,
                fld.source
            ])

    csv_str = output.getvalue()
    if filepath:
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            f.write(csv_str)
    return csv_str


export_header_fields_to_csv = export_to_csv  # Alias


def export_binary_payload(payload_bytes: bytes, filepath: str):
    """Export raw binary payload bytes to file."""
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    with open(filepath, "wb") as f:
        f.write(payload_bytes)
