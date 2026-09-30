"""
ASTRA Framing / Sync / Header / CRC Generator Module.
Synthetic Engine 2 for Automated Signal Analysis & Recovery Assistant.
"""

from .models import FrameRecord, FrameStreamRecord
from .sync import SyncWordGenerator
from .header import (
    HeaderGenerator,
    HeaderFieldDef,
    int_to_bits,
    bits_to_int,
)
from .crc import (
    CRCProfile,
    CRC_PROFILES,
    compute_crc,
    verify_crc,
    get_crc_profile,
)
from .validators import (
    validate_frame_record,
    parse_known_frame,
)
from .serializers import (
    save_frame,
    load_frame,
    save_frame_batch,
    save_frame_stream,
    load_frame_stream,
)
from .generator import FrameGenerator

__all__ = [
    "FrameGenerator",
    "FrameRecord",
    "FrameStreamRecord",
    "SyncWordGenerator",
    "HeaderGenerator",
    "HeaderFieldDef",
    "int_to_bits",
    "bits_to_int",
    "CRCProfile",
    "CRC_PROFILES",
    "compute_crc",
    "verify_crc",
    "get_crc_profile",
    "validate_frame_record",
    "parse_known_frame",
    "save_frame",
    "load_frame",
    "save_frame_batch",
    "save_frame_stream",
    "load_frame_stream",
]
