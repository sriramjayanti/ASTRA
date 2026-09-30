"""
ASTRA Payload / Bitstream Generator module.
Synthetic Engine 1 for Automated Signal Analysis & Recovery Assistant.
"""

from .models import (
    PayloadRecord,
    bytes_to_bits,
    bits_to_bytes,
    hex_to_bits,
    bits_to_hex,
    text_to_bits,
    calculate_entropy,
)
from .validators import (
    ValidationError,
    validate_payload_bits,
    validate_pattern,
    validate_probability,
    validate_hex_string,
    validate_payload_record,
    validate_config,
)
from .serializers import (
    save_payload,
    load_payload,
    save_batch,
)
from .generator import PayloadGenerator

__all__ = [
    "PayloadGenerator",
    "PayloadRecord",
    "bytes_to_bits",
    "bits_to_bytes",
    "hex_to_bits",
    "bits_to_hex",
    "text_to_bits",
    "calculate_entropy",
    "ValidationError",
    "validate_payload_bits",
    "validate_pattern",
    "validate_probability",
    "validate_hex_string",
    "validate_payload_record",
    "validate_config",
    "save_payload",
    "load_payload",
    "save_batch",
]
