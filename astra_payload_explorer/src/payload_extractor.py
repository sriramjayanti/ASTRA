"""
payload_extractor.py
Extracts payload bit regions from frames and converts them to PayloadViews.
"""

from typing import Optional, Dict, Any, Tuple
import numpy as np

from .models import PayloadViews
from .payload_decoders import decode_payload_views


def extract_payload_from_frame(
    frame_bits: np.ndarray,
    payload_start: int,
    payload_end: int,
    profile: Optional[Dict[str, Any]] = None
) -> Tuple[np.ndarray, PayloadViews]:
    """
    Isolate exact payload bits from frame and build multi-representation views.
    """
    n_bits = len(frame_bits)
    p_start = max(0, min(payload_start, n_bits))
    p_end = max(p_start, min(payload_end, n_bits))

    payload_bits = frame_bits[p_start:p_end]
    views = decode_payload_views(payload_bits)

    return payload_bits, views
