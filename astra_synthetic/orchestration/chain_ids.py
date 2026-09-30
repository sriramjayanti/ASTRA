"""
Unified ID Generation and Source-Chain Tracking for ASTRA Orchestration (Engine 8).
Ensures standard, globally consistent naming across all 7 synthetic stages.
"""

from __future__ import annotations


def make_chain_id(index: int) -> str:
    """Format unified source chain ID (e.g. 'chain_00000001')."""
    return f"chain_{index:08d}"


def make_dataset_record_id(index: int) -> str:
    """Format dataset record ID (e.g. 'ds_rec_00000001')."""
    return f"ds_rec_{index:08d}"


def make_payload_id(index: int) -> str:
    """Format payload record ID."""
    return f"payload_{index:08d}"


def make_frame_id(index: int) -> str:
    """Format frame record ID."""
    return f"frame_{index:08d}"


def make_fec_id(index: int) -> str:
    """Format FEC record ID."""
    return f"fec_{index:08d}"


def make_interleaver_id(index: int) -> str:
    """Format interleaver record ID."""
    return f"int_{index:08d}"


def make_modulation_id(index: int) -> str:
    """Format modulation record ID."""
    return f"mod_{index:08d}"


def make_channel_id(index: int) -> str:
    """Format channel record ID."""
    return f"chan_{index:08d}"


def make_capture_id(index: int) -> str:
    """Format capture record ID."""
    return f"cap_{index:08d}"
