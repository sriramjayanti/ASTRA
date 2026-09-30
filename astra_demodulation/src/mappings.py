"""
mappings.py
Canonical Gray bit mappings and constellation definitions for digital modulations in ASTRA.
Single source of truth for symbol-to-bit and bit-to-symbol relationships (MSB-first).
"""

from typing import Dict, Any, List, Tuple
import numpy as np
from .models import ConstellationDefinition


def get_bpsk_constellation() -> ConstellationDefinition:
    """Canonical BPSK Constellation (1 bit/symbol, Es=1.0)."""
    # 0 -> +1.0, 1 -> -1.0
    points = np.array([1.0 + 0.0j, -1.0 + 0.0j], dtype=np.complex64)
    bit_labels = np.array([[0], [1]], dtype=np.uint8)
    indices = np.array([0, 1], dtype=np.int32)
    return ConstellationDefinition(
        name="BPSK",
        complex_points=points,
        bit_labels=bit_labels,
        symbol_indices=indices,
        average_energy=1.0,
        bits_per_symbol=1
    )


def get_qpsk_constellation() -> ConstellationDefinition:
    """Canonical QPSK Gray Constellation (2 bits/symbol, Es=1.0)."""
    # 00 -> (+1+1j)/sqrt(2)
    # 01 -> (-1+1j)/sqrt(2)
    # 11 -> (-1-1j)/sqrt(2)
    # 10 -> (+1-1j)/sqrt(2)
    scale = 1.0 / np.sqrt(2.0)
    points = np.array([
        (1.0 + 1.0j) * scale,
        (-1.0 + 1.0j) * scale,
        (-1.0 - 1.0j) * scale,
        (1.0 - 1.0j) * scale,
    ], dtype=np.complex64)
    bit_labels = np.array([
        [0, 0],
        [0, 1],
        [1, 1],
        [1, 0]
    ], dtype=np.uint8)
    indices = np.array([0, 1, 2, 3], dtype=np.int32)
    return ConstellationDefinition(
        name="QPSK",
        complex_points=points,
        bit_labels=bit_labels,
        symbol_indices=indices,
        average_energy=1.0,
        bits_per_symbol=2
    )


def get_8psk_constellation() -> ConstellationDefinition:
    """Canonical 8PSK Gray Constellation (3 bits/symbol, Es=1.0)."""
    # Phase increments in pi/4 steps with circular Gray coding
    gray_bits = [
        [0, 0, 0],  # 0 deg
        [0, 0, 1],  # 45 deg
        [0, 1, 1],  # 90 deg
        [0, 1, 0],  # 135 deg
        [1, 1, 0],  # 180 deg
        [1, 1, 1],  # 225 deg
        [1, 0, 1],  # 270 deg
        [1, 0, 0]   # 315 deg
    ]
    angles = np.array([k * np.pi / 4.0 for k in range(8)], dtype=np.float32)
    points = np.exp(1j * angles).astype(np.complex64)
    bit_labels = np.array(gray_bits, dtype=np.uint8)
    indices = np.arange(8, dtype=np.int32)
    return ConstellationDefinition(
        name="8PSK",
        complex_points=points,
        bit_labels=bit_labels,
        symbol_indices=indices,
        average_energy=1.0,
        bits_per_symbol=3
    )


def get_16qam_constellation() -> ConstellationDefinition:
    """Canonical 16-QAM Square Gray Constellation (4 bits/symbol, Es=1.0)."""
    # 2-bit Gray axis: 00 -> -3, 01 -> -1, 11 -> +1, 10 -> +3
    axis_levels = np.array([-3.0, -1.0, 1.0, 3.0], dtype=np.float32)
    axis_bits = np.array([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=np.uint8)

    scale = 1.0 / np.sqrt(10.0)  # average energy = (2*1 + 2*9)/4 = 10 -> sqrt(10)
    points = []
    bit_labels = []

    for i in range(4):
        for q in range(4):
            re = axis_levels[i] * scale
            im = axis_levels[q] * scale
            points.append(re + 1j * im)
            bits = np.concatenate([axis_bits[i], axis_bits[q]])
            bit_labels.append(bits)

    return ConstellationDefinition(
        name="16-QAM",
        complex_points=np.array(points, dtype=np.complex64),
        bit_labels=np.array(bit_labels, dtype=np.uint8),
        symbol_indices=np.arange(16, dtype=np.int32),
        average_energy=1.0,
        bits_per_symbol=4
    )


def get_64qam_constellation() -> ConstellationDefinition:
    """Canonical 64-QAM Square Gray Constellation (6 bits/symbol, Es=1.0)."""
    # 3-bit Gray axis:
    # 000 -> -7, 001 -> -5, 011 -> -3, 010 -> -1
    # 110 -> +1, 111 -> +3, 101 -> +5, 100 -> +7
    axis_levels = np.array([-7.0, -5.0, -3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=np.float32)
    axis_bits = np.array([
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 1],
        [0, 1, 0],
        [1, 1, 0],
        [1, 1, 1],
        [1, 0, 1],
        [1, 0, 0]
    ], dtype=np.uint8)

    scale = 1.0 / np.sqrt(42.0)  # average energy = 42 -> sqrt(42)
    points = []
    bit_labels = []

    for i in range(8):
        for q in range(8):
            re = axis_levels[i] * scale
            im = axis_levels[q] * scale
            points.append(re + 1j * im)
            bits = np.concatenate([axis_bits[i], axis_bits[q]])
            bit_labels.append(bits)

    return ConstellationDefinition(
        name="64-QAM",
        complex_points=np.array(points, dtype=np.complex64),
        bit_labels=np.array(bit_labels, dtype=np.uint8),
        symbol_indices=np.arange(64, dtype=np.int32),
        average_energy=1.0,
        bits_per_symbol=6
    )


def get_256qam_constellation() -> ConstellationDefinition:
    """Canonical 256-QAM Square Gray Constellation (8 bits/symbol, Es=1.0)."""
    # 4-bit Gray axis levels: -15 to +15 (step 2)
    axis_levels = np.array([-15.0, -13.0, -11.0, -9.0, -7.0, -5.0, -3.0, -1.0,
                             1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0], dtype=np.float32)
    axis_bits = np.array([
        [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 1], [0, 0, 1, 0],
        [0, 1, 1, 0], [0, 1, 1, 1], [0, 1, 0, 1], [0, 1, 0, 0],
        [1, 1, 0, 0], [1, 1, 0, 1], [1, 1, 1, 1], [1, 1, 1, 0],
        [1, 0, 1, 0], [1, 0, 1, 1], [1, 0, 0, 1], [1, 0, 0, 0]
    ], dtype=np.uint8)

    scale = 1.0 / np.sqrt(170.0)  # average energy = 2/3 * (256 - 1) = 170 -> sqrt(170)
    points = []
    bit_labels = []

    for i in range(16):
        for q in range(16):
            re = axis_levels[i] * scale
            im = axis_levels[q] * scale
            points.append(re + 1j * im)
            bits = np.concatenate([axis_bits[i], axis_bits[q]])
            bit_labels.append(bits)

    return ConstellationDefinition(
        name="256-QAM",
        complex_points=np.array(points, dtype=np.complex64),
        bit_labels=np.array(bit_labels, dtype=np.uint8),
        symbol_indices=np.arange(256, dtype=np.int32),
        average_energy=1.0,
        bits_per_symbol=8
    )


# Lookup dictionary for all supported linear digital modulations
CANONICAL_CONSTELLATIONS: Dict[str, ConstellationDefinition] = {
    "BPSK": get_bpsk_constellation(),
    "QPSK": get_qpsk_constellation(),
    "8PSK": get_8psk_constellation(),
    "8-PSK": get_8psk_constellation(),
    "16-QAM": get_16qam_constellation(),
    "16QAM": get_16qam_constellation(),
    "64-QAM": get_64qam_constellation(),
    "64QAM": get_64qam_constellation(),
    "256-QAM": get_256qam_constellation(),
    "256QAM": get_256qam_constellation(),
}


def get_constellation(modulation: str) -> ConstellationDefinition:
    """Retrieve canonical constellation definition for a modulation scheme."""
    mod_clean = modulation.strip().upper()
    if mod_clean in CANONICAL_CONSTELLATIONS:
        return CANONICAL_CONSTELLATIONS[mod_clean]
    
    # Handle synonyms
    if "BPSK" in mod_clean:
        return CANONICAL_CONSTELLATIONS["BPSK"]
    elif "8PSK" in mod_clean:
        return CANONICAL_CONSTELLATIONS["8PSK"]
    elif "QPSK" in mod_clean or "4PSK" in mod_clean:
        return CANONICAL_CONSTELLATIONS["QPSK"]
    elif "16QAM" in mod_clean or "16-QAM" in mod_clean:
        return CANONICAL_CONSTELLATIONS["16-QAM"]
    elif "64QAM" in mod_clean or "64-QAM" in mod_clean:
        return CANONICAL_CONSTELLATIONS["64-QAM"]
    elif "256QAM" in mod_clean or "256-QAM" in mod_clean:
        return CANONICAL_CONSTELLATIONS["256-QAM"]
    elif "MSK" in mod_clean:
        return CANONICAL_CONSTELLATIONS["QPSK"]
    elif "DQPSK" in mod_clean:
        return CANONICAL_CONSTELLATIONS["QPSK"]
    else:
        raise ValueError(f"No canonical constellation defined for modulation '{modulation}'")
