"""
ASTRA Modulation / Clean IQ Waveform Generation Module.
Synthetic Engine 5 for Automated Signal Analysis & Recovery Assistant.
"""

from .models import ModulationRecord
from .bit_mapping import (
    pad_and_group_bits,
    binary_symbols_to_bits,
    binary_to_gray,
    gray_to_binary,
    QPSK_GRAY_MAP,
    QPSK_GRAY_INVERSE,
    PSK8_GRAY_MAP,
    PSK8_GRAY_INVERSE,
    QAM16_AXIS_GRAY,
    QAM16_AXIS_INVERSE,
    QAM64_AXIS_GRAY,
    QAM64_AXIS_INVERSE,
)
from .filters import root_raised_cosine_filter, apply_rrc_pulse_shaping
from .normalization import calculate_average_power, normalize_signal_power
from .pulse_shaping import compute_evm_rms, extract_matched_symbols
from .psk import PSKModulator
from .qam import QAMModulator
from .fsk import FSKModulator
from .validators import validate_modulation_record, reference_demodulate
from .serializers import save_modulation_record, load_modulation_record, save_modulation_batch
from .generator import ModulationGenerator, BUILTIN_MODULATION_PROFILES

__all__ = [
    "ModulationGenerator",
    "ModulationRecord",
    "BUILTIN_MODULATION_PROFILES",
    "PSKModulator",
    "QAMModulator",
    "FSKModulator",
    "root_raised_cosine_filter",
    "apply_rrc_pulse_shaping",
    "calculate_average_power",
    "normalize_signal_power",
    "compute_evm_rms",
    "extract_matched_symbols",
    "validate_modulation_record",
    "reference_demodulate",
    "save_modulation_record",
    "load_modulation_record",
    "save_modulation_batch",
    "pad_and_group_bits",
    "binary_symbols_to_bits",
    "binary_to_gray",
    "gray_to_binary",
    "QPSK_GRAY_MAP",
    "QPSK_GRAY_INVERSE",
    "PSK8_GRAY_MAP",
    "PSK8_GRAY_INVERSE",
    "QAM16_AXIS_GRAY",
    "QAM16_AXIS_INVERSE",
    "QAM64_AXIS_GRAY",
    "QAM64_AXIS_INVERSE",
]
