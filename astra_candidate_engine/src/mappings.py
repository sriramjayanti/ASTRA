"""
mappings.py
Explicit, central mapping tables for modulation families, demodulator routing,
and synchronization parameter hints in ASTRA.
"""

from typing import Dict, Any, List

# Explicit mapping of specific modulation schemes to broader modulation families
MODULATION_TO_FAMILY_MAP: Dict[str, str] = {
    # FSK Family
    "2-FSK": "FSK",
    "4-FSK": "FSK",
    "8-FSK": "FSK",
    "2FSK": "FSK",
    "4FSK": "FSK",
    "8FSK": "FSK",
    "FSK": "FSK",
    "GFSK": "FSK",
    "MSK": "FSK",
    "GMSK": "FSK",

    # PSK Family
    "BPSK": "PSK",
    "QPSK": "PSK",
    "8PSK": "PSK",
    "8-PSK": "PSK",
    "OQPSK": "PSK",
    "DQPSK": "PSK",
    "pi/4-DQPSK": "PSK",
    "PI4DQPSK": "PSK",

    # QAM Family
    "16-QAM": "QAM",
    "64-QAM": "QAM",
    "16QAM": "QAM",
    "64QAM": "QAM",
    "256-QAM": "QAM",
    "256QAM": "QAM",
    "QAM": "QAM",

    # Analog
    "AM-DSB": "ANALOG",
    "AM-SSB": "ANALOG",
    "WBFM": "ANALOG",
    "NBFM": "ANALOG",
    "FM": "ANALOG",
    "AM": "ANALOG",

    # Unknown
    "Unknown": "UNKNOWN",
    "UNKNOWN": "UNKNOWN",
    "noise": "NOISE",
    "NOISE": "NOISE",
}

# Supported standard digital demodulators
DEMODULATOR_ROUTING_MAP: Dict[str, Dict[str, Any]] = {
    "FSK": {
        "demodulator_type": "fsk_demodulator",
        "supported": True,
        "detection_mode": "envelope_or_discriminator",
        "output_domain": "bits_and_tones",
    },
    "PSK": {
        "demodulator_type": "psk_demodulator",
        "supported": True,
        "detection_mode": "phase_decision_slicer",
        "output_domain": "hard_and_soft_bits",
    },
    "QAM": {
        "demodulator_type": "qam_demodulator",
        "supported": True,
        "detection_mode": "rectangular_grid_slicer",
        "output_domain": "hard_and_soft_bits",
    },
    "ANALOG": {
        "demodulator_type": "analog_demodulator",
        "supported": False,
        "detection_mode": "envelope_or_fm_demod",
        "output_domain": "audio_waveform",
    },
    "UNKNOWN": {
        "demodulator_type": "custom_or_manual",
        "supported": False,
        "detection_mode": "unspecified",
        "output_domain": "raw_samples",
    },
    "NOISE": {
        "demodulator_type": "none",
        "supported": False,
        "detection_mode": "none",
        "output_domain": "none",
    },
}

# Modulation-Aware Synchronization Routing Hints (for Stage 6)
SYNC_ROUTING_HINTS: Dict[str, Dict[str, Any]] = {
    "FSK": {
        "carrier_recovery": "afc_or_quadrature_discriminator",
        "matched_filter_family": "gaussian_or_rectangular",
        "initial_rolloff_candidates": [0.3, 0.5],
        "timing_recovery_methods": ["early_late", "zero_crossing", "gardner"],
        "costas_order": None,
    },
    "PSK": {
        "carrier_recovery": "costas_loop_or_decision_directed",
        "matched_filter_family": "root_raised_cosine",
        "initial_rolloff_candidates": [0.25, 0.35, 0.5],
        "timing_recovery_methods": ["gardner", "mueller_muller"],
        "costas_order_map": {
            "BPSK": 2,
            "QPSK": 4,
            "8PSK": 8,
            "8-PSK": 8,
            "OQPSK": 4,
        },
    },
    "QAM": {
        "carrier_recovery": "decision_directed_pll",
        "matched_filter_family": "root_raised_cosine",
        "initial_rolloff_candidates": [0.25, 0.35],
        "timing_recovery_methods": ["gardner", "mueller_muller"],
        "costas_order": None,
    },
    "UNKNOWN": {
        "carrier_recovery": "coarse_fft_peak",
        "matched_filter_family": "none",
        "initial_rolloff_candidates": [],
        "timing_recovery_methods": ["gardner"],
        "costas_order": None,
    },
}


def get_modulation_family(modulation: str) -> str:
    """Look up broad signal family for a modulation class name."""
    clean_mod = modulation.strip()
    return MODULATION_TO_FAMILY_MAP.get(clean_mod, "UNKNOWN")


def get_demod_hints(modulation: str) -> Dict[str, Any]:
    """Retrieve demodulator routing configuration for a given modulation."""
    family = get_modulation_family(modulation)
    base_hints = DEMODULATOR_ROUTING_MAP.get(family, DEMODULATOR_ROUTING_MAP["UNKNOWN"]).copy()
    base_hints["modulation"] = modulation
    base_hints["family"] = family
    return base_hints


def get_sync_hints(modulation: str) -> Dict[str, Any]:
    """Retrieve suggested synchronization hints for a given modulation."""
    family = get_modulation_family(modulation)
    hints = SYNC_ROUTING_HINTS.get(family, SYNC_ROUTING_HINTS["UNKNOWN"]).copy()
    hints["modulation"] = modulation
    hints["family"] = family
    
    # Specific adjustment for Costas loop order if PSK
    if family == "PSK":
        order_map = hints.get("costas_order_map", {})
        hints["costas_order"] = order_map.get(modulation, 4)
        if "costas_order_map" in hints:
            del hints["costas_order_map"]
            
    return hints
