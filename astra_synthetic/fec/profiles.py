"""
Predefined FEC Profile Catalog for ASTRA FEC Engine.
Defines bounded, standardized coding configurations across all FEC families.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FECProfileDef:
    """Specification of an FEC profile."""
    name: str
    fec_type: str
    nominal_rate: float
    description: str
    config: dict[str, Any]


BUILTIN_FEC_PROFILES: dict[str, FECProfileDef] = {
    # 1. No FEC
    "none": FECProfileDef(
        name="none",
        fec_type="none",
        nominal_rate=1.0,
        description="Uncoded transmission / No FEC",
        config={},
    ),

    # 2. Convolutional Codes
    "conv_k3_r12": FECProfileDef(
        name="conv_k3_r12",
        fec_type="convolutional",
        nominal_rate=0.5,
        description="Convolutional K=3, Rate 1/2, Generators [0o7, 0o5]",
        config={
            "constraint_length": 3,
            "generators": [0o7, 0o5],
            "rate": "1/2",
            "termination_mode": "zero_tail",
        },
    ),
    "conv_k5_r12": FECProfileDef(
        name="conv_k5_r12",
        fec_type="convolutional",
        nominal_rate=0.5,
        description="Convolutional K=5, Rate 1/2, Generators [0o35, 0o23]",
        config={
            "constraint_length": 5,
            "generators": [0o35, 0o23],
            "rate": "1/2",
            "termination_mode": "zero_tail",
        },
    ),
    "conv_k7_r12": FECProfileDef(
        name="conv_k7_r12",
        fec_type="convolutional",
        nominal_rate=0.5,
        description="NASA Standard Convolutional K=7, Rate 1/2, Generators [0o171, 0o133]",
        config={
            "constraint_length": 7,
            "generators": [0o171, 0o133],
            "rate": "1/2",
            "termination_mode": "zero_tail",
        },
    ),

    # 3. Reed-Solomon Codes
    "rs_255_223": FECProfileDef(
        name="rs_255_223",
        fec_type="reed_solomon",
        nominal_rate=223.0 / 255.0,
        description="CCSDS Standard Reed-Solomon (255, 223), t=16 symbols",
        config={
            "n": 255,
            "k": 223,
            "symbol_size_bits": 8,
        },
    ),
    "rs_255_239": FECProfileDef(
        name="rs_255_239",
        fec_type="reed_solomon",
        nominal_rate=239.0 / 255.0,
        description="ITU Standard Reed-Solomon (255, 239), t=8 symbols",
        config={
            "n": 255,
            "k": 239,
            "symbol_size_bits": 8,
        },
    ),
    "rs_64_48": FECProfileDef(
        name="rs_64_48",
        fec_type="reed_solomon",
        nominal_rate=48.0 / 64.0,
        description="Shortened Reed-Solomon (64, 48), t=8 symbols",
        config={
            "n": 64,
            "k": 48,
            "symbol_size_bits": 8,
        },
    ),

    # 4. Concatenated Codes
    "concat_rs255223_conv_k7": FECProfileDef(
        name="concat_rs255223_conv_k7",
        fec_type="concatenated",
        nominal_rate=(223.0 / 255.0) * 0.5,
        description="Concatenated: Outer RS(255,223) + Inner Conv K=7 Rate 1/2",
        config={
            "outer": "rs_255_223",
            "inner": "conv_k7_r12",
        },
    ),
    "concat_rs255239_conv_k3": FECProfileDef(
        name="concat_rs255239_conv_k3",
        fec_type="concatenated",
        nominal_rate=(239.0 / 255.0) * 0.5,
        description="Concatenated: Outer RS(255,239) + Inner Conv K=3 Rate 1/2",
        config={
            "outer": "rs_255_239",
            "inner": "conv_k3_r12",
        },
    ),

    # 5. LDPC Codes
    "ldpc_n128_k64_r12": FECProfileDef(
        name="ldpc_n128_k64_r12",
        fec_type="ldpc",
        nominal_rate=0.5,
        description="Systematic LDPC (128, 64) Rate 1/2",
        config={
            "profile_id": "ldpc_n128_k64_r12",
        },
    ),
    "ldpc_n256_k128_r12": FECProfileDef(
        name="ldpc_n256_k128_r12",
        fec_type="ldpc",
        nominal_rate=0.5,
        description="Systematic LDPC (256, 128) Rate 1/2",
        config={
            "profile_id": "ldpc_n256_k128_r12",
        },
    ),
    "ldpc_n512_k256_r12": FECProfileDef(
        name="ldpc_n512_k256_r12",
        fec_type="ldpc",
        nominal_rate=0.5,
        description="Systematic LDPC (512, 256) Rate 1/2",
        config={
            "profile_id": "ldpc_n512_k256_r12",
        },
    ),
    "ldpc_n96_k64_r23": FECProfileDef(
        name="ldpc_n96_k64_r23",
        fec_type="ldpc",
        nominal_rate=64.0 / 96.0,
        description="Systematic LDPC (96, 64) Rate 2/3",
        config={
            "profile_id": "ldpc_n96_k64_r23",
        },
    ),
}


def get_fec_profile(name: str) -> FECProfileDef:
    """Retrieve an FEC profile definition by name."""
    key = str(name).strip().lower()
    if key in BUILTIN_FEC_PROFILES:
        return BUILTIN_FEC_PROFILES[key]
    
    available = ", ".join(BUILTIN_FEC_PROFILES.keys())
    raise ValueError(f"Unknown FEC profile '{name}'. Available profiles: {available}")
