"""
profiles.py
FEC Profile registry loader, validator, and manager.
Loads standardized convolutional, Reed-Solomon, concatenated, and LDPC profiles from YAML configuration.
"""

from typing import Dict, List, Optional, Any
import os
import yaml
from .models import FECProfile, FECFamily


class ProfileRegistry:
    """Central registry of registered FEC profiles."""
    def __init__(self, profiles_path: Optional[str] = None):
        self.profiles: Dict[str, FECProfile] = {}
        if profiles_path is not None and os.path.exists(profiles_path):
            self.load_from_yaml(profiles_path)
        else:
            default_path = os.path.join(os.path.dirname(__file__), "..", "configs", "fec_profiles.yaml")
            if os.path.exists(default_path):
                self.load_from_yaml(default_path)
            else:
                self._load_fallback_profiles()

    def load_from_yaml(self, file_path: str):
        with open(file_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
            raw_profiles = data.get("profiles", {})
            for pid, pdata in raw_profiles.items():
                pdata["profile_id"] = pid
                if "generators_octal" in pdata and pdata["generators_octal"] is not None:
                    parsed_gens = []
                    for g in pdata["generators_octal"]:
                        if isinstance(g, str):
                            parsed_gens.append(int(g, 8) if not g.startswith("0x") else int(g, 16))
                        elif isinstance(g, int):
                            try:
                                if g > 7 and all(c in "01234567" for c in str(g)):
                                    parsed_gens.append(int(str(g), 8))
                                else:
                                    parsed_gens.append(g)
                            except Exception:
                                parsed_gens.append(g)
                    pdata["generators_octal"] = parsed_gens
                self.profiles[pid] = FECProfile(**pdata)

    def _load_fallback_profiles(self):
        """Hardcoded fallbacks if YAML is absent."""
        self.profiles["conv_k7_r12_nasa"] = FECProfile(
            profile_id="conv_k7_r12_nasa",
            family="convolutional",
            constraint_length=7,
            rate=0.5,
            rate_str="1/2",
            generators_octal=[171, 133],
            termination="terminated",
            description="NASA Standard K=7 R=1/2"
        )
        self.profiles["rs_255_223_ccsds"] = FECProfile(
            profile_id="rs_255_223_ccsds",
            family="reed_solomon",
            n=255,
            k=223,
            symbol_bits=8,
            prim_poly=0x187,
            fcr=112,
            description="CCSDS Standard RS(255, 223)"
        )
        self.profiles["ldpc_128_r12"] = FECProfile(
            profile_id="ldpc_128_r12",
            family="ldpc",
            n=128,
            k=64,
            rate=0.5,
            rate_str="1/2",
            block_size=16,
            description="LDPC(128, 64) Rate 1/2"
        )

    def get_profile(self, profile_id: str) -> Optional[FECProfile]:
        return self.profiles.get(profile_id)

    def list_profiles(self, family: Optional[str] = None) -> List[FECProfile]:
        if family is None:
            return list(self.profiles.values())
        return [p for p in self.profiles.values() if p.family == family]


# Global singleton instance
_GLOBAL_REGISTRY: Optional[ProfileRegistry] = None


def get_profile_registry() -> ProfileRegistry:
    global _GLOBAL_REGISTRY
    if _GLOBAL_REGISTRY is None:
        _GLOBAL_REGISTRY = ProfileRegistry()
    return _GLOBAL_REGISTRY
