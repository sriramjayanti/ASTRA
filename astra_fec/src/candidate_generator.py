"""
candidate_generator.py
Generates bounded FEC candidate hypotheses and alignment offsets from registered profile catalogs.
"""

from typing import List, Dict, Any, Optional
import numpy as np
from .models import FECProfile, FECFamily
from .profiles import get_profile_registry


class FECHypothesis:
    """Descriptor of a FEC candidate hypothesis to test."""
    def __init__(
        self,
        candidate_id: str,
        profile: Optional[FECProfile],
        family: str,
        parameters: Dict[str, Any],
        bit_offset: int = 0
    ):
        self.candidate_id = candidate_id
        self.profile = profile
        self.family = family
        self.parameters = parameters
        self.bit_offset = int(bit_offset)


class FECCandidateGenerator:
    """
    Constructs bounded FEC candidate search sets across enabled families.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = config or {}
        self.cfg = cfg.get("fec", cfg)
        self.families_cfg = self.cfg.get("families", {})
        self.search_cfg = self.cfg.get("search", {})
        self.max_candidates = int(self.search_cfg.get("max_candidates_per_input", 25))
        self.enable_alignment = bool(self.search_cfg.get("enable_alignment_search", True))
        self.max_offsets = int(self.search_cfg.get("max_alignment_offsets", 8))
        self.registry = get_profile_registry()

    def generate(self, bit_count: int) -> List[FECHypothesis]:
        """
        Generates bounded list of FEC candidate hypotheses.
        """
        hypotheses: List[FECHypothesis] = []
        cand_counter = 1
        
        # 1. NO_FEC (Mandatory)
        none_cfg = self.families_cfg.get("none", {"enabled": True})
        if none_cfg.get("enabled", True):
            cid = f"fec_none_{cand_counter:04d}"
            cand_counter += 1
            hypotheses.append(FECHypothesis(
                candidate_id=cid,
                profile=None,
                family=FECFamily.NONE.value,
                parameters={"type": "uncoded", "code_rate": 1.0},
                bit_offset=0
            ))
            
        # 2. Convolutional Profiles
        conv_cfg = self.families_cfg.get("convolutional", {"enabled": True})
        if conv_cfg.get("enabled", True):
            profiles = self.registry.list_profiles(family="convolutional")
            for prof in profiles:
                if len(hypotheses) >= self.max_candidates:
                    break
                # Test both code symbol alignment offsets (0 and 1) for rate 1/2 codes
                offsets_to_test = [0, 1] if self.enable_alignment else [0]
                for off in offsets_to_test:
                    if len(hypotheses) >= self.max_candidates:
                        break
                    off_tag = f"_off{off}" if off > 0 else ""
                    cid = f"fec_{prof.profile_id}{off_tag}_{cand_counter:04d}"
                    cand_counter += 1
                    hypotheses.append(FECHypothesis(
                        candidate_id=cid,
                        profile=prof,
                        family=prof.family,
                        parameters={"profile_id": prof.profile_id, "rate": prof.rate, "K": prof.constraint_length, "bit_offset": off},
                        bit_offset=off
                    ))
                
        # 3. Reed-Solomon Profiles (with optional bit offsets for symbol alignment)
        rs_cfg = self.families_cfg.get("reed_solomon", {"enabled": True})
        if rs_cfg.get("enabled", True):
            profiles = self.registry.list_profiles(family="reed_solomon")
            for prof in profiles:
                if len(hypotheses) >= self.max_candidates:
                    break
                m = prof.symbol_bits or 8
                min_bits_needed = (prof.shortened_n or prof.n) * m
                if bit_count < min_bits_needed:
                    continue
                    
                offsets_to_test = [0]
                if self.enable_alignment and m > 1:
                    offsets_to_test.extend([o for o in range(1, min(m, self.max_offsets))])
                    
                for off in offsets_to_test:
                    if len(hypotheses) >= self.max_candidates:
                        break
                    off_tag = f"_off{off}" if off > 0 else ""
                    cid = f"fec_{prof.profile_id}{off_tag}_{cand_counter:04d}"
                    cand_counter += 1
                    hypotheses.append(FECHypothesis(
                        candidate_id=cid,
                        profile=prof,
                        family=prof.family,
                        parameters={"profile_id": prof.profile_id, "n": prof.n, "k": prof.k, "symbol_bits": m, "offset": off},
                        bit_offset=off
                    ))
                    
        # 4. LDPC Profiles
        ldpc_cfg = self.families_cfg.get("ldpc", {"enabled": True})
        if ldpc_cfg.get("enabled", True):
            profiles = self.registry.list_profiles(family="ldpc")
            for prof in profiles:
                if len(hypotheses) >= self.max_candidates:
                    break
                if bit_count >= (prof.n or 128):
                    cid = f"fec_{prof.profile_id}_{cand_counter:04d}"
                    cand_counter += 1
                    hypotheses.append(FECHypothesis(
                        candidate_id=cid,
                        profile=prof,
                        family=prof.family,
                        parameters={"profile_id": prof.profile_id, "n": prof.n, "k": prof.k, "rate": prof.rate},
                        bit_offset=0
                    ))
                    
        # 5. Concatenated Profiles
        concat_cfg = self.families_cfg.get("concatenated", {"enabled": True})
        if concat_cfg.get("enabled", True):
            profiles = self.registry.list_profiles(family="concatenated")
            for prof in profiles:
                if len(hypotheses) >= self.max_candidates:
                    break
                cid = f"fec_{prof.profile_id}_{cand_counter:04d}"
                cand_counter += 1
                hypotheses.append(FECHypothesis(
                    candidate_id=cid,
                    profile=prof,
                    family=prof.family,
                    parameters={"profile_id": prof.profile_id, "outer": prof.outer_profile, "inner": prof.inner_profile},
                    bit_offset=0
                ))
                
        return hypotheses[:self.max_candidates]
