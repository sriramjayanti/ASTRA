"""
candidate_generator.py
Generates bounded, deduplicated deinterleaver candidate hypotheses across enabled families.
Uses divisibility heuristics, bounds checking, canonical ID generation, and permutation deduplication.
"""

from typing import List, Dict, Any, Optional, Set
import numpy as np
from .models import InterleaverFamily, PermutationMapping
from .identity import create_identity_mapping
from .block import create_block_mapping
from .helical import create_helical_mapping
from .pseudo_random import create_pseudorandom_mapping


class CandidateHypothesis:
    """Descriptor of an interleaver hypothesis to test."""
    def __init__(
        self,
        candidate_id: str,
        family: str,
        parameters: Dict[str, Any],
        mapping: Optional[PermutationMapping] = None
    ):
        self.candidate_id = candidate_id
        self.family = family
        self.parameters = parameters
        self.mapping = mapping
        self.mapping_hash = mapping.mapping_hash if mapping else ""


class InterleaverCandidateGenerator:
    """
    Constructs bounded candidate grids across identity, block, convolutional, helical, and pseudo-random families.
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = config or {}
        self.cfg = cfg.get("interleaver", cfg)
        self.families_cfg = self.cfg.get("families", {
            "identity": True,
            "block": True,
            "convolutional": True,
            "helical": True,
            "pseudo_random": True
        })
        self.block_cfg = self.cfg.get("block", {})
        self.conv_cfg = self.cfg.get("convolutional", {})
        self.hel_cfg = self.cfg.get("helical", {})
        self.pr_cfg = self.cfg.get("pseudo_random", {})
        self.search_cfg = self.cfg.get("search", {})
        
        self.max_total = int(self.search_cfg.get("max_total_candidates", 120))

    def generate(self, bit_count: int, variant_id: str = "var0") -> List[CandidateHypothesis]:
        """
        Generates a bounded, deduplicated list of CandidateHypothesis objects.
        """
        candidates: List[CandidateHypothesis] = []
        seen_hashes: Set[str] = set()
        cand_counter = 1
        
        # 1. Identity (NO_INTERLEAVER) - Mandatory
        if self.families_cfg.get("identity", True):
            id_mapping = create_identity_mapping(min(bit_count, 64))
            cid = f"int_none_{cand_counter:04d}"
            cand_counter += 1
            candidates.append(CandidateHypothesis(
                candidate_id=cid,
                family=InterleaverFamily.IDENTITY.value,
                parameters={"type": "identity"},
                mapping=id_mapping
            ))
            seen_hashes.add(id_mapping.mapping_hash)
            
        # 2. Block Candidates
        if self.families_cfg.get("block", True):
            row_cands = self.block_cfg.get("row_candidates", [4, 8, 16, 32, 64])
            col_cands = self.block_cfg.get("column_candidates", [4, 8, 16, 32, 64, 128, 256])
            orientations = self.block_cfg.get("orientations", ["row_to_column", "column_to_row"])
            max_block = int(self.block_cfg.get("max_candidates", 35))
            
            block_count = 0
            # Build balanced candidate (r, c) pairs
            block_pairs = []
            for r in row_cands:
                for c in col_cands:
                    B = r * c
                    if B <= bit_count:
                        rem = bit_count % B
                        # Prefer balanced aspect ratios and exact divisors
                        aspect_ratio = max(r / c, c / r)
                        block_pairs.append((rem, aspect_ratio, B, r, c))
                        
            # Sort by remainder = 0 first, then moderate aspect ratio
            block_pairs.sort(key=lambda x: (x[0], x[1], x[2]))
            
            for rem, aspect, B, r, c in block_pairs:
                if block_count >= max_block or len(candidates) >= self.max_total:
                    break
                for orient in orientations:
                    if block_count >= max_block or len(candidates) >= self.max_total:
                        break
                    try:
                        mapping = create_block_mapping(r, c, orient)
                        if mapping.mapping_hash in seen_hashes:
                            continue
                        if np.array_equal(mapping.inverse_indices, np.arange(B)):
                            continue
                            
                        seen_hashes.add(mapping.mapping_hash)
                        orient_tag = "rc" if orient == "row_to_column" else "cr"
                        cid = f"int_block_r{r}_c{c}_{orient_tag}_{cand_counter:04d}"
                        cand_counter += 1
                        candidates.append(CandidateHypothesis(
                            candidate_id=cid,
                            family=InterleaverFamily.BLOCK.value,
                            parameters={"rows": r, "cols": c, "block_size": B, "orientation": orient},
                            mapping=mapping
                        ))
                        block_count += 1
                    except Exception:
                        continue
                        
        # 3. Convolutional Candidates
        if self.families_cfg.get("convolutional", True):
            branches = self.conv_cfg.get("branch_candidates", [2, 4, 8, 16])
            delay_steps = self.conv_cfg.get("delay_steps", [1, 2, 4, 8])
            max_conv = int(self.conv_cfg.get("max_candidates", 20))
            
            conv_count = 0
            for b in branches:
                for d in delay_steps:
                    if conv_count >= max_conv or len(candidates) >= self.max_total:
                        break
                    latency = b * (b - 1) * d
                    if latency >= bit_count:
                        continue
                    cid = f"int_conv_b{b}_d{d}_{cand_counter:04d}"
                    cand_counter += 1
                    candidates.append(CandidateHypothesis(
                        candidate_id=cid,
                        family=InterleaverFamily.CONVOLUTIONAL.value,
                        parameters={"branches": b, "delay_step": d, "latency": latency},
                        mapping=None
                    ))
                    conv_count += 1
                    
        # 4. Helical Candidates
        if self.families_cfg.get("helical", True):
            hel_rows = self.hel_cfg.get("rows", [4, 8, 16, 32])
            hel_steps = [1, 3]  # Standard prime steps
            hel_orients = self.hel_cfg.get("orientations", ["row_diagonal", "diagonal_row"])
            max_hel = int(self.hel_cfg.get("max_candidates", 30))
            
            hel_count = 0
            for r in hel_rows:
                for c in [r, r * 2]:
                    B = r * c
                    if B > bit_count:
                        continue
                    for step in hel_steps:
                        for orient in hel_orients:
                            if hel_count >= max_hel or len(candidates) >= self.max_total:
                                break
                            try:
                                mapping = create_helical_mapping(r, c, step, orient)
                                if mapping.mapping_hash in seen_hashes:
                                    continue
                                if np.array_equal(mapping.inverse_indices, np.arange(B)):
                                    continue
                                seen_hashes.add(mapping.mapping_hash)
                                cid = f"int_helix_r{r}_c{c}_s{step}_{cand_counter:04d}"
                                cand_counter += 1
                                candidates.append(CandidateHypothesis(
                                    candidate_id=cid,
                                    family=InterleaverFamily.HELICAL.value,
                                    parameters={"rows": r, "cols": c, "step": step, "block_size": B, "orientation": orient},
                                    mapping=mapping
                                ))
                                hel_count += 1
                            except Exception:
                                continue
                                
        # 5. Pseudo-Random Candidates (Constrained)
        if self.families_cfg.get("pseudo_random", True):
            seeds = self.pr_cfg.get("seed_candidates", [0, 1, 42, 123, 1337])
            algorithms = self.pr_cfg.get("algorithms", ["pcg64", "fisher_yates"])
            profiles = self.pr_cfg.get("profiles", [])
            max_pr = int(self.pr_cfg.get("max_candidates", 25))
            
            pr_count = 0
            # Test registered profiles first
            for prof in profiles:
                if pr_count >= max_pr or len(candidates) >= self.max_total:
                    break
                p_len = prof.get("block_length", 1024)
                if p_len <= bit_count:
                    try:
                        mapping = create_pseudorandom_mapping(
                            length=p_len,
                            seed=prof.get("seed", 42),
                            algorithm=prof.get("algorithm", "pcg64"),
                            profile_name=prof.get("name")
                        )
                        if mapping.mapping_hash in seen_hashes:
                            continue
                        seen_hashes.add(mapping.mapping_hash)
                        cid = f"int_pr_{prof.get('name')}_{cand_counter:04d}"
                        cand_counter += 1
                        candidates.append(CandidateHypothesis(
                            candidate_id=cid,
                            family=InterleaverFamily.PSEUDO_RANDOM.value,
                            parameters={"seed": prof.get("seed"), "length": p_len, "algorithm": prof.get("algorithm", "pcg64"), "profile": prof.get("name")},
                            mapping=mapping
                        ))
                        pr_count += 1
                    except Exception:
                        continue
                        
            # Test multi-length candidates across seeds
            for p_len in [256, 512, 1024, 128, 64]:
                if p_len > bit_count or pr_count >= max_pr:
                    continue
                for seed in seeds:
                    if pr_count >= max_pr:
                        break
                    for algo in algorithms:
                        if pr_count >= max_pr or len(candidates) >= self.max_total:
                            break
                        try:
                            mapping = create_pseudorandom_mapping(length=p_len, seed=seed, algorithm=algo)
                            if mapping.mapping_hash in seen_hashes:
                                continue
                            seen_hashes.add(mapping.mapping_hash)
                            cid = f"int_pr_{algo}_s{seed}_l{p_len}_{cand_counter:04d}"
                            cand_counter += 1
                            candidates.append(CandidateHypothesis(
                                candidate_id=cid,
                                family=InterleaverFamily.PSEUDO_RANDOM.value,
                                parameters={"seed": seed, "length": p_len, "algorithm": algo},
                                mapping=mapping
                            ))
                            pr_count += 1
                        except Exception:
                            continue
                            
        return candidates[:self.max_total]
