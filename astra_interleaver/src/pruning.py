"""
pruning.py
Beam pruning, duplicate permutation collision elimination, and hypothesis diversity enforcement.
"""

from typing import List, Dict, Any, Set
from .models import InterleaverCandidateResult, InterleaverFamily, InterleaverStatus


def prune_candidates(
    candidates: List[InterleaverCandidateResult],
    beam_width: int = 10,
    always_preserve_identity: bool = True,
    enforce_family_diversity: bool = True
) -> List[InterleaverCandidateResult]:
    """
    Ranks candidate hypotheses by overall structural score and retains top-K.
    Ensures:
      1. Identity (NO_INTERLEAVER) is always preserved.
      2. Multi-family representation is guaranteed (avoids single family crowding).
      3. Identical duplicate permutations are eliminated.
    """
    if not candidates:
        return []
        
    # Separate valid and invalid/rejected
    valid_candidates: List[InterleaverCandidateResult] = []
    seen_hashes: Set[str] = set()
    
    for cand in candidates:
        if not cand.success or cand.rejected:
            continue
        # Deduplicate permutation hashes if hash is available
        if cand.permutation_hash and cand.permutation_hash in seen_hashes:
            cand.rejected = True
            cand.rejection_reason = "Duplicate permutation mapping"
            continue
        if cand.permutation_hash:
            seen_hashes.add(cand.permutation_hash)
        valid_candidates.append(cand)
        
    # Sort valid candidates descending by overall_interleaver_score
    valid_candidates.sort(key=lambda c: c.overall_interleaver_score, reverse=True)
    
    if not enforce_family_diversity:
        survivors = valid_candidates[:beam_width]
        if always_preserve_identity:
            identity_cand = next(
                (c for c in valid_candidates if c.interleaver_family in ("identity", InterleaverFamily.IDENTITY.value)),
                None
            )
            if identity_cand is not None and identity_cand not in survivors and len(survivors) > 0:
                survivors[-1] = identity_cand
        return survivors
        
    # Enforce balanced family diversity in beam
    survivors: List[InterleaverCandidateResult] = []
    family_buckets: Dict[str, List[InterleaverCandidateResult]] = {}
    
    for cand in valid_candidates:
        fam = cand.interleaver_family
        if fam not in family_buckets:
            family_buckets[fam] = []
        family_buckets[fam].append(cand)
        
    # Guarantee identity
    if always_preserve_identity and "identity" in family_buckets and len(family_buckets["identity"]) > 0:
        survivors.append(family_buckets["identity"].pop(0))
        
    # Round-robin top candidate from each family
    families_order = ["block", "convolutional", "helical", "pseudo_random"]
    for fam in families_order:
        if len(survivors) >= beam_width:
            break
        if fam in family_buckets and len(family_buckets[fam]) > 0:
            survivors.append(family_buckets[fam].pop(0))
            
    # Second round for remaining capacity
    for fam in families_order:
        if len(survivors) >= beam_width:
            break
        if fam in family_buckets and len(family_buckets[fam]) > 0:
            survivors.append(family_buckets[fam].pop(0))
            
    # Fill any remaining slots by highest global score among remaining
    remaining_cands = [c for bucket in family_buckets.values() for c in bucket]
    remaining_cands.sort(key=lambda c: c.overall_interleaver_score, reverse=True)
    
    while len(survivors) < beam_width and len(remaining_cands) > 0:
        cand = remaining_cands.pop(0)
        if cand not in survivors:
            survivors.append(cand)
            
    # Final sort of surviving candidates by score descending
    survivors.sort(key=lambda c: c.overall_interleaver_score, reverse=True)
    return survivors
