"""
pruning.py
Beam pruning and multi-family candidate diversity preservation for Stage 9 FEC hypotheses.
"""

from typing import List, Dict, Any, Set
from .models import FECCandidateResult, FECFamily, FECStatus


def prune_fec_candidates(
    candidates: List[FECCandidateResult],
    beam_width: int = 8,
    always_preserve_none: bool = True,
    enforce_family_diversity: bool = True
) -> List[FECCandidateResult]:
    """
    Ranks FEC candidate results by fec_quality_score and retains top-K.
    Ensures:
      1. NO_FEC is always retained in the candidate set.
      2. Multi-family representation is balanced across beam slots.
    """
    if not candidates:
        return []
        
    valid_cands = [c for c in candidates if not c.rejected]
    if not valid_cands:
        # Fallback to whatever candidates are available
        valid_cands = list(candidates)
        
    valid_cands.sort(key=lambda c: c.fec_quality_score, reverse=True)
    
    if not enforce_family_diversity:
        survivors = valid_cands[:beam_width]
        if always_preserve_none:
            none_cand = next((c for c in valid_cands if c.fec_family in ("none", FECFamily.NONE.value)), None)
            if none_cand is not None and none_cand not in survivors and len(survivors) > 0:
                survivors[-1] = none_cand
        return survivors
        
    survivors: List[FECCandidateResult] = []
    family_buckets: Dict[str, List[FECCandidateResult]] = {}
    
    for c in valid_cands:
        fam = c.fec_family
        if fam not in family_buckets:
            family_buckets[fam] = []
        family_buckets[fam].append(c)
        
    # 1. Guarantee NO_FEC
    if always_preserve_none and "none" in family_buckets and len(family_buckets["none"]) > 0:
        survivors.append(family_buckets["none"].pop(0))
        
    # 2. Round-robin top candidate from each family
    families_order = ["convolutional", "reed_solomon", "ldpc", "concatenated"]
    for fam in families_order:
        if len(survivors) >= beam_width:
            break
        if fam in family_buckets and len(family_buckets[fam]) > 0:
            survivors.append(family_buckets[fam].pop(0))
            
    # 3. Second round for remaining capacity
    for fam in families_order:
        if len(survivors) >= beam_width:
            break
        if fam in family_buckets and len(family_buckets[fam]) > 0:
            survivors.append(family_buckets[fam].pop(0))
            
    # 4. Fill remaining slots by highest global score among remaining
    remaining = [c for b in family_buckets.values() for c in b]
    remaining.sort(key=lambda c: c.fec_quality_score, reverse=True)
    
    while len(survivors) < beam_width and len(remaining) > 0:
        cand = remaining.pop(0)
        if cand not in survivors:
            survivors.append(cand)
            
    survivors.sort(key=lambda c: c.fec_quality_score, reverse=True)
    return survivors
