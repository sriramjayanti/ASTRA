"""
router.py
Routes deinterleaver candidate hypotheses to family-specific execution backends.
"""

from typing import Tuple, Optional, Dict, Any
import numpy as np
from .models import InterleaverFamily, PermutationMapping, ConvolutionalDeinterleaverState
from .identity import deinterleave_identity
from .block import deinterleave_block
from .convolutional import deinterleave_convolutional
from .helical import deinterleave_helical
from .pseudo_random import deinterleave_pseudorandom


def execute_deinterleaver(
    family: str,
    parameters: Dict[str, Any],
    hard_bits: np.ndarray,
    soft_llrs: Optional[np.ndarray] = None,
    state: Optional[ConvolutionalDeinterleaverState] = None,
    padding_policy: str = "truncate_tail"
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[PermutationMapping], Optional[ConvolutionalDeinterleaverState], float]:
    """
    Executes family-specific deinterleaving algorithm.
    
    Returns:
        (deinterleaved_hard, deinterleaved_soft, mapping, conv_state, remainder_fraction)
    """
    N = len(hard_bits)
    if family == InterleaverFamily.IDENTITY.value or family == "identity" or family == "none":
        deint_hard, deint_soft, mapping = deinterleave_identity(hard_bits, soft_llrs)
        return deint_hard, deint_soft, mapping, None, 0.0
        
    elif family == InterleaverFamily.BLOCK.value or family == "block":
        rows = parameters.get("rows", 16)
        cols = parameters.get("cols", 16)
        orient = parameters.get("orientation", "row_to_column")
        deint_hard, deint_soft, mapping, rem = deinterleave_block(
            hard_bits=hard_bits,
            soft_llrs=soft_llrs,
            rows=rows,
            cols=cols,
            orientation=orient,
            padding_policy=padding_policy
        )
        rem_frac = float(rem) / float(N) if N > 0 else 0.0
        return deint_hard, deint_soft, mapping, None, rem_frac
        
    elif family == InterleaverFamily.CONVOLUTIONAL.value or family == "convolutional":
        branches = parameters.get("branches", 4)
        delay_step = parameters.get("delay_step", 2)
        deint_hard, deint_soft, new_state, latency = deinterleave_convolutional(
            hard_bits=hard_bits,
            soft_llrs=soft_llrs,
            branch_count=branches,
            delay_step=delay_step,
            state=state,
            trim_latency=False
        )
        rem_frac = min(float(latency) / float(N), 1.0) if N > 0 else 0.0
        return deint_hard, deint_soft, None, new_state, rem_frac
        
    elif family == InterleaverFamily.HELICAL.value or family == "helical":
        rows = parameters.get("rows", 16)
        cols = parameters.get("cols", 16)
        step = parameters.get("step", 1)
        orient = parameters.get("orientation", "row_diagonal")
        deint_hard, deint_soft, mapping, rem = deinterleave_helical(
            hard_bits=hard_bits,
            soft_llrs=soft_llrs,
            rows=rows,
            cols=cols,
            step=step,
            orientation=orient,
            padding_policy=padding_policy
        )
        rem_frac = float(rem) / float(N) if N > 0 else 0.0
        return deint_hard, deint_soft, mapping, None, rem_frac
        
    elif family == InterleaverFamily.PSEUDO_RANDOM.value or family == "pseudo_random":
        length = parameters.get("length", 1024)
        seed = parameters.get("seed", 42)
        algo = parameters.get("algorithm", "pcg64")
        deint_hard, deint_soft, mapping, rem = deinterleave_pseudorandom(
            hard_bits=hard_bits,
            soft_llrs=soft_llrs,
            length=length,
            seed=seed,
            algorithm=algo,
            padding_policy=padding_policy
        )
        rem_frac = float(rem) / float(N) if N > 0 else 0.0
        return deint_hard, deint_soft, mapping, None, rem_frac
        
    else:
        raise ValueError(f"Unknown interleaver family: {family}")
