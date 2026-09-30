"""
router.py
Routes FEC candidate hypotheses to their appropriate decoding backends.
"""

from typing import Optional, Dict, Any
import numpy as np
from .models import DecoderResult, FECProfile, FECFamily
from .no_fec import decode_no_fec
from .viterbi import decode_convolutional_profile
from .reed_solomon import decode_reed_solomon_profile
from .ldpc import decode_ldpc_profile
from .concatenated import decode_concatenated_profile


def execute_decoder(
    profile: Optional[FECProfile],
    hard_bits: np.ndarray,
    soft_llrs: Optional[np.ndarray] = None,
    prefer_soft: bool = True
) -> DecoderResult:
    """
    Routes to family-specific decoder backend based on FECProfile.
    """
    if profile is None or profile.family in ("none", FECFamily.NONE.value):
        return decode_no_fec(hard_bits, soft_llrs)
        
    fam = profile.family
    if fam in ("convolutional", FECFamily.CONVOLUTIONAL.value):
        return decode_convolutional_profile(hard_bits, soft_llrs, profile, prefer_soft=prefer_soft)
    elif fam in ("reed_solomon", FECFamily.REED_SOLOMON.value):
        return decode_reed_solomon_profile(hard_bits, profile)
    elif fam in ("ldpc", FECFamily.LDPC.value):
        return decode_ldpc_profile(hard_bits, soft_llrs, profile)
    elif fam in ("concatenated", FECFamily.CONCATENATED.value):
        return decode_concatenated_profile(hard_bits, soft_llrs, profile, prefer_soft=prefer_soft)
    else:
        return DecoderResult(
            success=False,
            decoded_bits=np.array([], dtype=np.uint8),
            failure_reason=f"Unsupported FEC family: {fam}",
            decoder_name="unknown",
            profile_id=profile.profile_id
        )
