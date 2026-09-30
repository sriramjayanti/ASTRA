"""
ASTRA FEC (Forward Error Correction) Encoding & Coding Ground-Truth Module.
Synthetic Engine 3 for Automated Signal Analysis & Recovery Assistant.
"""

from .models import FECRecord
from .padding import pad_to_multiple, segment_into_blocks, unpad
from .convolutional import ConvolutionalCode, octal_to_taps
from .reed_solomon import ReedSolomonCode
from .concatenated import ConcatenatedCode
from .ldpc import LDPCCode, LDPCMatrixProfile, LDPC_PROFILES, generate_systematic_ldpc_profile
from .profiles import BUILTIN_FEC_PROFILES, FECProfileDef, get_fec_profile
from .validators import validate_fec_record, reference_decode
from .serializers import save_fec_record, load_fec_record, save_fec_batch
from .generator import FECGenerator

__all__ = [
    "FECGenerator",
    "FECRecord",
    "ConvolutionalCode",
    "octal_to_taps",
    "ReedSolomonCode",
    "ConcatenatedCode",
    "LDPCCode",
    "LDPCMatrixProfile",
    "LDPC_PROFILES",
    "generate_systematic_ldpc_profile",
    "BUILTIN_FEC_PROFILES",
    "FECProfileDef",
    "get_fec_profile",
    "pad_to_multiple",
    "segment_into_blocks",
    "unpad",
    "validate_fec_record",
    "reference_decode",
    "save_fec_record",
    "load_fec_record",
    "save_fec_batch",
]
