"""
convolutional.py
Convolutional encoder, trellis construction, puncturing, and depuncturing utilities.
Supports arbitrary constraint length K, generator polynomials in octal, and arbitrary puncturing patterns.
"""

from typing import List, Tuple, Optional, Dict, Any
import numpy as np
from .models import FECProfile


class ConvolutionalTrellis:
    """
    Precomputed trellis representation for rate 1/n convolutional code.
    Number of states S = 2^(K-1).
    """
    def __init__(self, constraint_length: int, generators_octal: List[int]):
        self.K = int(constraint_length)
        self.num_states = 1 << (self.K - 1)
        self.generators = [int(g) for g in generators_octal]
        self.n_outputs = len(self.generators)
        
        # Precompute next_state[s, u] and outputs[s, u]
        # s in [0, S-1], u in [0, 1]
        self.next_state = np.zeros((self.num_states, 2), dtype=np.int32)
        self.outputs = np.zeros((self.num_states, 2, self.n_outputs), dtype=np.uint8)
        
        # Precompute previous transitions: prev_states[s_next] = [(s_prev, u, outputs)]
        self.prev_transitions = [[] for _ in range(self.num_states)]
        
        for s in range(self.num_states):
            for u in (0, 1):
                # 7-bit / K-bit register word
                reg = (u << (self.K - 1)) | s
                s_next = reg >> 1
                self.next_state[s, u] = s_next
                
                out_bits = []
                for g in self.generators:
                    parity = bin(reg & g).count('1') % 2
                    out_bits.append(parity)
                    
                self.outputs[s, u] = np.array(out_bits, dtype=np.uint8)
                self.prev_transitions[s_next].append((s, u, np.array(out_bits, dtype=np.uint8)))


def encode_convolutional(
    message_bits: np.ndarray,
    constraint_length: int = 7,
    generators_octal: Optional[List[int]] = None,
    termination: str = "terminated",
    puncturing_pattern: Optional[List[int]] = None
) -> np.ndarray:
    """
    Encodes a binary message bitstream with a convolutional encoder.
    
    Args:
        message_bits: 1D uint8 array
        constraint_length: K (e.g. 7)
        generators_octal: list of octal polynomials (default: [171, 133])
        termination: 'terminated' (appends K-1 zero tail bits) or 'continuous'
        puncturing_pattern: optional list of 1s and 0s (e.g. [1, 1, 0, 1])
        
    Returns:
        1D uint8 array of encoded code bits
    """
    generators = generators_octal if generators_octal is not None else [171, 133]
    trellis = ConvolutionalTrellis(constraint_length, generators)
    
    bits = np.asarray(message_bits, dtype=np.uint8)
    if termination == "terminated":
        # Append K-1 tail zeros to flush trellis back to state 0
        tail = np.zeros(constraint_length - 1, dtype=np.uint8)
        full_input = np.concatenate([bits, tail])
    else:
        full_input = bits
        
    state = 0
    raw_encoded = []
    for u in full_input:
        out = trellis.outputs[state, int(u)]
        state = trellis.next_state[state, int(u)]
        raw_encoded.extend(out)
        
    raw_encoded = np.array(raw_encoded, dtype=np.uint8)
    
    if puncturing_pattern is not None and len(puncturing_pattern) > 0:
        p_pat = np.array(puncturing_pattern, dtype=np.uint8)
        p_len = len(p_pat)
        mask = np.tile(p_pat, int(np.ceil(len(raw_encoded) / p_len)))[:len(raw_encoded)]
        return raw_encoded[mask == 1]
        
    return raw_encoded


def depuncture_stream(
    hard_bits: np.ndarray,
    soft_llrs: Optional[np.ndarray],
    puncturing_pattern: List[int]
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Expands a punctured stream back to the mother code rate by inserting neutral LLRs (0.0)
    and dummy hard bits (0).
    """
    p_pat = np.array(puncturing_pattern, dtype=np.uint8)
    p_len = len(p_pat)
    ones_per_period = int(np.sum(p_pat))
    
    n_in = len(hard_bits)
    num_periods = int(np.ceil(n_in / ones_per_period))
    full_mother_len = num_periods * p_len
    
    depunct_hard = np.zeros(full_mother_len, dtype=np.uint8)
    depunct_soft = np.zeros(full_mother_len, dtype=np.float32) if soft_llrs is not None else None
    
    in_idx = 0
    out_idx = 0
    while in_idx < n_in and out_idx < full_mother_len:
        p_val = p_pat[out_idx % p_len]
        if p_val == 1:
            depunct_hard[out_idx] = hard_bits[in_idx]
            if soft_llrs is not None and depunct_soft is not None:
                depunct_soft[out_idx] = soft_llrs[in_idx]
            in_idx += 1
        else:
            # Punctured bit: neutral evidence (0.0 LLR)
            depunct_hard[out_idx] = 0
            if depunct_soft is not None:
                depunct_soft[out_idx] = 0.0
        out_idx += 1
        
    return depunct_hard[:out_idx], depunct_soft[:out_idx] if depunct_soft is not None else None
