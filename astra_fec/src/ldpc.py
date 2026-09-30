"""
ldpc.py
Low-Density Parity-Check (LDPC) codec and normalized Min-Sum decoder.
Supports quasi-cyclic matrix generation, belief-propagation / min-sum iterations, syndrome tracking, and early stopping.
"""

from typing import Tuple, Optional, Dict, Any, List
import time
import numpy as np
from .models import DecoderResult, FECProfile


def construct_qc_ldpc_matrix(n: int = 128, k: int = 64, block_size: int = 16) -> np.ndarray:
    """
    Constructs a deterministic Quasi-Cyclic LDPC parity-check matrix H of shape (M, N).
    M = N - K.
    """
    M = n - k
    n_blocks_row = M // block_size
    n_blocks_col = n // block_size
    
    rng = np.random.default_rng(1337 + n)
    H = np.zeros((M, n), dtype=np.uint8)
    
    # Base prototype shift matrix
    for r in range(n_blocks_row):
        for c in range(n_blocks_col):
            # Sparse block with 2-3 active circulants per column
            if (r + c) % n_blocks_row == 0 or (r * 2 + c) % n_blocks_col == 1:
                shift = (r * 3 + c * 5) % block_size
                I_circ = np.roll(np.eye(block_size, dtype=np.uint8), shift, axis=1)
                H[r * block_size : (r + 1) * block_size, c * block_size : (c + 1) * block_size] = I_circ
                
    # Ensure full row rank / non-empty checks
    for r in range(M):
        if np.sum(H[r]) == 0:
            H[r, r % n] = 1
            H[r, (r + 1) % n] = 1
            
    return H


class LDPCCodec:
    """
    LDPC Codec with Normalized Min-Sum iterative decoder.
    """
    def __init__(self, H: np.ndarray, alpha: float = 0.80, max_iter: int = 30):
        self.H = np.asarray(H, dtype=np.uint8)
        self.M, self.N = self.H.shape
        self.K = self.N - self.M
        self.alpha = float(alpha)
        self.max_iter = int(max_iter)
        
        # Precompute check and variable node adjacency lists
        self.check_neighbors = [np.where(self.H[c, :] == 1)[0] for c in range(self.M)]
        self.var_neighbors = [np.where(self.H[:, v] == 1)[0] for v in range(self.N)]

    def encode(self, info_bits: np.ndarray) -> np.ndarray:
        """
        Encodes K information bits to an N-bit LDPC codeword.
        """
        info_bits = np.asarray(info_bits, dtype=np.uint8)
        if len(info_bits) != self.K:
            raise ValueError(f"Information bits length ({len(info_bits)}) != K ({self.K})")
            
        # Simplified systematic generator: codeword = [info_bits, parity_bits]
        # H = [H_info | H_parity]. H_info * info + H_parity * parity = 0
        H_info = self.H[:, :self.K]
        H_par = self.H[:, self.K:]
        
        # For our QC-LDPC, compute parity by back-substitution or XOR check
        parity = np.zeros(self.M, dtype=np.uint8)
        info_syn = np.dot(H_info, info_bits.astype(np.int32)) % 2
        
        # Invertible lower-triangular or pseudoinverse for parity
        # Using fast GF(2) solver
        try:
            # Solve H_par * p = info_syn (mod 2)
            A = H_par.copy()
            b = info_syn.copy()
            # Gaussian elimination mod 2
            m = self.M
            for col in range(m):
                pivot = -1
                for row in range(col, m):
                    if A[row, col] == 1:
                        pivot = row
                        break
                if pivot >= 0:
                    if pivot != col:
                        A[[col, pivot]] = A[[pivot, col]]
                        b[[col, pivot]] = b[[pivot, col]]
                    for row in range(m):
                        if row != col and A[row, col] == 1:
                            A[row] ^= A[col]
                            b[row] ^= b[col]
            parity = b[:m]
        except Exception:
            # Fallback simple parity
            parity = info_syn
            
        codeword = np.concatenate([info_bits, parity])
        return codeword

    def decode_min_sum(
        self,
        rx_llrs: np.ndarray
    ) -> Tuple[bool, np.ndarray, int, float, float, np.ndarray]:
        """
        Decodes soft LLRs with Normalized Min-Sum algorithm.
        
        Returns:
            (converged, decoded_bits, iterations_used, initial_syndrome_wt, final_syndrome_wt, posterior_llrs)
        """
        rx_llrs = np.asarray(rx_llrs, dtype=np.float32)
        if len(rx_llrs) != self.N:
            raise ValueError(f"LLR vector length ({len(rx_llrs)}) != codeword length ({self.N})")
            
        # Initial hard decision & syndrome
        init_bits = (rx_llrs < 0).astype(np.uint8)  # ASTRA: negative LLR -> bit 1
        init_syn = np.dot(self.H, init_bits.astype(np.int32)) % 2
        init_syn_wt = float(np.sum(init_syn))
        
        if init_syn_wt == 0:
            return True, init_bits, 0, 0.0, 0.0, rx_llrs
            
        # Fast vectorized message passing on dense matrices masked by self.H
        c2v_mat = np.zeros((self.M, self.N), dtype=np.float32)
        v2c_mat = np.tile(rx_llrs, (self.M, 1)) * self.H
        
        converged = False
        iters_used = 0
        decoded_bits = init_bits.copy()
        post_llrs = rx_llrs.copy()
        
        for it in range(1, min(self.max_iter + 1, 15)):
            iters_used = it
            
            # 1. Check Node Update (Normalized Min-Sum)
            for c in range(self.M):
                v_idx = self.check_neighbors[c]
                if len(v_idx) == 0:
                    continue
                v_msgs = v2c_mat[c, v_idx]
                signs = np.where(v_msgs >= 0, 1.0, -1.0)
                mags = np.abs(v_msgs)
                
                prod_sign = np.prod(signs)
                
                # Find min and second min
                if len(mags) >= 2:
                    sorted_indices = np.argsort(mags)
                    min1_val = mags[sorted_indices[0]]
                    min2_val = mags[sorted_indices[1]]
                    min1_idx = sorted_indices[0]
                    
                    c_mags = np.full(len(mags), min1_val, dtype=np.float32)
                    c_mags[min1_idx] = min2_val
                else:
                    c_mags = mags
                    
                c_signs = prod_sign * signs
                c2v_mat[c, v_idx] = self.alpha * c_signs * c_mags
                
            # 2. Variable Node Update & Hard Decision
            total_c2v = np.sum(c2v_mat, axis=0)
            post_llrs = rx_llrs + total_c2v
            decoded_bits = (post_llrs < 0).astype(np.uint8)
            
            # 3. Early stopping syndrome check
            syn = np.dot(self.H, decoded_bits.astype(np.int32)) % 2
            if np.sum(syn) == 0:
                converged = True
                break
                
            v2c_mat = (post_llrs - c2v_mat) * self.H
                
        final_syn = np.dot(self.H, decoded_bits.astype(np.int32)) % 2
        final_syn_wt = float(np.sum(final_syn))
        
        return converged, decoded_bits, iters_used, init_syn_wt, final_syn_wt, post_llrs


def decode_ldpc_profile(
    hard_bits: np.ndarray,
    soft_llrs: Optional[np.ndarray],
    profile: FECProfile,
    alpha: float = 0.80,
    max_iter: int = 30
) -> DecoderResult:
    """
    Executes LDPC decoding for a registered LDPC profile.
    """
    start_time = time.perf_counter()
    n = profile.n or 128
    k = profile.k or 64
    block_sz = profile.block_size or 16
    
    H = construct_qc_ldpc_matrix(n=n, k=k, block_size=block_sz)
    codec = LDPCCodec(H, alpha=alpha, max_iter=max_iter)
    
    # Form soft LLRs (if only hard bits provided, create synthetic LLRs +5.0 / -5.0)
    if soft_llrs is not None and len(soft_llrs) >= n:
        input_llrs = soft_llrs[:n]
    else:
        # Map hard bits 0 -> +5.0, 1 -> -5.0
        input_llrs = np.where(hard_bits[:n] == 0, 5.0, -5.0).astype(np.float32)
        
    converged, decoded_bits, iters, init_wt, final_wt, post_llrs = codec.decode_min_sum(input_llrs)
    
    # Information bits: first K bits of systematic codeword
    info_bits = decoded_bits[:k]
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    
    return DecoderResult(
        success=converged,
        decoded_bits=info_bits,
        decoded_soft_info=post_llrs[:k],
        metrics={
            "fec_family": "ldpc",
            "profile_id": profile.profile_id,
            "code_rate": profile.rate,
            "converged": bool(converged),
            "iterations_used": int(iters),
            "initial_syndrome_weight": float(init_wt),
            "final_syndrome_weight": float(final_wt),
            "parity_check_success": bool(converged)
        },
        failure_reason=None if converged else f"Non-convergence after {iters} iterations (final syndrome weight={final_wt})",
        decoder_name="normalized_min_sum",
        profile_id=profile.profile_id,
        runtime_ms=elapsed_ms
    )
