"""
reed_solomon.py
Reed-Solomon codec over Galois Field GF(2^m).
Features Berlekamp-Massey syndrome decoding, Chien search, Forney error value computation, and shortened RS handling.
"""

from typing import Tuple, Optional, Dict, Any, List
import time
import numpy as np
from .models import DecoderResult, FECProfile


class GaloisField:
    """
    Galois Field GF(2^m) arithmetic engine using precomputed log and exp lookup tables.
    """
    def __init__(self, m: int = 8, prim_poly: int = 0x11D):
        self.m = int(m)
        self.size = 1 << self.m
        self.prim_poly = int(prim_poly)
        
        self.gf_exp = np.zeros(self.size * 2, dtype=np.int32)
        self.gf_log = np.zeros(self.size, dtype=np.int32)
        
        # Initialize tables
        x = 1
        for i in range(self.size - 1):
            self.gf_exp[i] = x
            self.gf_log[x] = i
            x <<= 1
            if x & self.size:
                x ^= self.prim_poly
                
        for i in range(self.size - 1, self.size * 2):
            self.gf_exp[i] = self.gf_exp[i - (self.size - 1)]

    def add(self, a: int, b: int) -> int:
        return int(a ^ b)

    def sub(self, a: int, b: int) -> int:
        return int(a ^ b)

    def mul(self, a: int, b: int) -> int:
        if a == 0 or b == 0:
            return 0
        return int(self.gf_exp[self.gf_log[a] + self.gf_log[b]])

    def div(self, a: int, b: int) -> int:
        if a == 0:
            return 0
        if b == 0:
            raise ZeroDivisionError("Division by zero in GF(2^m)")
        diff = self.gf_log[a] - self.gf_log[b]
        if diff < 0:
            diff += self.size - 1
        return int(self.gf_exp[diff])

    def inv(self, a: int) -> int:
        if a == 0:
            raise ZeroDivisionError("Inverse of zero in GF(2^m)")
        return int(self.gf_exp[(self.size - 1) - self.gf_log[a]])

    def poly_mul(self, p: List[int], q: List[int]) -> List[int]:
        r = [0] * (len(p) + len(q) - 1)
        for i, a in enumerate(p):
            for j, b in enumerate(q):
                r[i + j] ^= self.mul(a, b)
        return r

    def poly_eval(self, poly: List[int], x: int) -> int:
        y = 0
        for coeff in poly:
            y = self.mul(y, x) ^ coeff
        return y


class ReedSolomonCodec:
    """
    Reed-Solomon RS(n, k) encoder and decoder over GF(2^m).
    """
    def __init__(self, n: int = 255, k: int = 223, m: int = 8, prim_poly: int = 0x11D, fcr: int = 0):
        self.n = int(n)
        self.k = int(k)
        self.m = int(m)
        self.fcr = int(fcr)
        self.gf = GaloisField(m, prim_poly)
        self.n_sym = self.n - self.k  # 2t parity symbols
        self.t = self.n_sym // 2
        
        # Generator polynomial g(x) = prod_{i=0}^{2t-1} (x - alpha^(i + fcr))
        self.gen_poly = [1]
        for i in range(self.n_sym):
            root = self.gf.gf_exp[i + self.fcr]
            self.gen_poly = self.gf.poly_mul(self.gen_poly, [1, root])

    def encode(self, msg_symbols: List[int]) -> List[int]:
        """Encodes k message symbols to an n-symbol RS codeword."""
        if len(msg_symbols) != self.k:
            raise ValueError(f"Message length ({len(msg_symbols)}) must equal k ({self.k})")
            
        # Shift message by n_sym
        msg_shifted = list(msg_symbols) + [0] * self.n_sym
        gen = self.gen_poly
        
        # Polynomial division to find remainder
        rem = list(msg_shifted)
        for i in range(len(msg_symbols)):
            coef = rem[i]
            if coef != 0:
                for j in range(1, len(gen)):
                    rem[i + j] ^= self.gf.mul(gen[j], coef)
                    
        parity = rem[len(msg_symbols):]
        return list(msg_symbols) + parity

    def decode_codeword(self, rx_symbols: List[int]) -> Tuple[bool, List[int], int]:
        """
        Decodes a single received n-symbol codeword.
        
        Returns:
            (success, corrected_msg_symbols, num_errors_corrected)
        """
        if len(rx_symbols) != self.n:
            raise ValueError(f"Received codeword length ({len(rx_symbols)}) must equal n ({self.n})")
            
        # 1. Compute syndromes S_i = rx(alpha^(i + fcr)) for i = 0 ... 2t-1
        syndromes = []
        for i in range(self.n_sym):
            alpha_pow = self.gf.gf_exp[i + self.fcr]
            s = self.gf.poly_eval(rx_symbols, alpha_pow)
            syndromes.append(s)
            
        # Error-free check
        if all(s == 0 for s in syndromes):
            return True, rx_symbols[:self.k], 0
            
        # 2. Berlekamp-Massey algorithm to find error locator polynomial Lambda(x)
        # Lambda(x) = 1 + L1*x + ... + Ll*x^l
        Lambda = [1]
        B = [1]
        L = 0
        m_step = 1
        b_val = 1
        
        for r in range(self.n_sym):
            # Compute discrepancy delta
            delta = syndromes[r]
            for j in range(1, L + 1):
                delta ^= self.gf.mul(Lambda[j], syndromes[r - j])
                
            if delta == 0:
                m_step += 1
            else:
                T_poly = list(Lambda)
                # scale factor: delta / b_val
                scale = self.gf.div(delta, b_val)
                # term: scale * x^m_step * B(x)
                scaled_B = [0] * m_step + [self.gf.mul(x, scale) for x in B]
                
                # Lambda = Lambda ^ scaled_B
                max_len = max(len(Lambda), len(scaled_B))
                new_Lambda = [0] * max_len
                for i in range(len(Lambda)):
                    new_Lambda[i] ^= Lambda[i]
                for i in range(len(scaled_B)):
                    new_Lambda[i] ^= scaled_B[i]
                Lambda = new_Lambda
                
                if 2 * L <= r:
                    L = r + 1 - L
                    B = T_poly
                    b_val = delta
                    m_step = 1
                else:
                    m_step += 1
                    
        # Check degree of locator
        if L > self.t:
            return False, rx_symbols[:self.k], -1
            
        # 3. Chien search for error positions
        error_pos = []
        for i in range(self.n):
            # Test x = alpha^(-(n-1-i))
            alpha_inv = self.gf.gf_exp[(self.gf.size - 1) - (self.n - 1 - i) % (self.gf.size - 1)]
            # Evaluate Lambda at alpha_inv: Lambda(alpha_inv) == 0
            val = 0
            term = 1
            for j in range(len(Lambda)):
                val ^= self.gf.mul(Lambda[j], term)
                term = self.gf.mul(term, alpha_inv)
            if val == 0:
                error_pos.append(i)
                
        if len(error_pos) != L:
            return False, rx_symbols[:self.k], -1
            
        # 4. Forney algorithm for error values
        # Omega(x) = [Syndromes(x) * Lambda(x)] mod x^(2t)
        # Reverse syndromes for standard convolution
        Omega = self.gf.poly_mul(syndromes, Lambda)[:self.n_sym]
        
        # Formal derivative of Lambda: Lambda'(x)
        Lambda_prime = [0] * len(Lambda)
        for i in range(1, len(Lambda), 2):  # in GF(2), only odd power coefficients remain
            Lambda_prime[i - 1] = Lambda[i]
            
        corrected = list(rx_symbols)
        for pos in error_pos:
            # X_k = alpha^(n-1-pos)
            X_k = self.gf.gf_exp[(self.n - 1 - pos) % (self.gf.size - 1)]
            X_k_inv = self.gf.inv(X_k)
            
            # Omega(X_k_inv)
            omega_val = 0
            term = 1
            for j in range(len(Omega)):
                omega_val ^= self.gf.mul(Omega[j], term)
                term = self.gf.mul(term, X_k_inv)
                
            # Lambda'(X_k_inv)
            lambda_p_val = 0
            term = 1
            for j in range(len(Lambda_prime)):
                lambda_p_val ^= self.gf.mul(Lambda_prime[j], term)
                term = self.gf.mul(term, X_k_inv)
                
            if lambda_p_val == 0:
                return False, rx_symbols[:self.k], -1
                
            # Error magnitude = (Omega(X_k_inv) / Lambda'(X_k_inv)) * (X_k)^(1 - fcr)
            err_mag = self.gf.div(omega_val, lambda_p_val)
            if self.fcr != 1:
                factor = self.gf.gf_exp[((1 - self.fcr) * (self.n - 1 - pos)) % (self.gf.size - 1)]
                err_mag = self.gf.mul(err_mag, factor)
                
            corrected[pos] ^= err_mag
            
        return True, corrected[:self.k], len(error_pos)


def bits_to_symbols(bits: np.ndarray, m: int = 8) -> List[int]:
    """Packs 1D uint8 binary bits into integer symbols of m bits."""
    n_syms = len(bits) // m
    if n_syms == 0:
        return []
    usable = bits[:n_syms * m].reshape(n_syms, m)
    # Weights for binary conversion [2^(m-1), ..., 2^0]
    weights = (1 << np.arange(m - 1, -1, -1)).astype(np.int64)
    symbols = np.dot(usable.astype(np.int64), weights).tolist()
    return symbols


def symbols_to_bits(symbols: List[int], m: int = 8) -> np.ndarray:
    """Unpacks integer symbols into 1D uint8 binary array."""
    out_bits = []
    for s in symbols:
        for bit_idx in range(m - 1, -1, -1):
            out_bits.append((int(s) >> bit_idx) & 1)
    return np.array(out_bits, dtype=np.uint8)


def decode_reed_solomon_profile(
    hard_bits: np.ndarray,
    profile: FECProfile
) -> DecoderResult:
    """
    Executes Reed-Solomon decoding across full bitstream blocks.
    """
    start_time = time.perf_counter()
    m = profile.symbol_bits or 8
    n = profile.n or 255
    k = profile.k or 223
    fcr = profile.fcr or 0
    prim_poly = profile.prim_poly or 0x11D
    
    # Shortened RS handling
    short_n = profile.shortened_n
    short_k = profile.shortened_k
    is_shortened = (short_n is not None and short_k is not None)
    
    active_n = short_n if is_shortened else n
    active_k = short_k if is_shortened else k
    pad_len = (n - active_n) if is_shortened else 0
    
    codec = ReedSolomonCodec(n=n, k=k, m=m, prim_poly=prim_poly, fcr=fcr)
    
    symbols = bits_to_symbols(hard_bits, m=m)
    num_codewords = len(symbols) // active_n
    
    if num_codewords == 0:
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return DecoderResult(
            success=False,
            decoded_bits=np.array([], dtype=np.uint8),
            metrics={"fec_family": "reed_solomon", "error": "Insufficient bits for RS codeword"},
            failure_reason="Stream shorter than single RS codeword",
            decoder_name="reed_solomon",
            profile_id=profile.profile_id,
            runtime_ms=elapsed_ms
        )
        
    decoded_msg_symbols = []
    total_corrected_symbols = 0
    uncorrectable_count = 0
    
    for cw_idx in range(num_codewords):
        cw_syms = symbols[cw_idx * active_n : (cw_idx + 1) * active_n]
        if is_shortened:
            # Prepend virtual zeros
            cw_syms = [0] * pad_len + list(cw_syms)
            
        success, corrected_msg, n_err = codec.decode_codeword(cw_syms)
        if success:
            if is_shortened:
                # Strip virtual zeros from message
                corrected_msg = corrected_msg[pad_len:]
            decoded_msg_symbols.extend(corrected_msg)
            total_corrected_symbols += max(0, n_err)
        else:
            uncorrectable_count += 1
            decoded_msg_symbols.extend(cw_syms[pad_len : pad_len + active_k] if is_shortened else cw_syms[:active_k])
            
    decoded_bits = symbols_to_bits(decoded_msg_symbols, m=m)
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    
    overall_success = (uncorrectable_count == 0) and (num_codewords > 0)
    
    return DecoderResult(
        success=overall_success,
        decoded_bits=decoded_bits,
        metrics={
            "fec_family": "reed_solomon",
            "profile_id": profile.profile_id,
            "code_rate": round(float(active_k) / float(active_n), 4),
            "total_codewords": num_codewords,
            "uncorrectable_codewords": uncorrectable_count,
            "corrected_symbols": total_corrected_symbols,
            "syndrome_success": (uncorrectable_count == 0),
            "is_shortened": is_shortened
        },
        failure_reason=None if overall_success else f"{uncorrectable_count} uncorrectable RS codewords",
        decoder_name="reed_solomon",
        profile_id=profile.profile_id,
        runtime_ms=elapsed_ms
    )
