"""
LDPC (Low-Density Parity-Check) Block Encoder and Syndrome Validator for ASTRA FEC Engine.
Implements systematic Quasi-Cyclic / Sparse parity-check codes with exact matrix ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np

from .padding import segment_into_blocks


@dataclass
class LDPCMatrixProfile:
    """Specification of an LDPC code profile with Generator (G) and Parity-Check (H) matrices."""
    profile_id: str
    n: int                    # Codeword bit length
    k: int                    # Information bit length
    nominal_code_rate: float
    H: np.ndarray             # (n-k) x n binary parity-check matrix
    G: np.ndarray             # k x n binary generator matrix
    P: np.ndarray             # (n-k) x k parity submatrix where H = [P | I_(n-k)], G = [I_k | P^T]


def generate_systematic_ldpc_profile(
    profile_id: str,
    n: int,
    k: int,
    seed: int = 42,
    column_weight: int = 3,
) -> LDPCMatrixProfile:
    """Generate a valid, deterministic systematic LDPC profile [H = (P | I), G = (I | P^T)].

    Args:
        profile_id: Identifier name.
        n: Total codeword length.
        k: Message length.
        seed: Deterministic seed for parity structure.
        column_weight: Average weight of parity matrix columns.

    Returns:
        LDPCMatrixProfile instance.
    """
    m = n - k
    rng = np.random.default_rng(seed)

    # Generate sparse random parity matrix P of size m x k
    P = np.zeros((m, k), dtype=np.uint8)
    for col in range(k):
        # Choose column_weight random distinct row indices
        rows = rng.choice(m, size=min(column_weight, m), replace=False)
        P[rows, col] = 1

    # Ensure no all-zero rows in P
    for row in range(m):
        if np.sum(P[row, :]) == 0:
            col = int(rng.integers(0, k))
            P[row, col] = 1

    # Form H = [P | I_m] of size m x n
    I_m = np.eye(m, dtype=np.uint8)
    H = np.hstack([P, I_m])

    # Form G = [I_k | P^T] of size k x n
    I_k = np.eye(k, dtype=np.uint8)
    P_T = P.T  # size k x m
    G = np.hstack([I_k, P_T])

    # Mathematical identity verification: H * G^T = P * I_k^T + I_m * (P^T)^T = P + P = 0 mod 2
    synd_check = (H @ G.T) % 2
    assert np.all(synd_check == 0), "LDPC matrix generation identity H * G^T = 0 failed!"

    return LDPCMatrixProfile(
        profile_id=profile_id,
        n=n,
        k=k,
        nominal_code_rate=round(k / n, 6),
        H=H,
        G=G,
        P=P,
    )


# Standard Built-In LDPC Profiles
LDPC_PROFILES: dict[str, LDPCMatrixProfile] = {
    "ldpc_n128_k64_r12": generate_systematic_ldpc_profile("ldpc_n128_k64_r12", n=128, k=64, seed=1001, column_weight=3),
    "ldpc_n256_k128_r12": generate_systematic_ldpc_profile("ldpc_n256_k128_r12", n=256, k=128, seed=1002, column_weight=3),
    "ldpc_n512_k256_r12": generate_systematic_ldpc_profile("ldpc_n512_k256_r12", n=512, k=256, seed=1003, column_weight=3),
    "ldpc_n96_k64_r23": generate_systematic_ldpc_profile("ldpc_n96_k64_r23", n=96, k=64, seed=1004, column_weight=3),
}


class LDPCCode:
    """Low-Density Parity-Check (LDPC) systematic block encoder."""

    def __init__(self, profile: str | LDPCMatrixProfile = "ldpc_n128_k64_r12"):
        """Initialize LDPCCode with a matrix profile.

        Args:
            profile: Profile name string or LDPCMatrixProfile instance.
        """
        if isinstance(profile, LDPCMatrixProfile):
            self.profile = profile
        elif profile in LDPC_PROFILES:
            self.profile = LDPC_PROFILES[profile]
        else:
            available = ", ".join(LDPC_PROFILES.keys())
            raise ValueError(f"Unknown LDPC profile '{profile}'. Available: {available}")

        self.n = self.profile.n
        self.k = self.profile.k
        self.nominal_rate = self.profile.nominal_code_rate
        self.H = self.profile.H
        self.G = self.profile.G
        self.P = self.profile.P

    def compute_syndrome(self, codeword: np.ndarray) -> np.ndarray:
        """Compute syndrome s = H * c^T (mod 2). For valid codeword, s must be all zeros."""
        if len(codeword) != self.n:
            raise ValueError(f"Codeword length ({len(codeword)}) must equal n ({self.n})")
        return (self.H @ codeword) % 2

    def is_valid_codeword(self, codeword: np.ndarray) -> bool:
        """Check if codeword satisfies parity-check equations H * c^T = 0 mod 2."""
        s = self.compute_syndrome(codeword)
        return bool(np.all(s == 0))

    def encode(
        self,
        bits: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, int, list[dict[str, Any]], dict[str, Any]]:
        """Encode input bits into systematic LDPC codewords.

        Args:
            bits: 1D NumPy uint8 array of input bits (0 and 1).

        Returns:
            tuple (encoded_bits, padding_bits, total_pad_length, block_boundaries, parameters)
        """
        if not isinstance(bits, np.ndarray):
            bits = np.asarray(bits, dtype=np.uint8)
        if bits.ndim != 1:
            raise ValueError(f"Input bits must be 1D, got shape {bits.shape}")

        # 1. Segment input into k-bit message blocks
        blocks, pad_bits, pad_len, block_meta = segment_into_blocks(
            bits=bits,
            block_size=self.k,
            pad_final=True,
            pad_value=0,
        )

        encoded_blocks: list[np.ndarray] = []
        current_offset = 0

        for idx, m_block in enumerate(blocks):
            # Systematic encoding: c = m * G mod 2 = [m | m * P^T mod 2]
            # Parity bits p = (m @ P.T) % 2 of length (n-k)
            parity_bits = (m_block @ self.P.T) % 2
            codeword = np.concatenate([m_block, parity_bits]).astype(np.uint8)
            encoded_blocks.append(codeword)

            # Update block metadata with encoded boundaries and syndrome check
            syndrome = self.compute_syndrome(codeword)
            is_valid = bool(np.all(syndrome == 0))
            block_meta[idx]["encoded_start_bit"] = current_offset
            block_meta[idx]["encoded_bit_length"] = self.n
            block_meta[idx]["parity_check_valid"] = is_valid
            block_meta[idx]["syndrome_weight"] = int(np.sum(syndrome))
            current_offset += self.n

        encoded_arr = (
            np.concatenate(encoded_blocks) if encoded_blocks else np.empty(0, dtype=np.uint8)
        )

        params = {
            "profile_id": self.profile.profile_id,
            "n": self.n,
            "k": self.k,
            "parity_bits_per_block": self.n - self.k,
            "nominal_code_rate": round(self.nominal_rate, 6),
            "h_shape": list(self.H.shape),
            "g_shape": list(self.G.shape),
        }

        return encoded_arr, pad_bits, pad_len, block_meta, params

    def decode_reference(
        self,
        encoded_bits: np.ndarray,
        original_bit_length: int,
        block_boundaries: list[dict[str, Any]] | None = None,
    ) -> np.ndarray:
        """Reference Systematic LDPC Decoder for generator integrity testing.

        Extracts information bits m from systematic positions [0:k] of each n-bit block
        and verifies that each block is a valid codeword (H * c^T == 0).

        Args:
            encoded_bits: Received bitstream.
            original_bit_length: Expected original information bit count.
            block_boundaries: Block boundary telemetry.

        Returns:
            Decoded original bit array of length `original_bit_length`.
        """
        num_blocks = len(encoded_bits) // self.n
        recovered_chunks: list[np.ndarray] = []

        for b_idx in range(num_blocks):
            block = encoded_bits[b_idx * self.n : (b_idx + 1) * self.n]
            # Verify syndrome
            if not self.is_valid_codeword(block):
                raise ValueError(f"LDPC block {b_idx} failed parity check (invalid codeword)!")
            # Extract systematic message bits (first k bits)
            m = block[: self.k]
            recovered_chunks.append(m)

        all_recovered = (
            np.concatenate(recovered_chunks) if recovered_chunks else np.empty(0, dtype=np.uint8)
        )
        return all_recovered[:original_bit_length]
