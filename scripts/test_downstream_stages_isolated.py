"""
test_downstream_stages_isolated.py
Isolated unit and round-trip verification for Stages 7, 8, 9, 10 before Rung E.
"""

import sys
import os
from pathlib import Path
import numpy as np

WORKSPACE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WORKSPACE))

from astra_demodulation.src.models import DemodulationResult, DemodulationVariant, DemodulationQuality
from astra_interleaver.src.identity import interleave_identity, deinterleave_identity
from astra_interleaver.src.block import interleave_block, create_block_mapping
from astra_interleaver.src.helical import interleave_helical, create_helical_mapping
from astra_interleaver.src.pseudo_random import interleave_pseudorandom, create_pseudorandom_mapping
from astra_interleaver.src.convolutional import ConvolutionalInterleaverEngine, ConvolutionalDeinterleaverEngine
from astra_interleaver.src.router import execute_deinterleaver
from astra_interleaver.src.inference import InterleaverTestingEngine

from astra_fec.src.models import FECProfile, FECFamily
from astra_fec.src.router import execute_decoder
from astra_fec.src.profiles import get_profile_registry
from astra_fec.src.inference import FECTestingEngine

from astra_validation.src.inference import ValidationEngine
from astra_validation.src.models import EvidenceCheckState


def test_interleaver_roundtrips():
    print("=" * 70)
    print("1. TESTING ISOLATED INTERLEAVER ROUND-TRIPS (Hard Bits & Soft LLRs)")
    print("=" * 70)
    
    np.random.seed(42)
    n_bits = 512
    orig_bits = np.random.randint(0, 2, n_bits, dtype=np.uint8)
    orig_llrs = (1.0 - 2.0 * orig_bits.astype(np.float32)) * 5.0  # clean LLRs

    # 1.1 NONE / Identity
    int_hard, int_llrs = interleave_identity(orig_bits, orig_llrs)
    de_hard, de_llrs, _ = deinterleave_identity(int_hard, int_llrs)
    assert np.array_equal(orig_bits, de_hard), "Identity hard bit mismatch"
    assert np.array_equal(orig_llrs, de_llrs), "Identity soft LLR mismatch"
    print("  [PASS] NONE / Identity: Exact bit & LLR round-trip confirmed.")

    # 1.2 Block Interleaver
    for rows, cols in [(16, 16), (8, 32), (32, 16)]:
        for orient in ["row_to_column", "column_to_row"]:
            int_hard, int_llrs = interleave_block(orig_bits, orig_llrs, rows=rows, cols=cols, orientation=orient)
            de_hard, de_llrs, _, _, _ = execute_deinterleaver(
                family="block",
                parameters={"rows": rows, "cols": cols, "orientation": orient},
                hard_bits=int_hard,
                soft_llrs=int_llrs
            )
            assert np.array_equal(orig_bits[:len(de_hard)], de_hard), f"Block ({rows}x{cols}, {orient}) hard mismatch"
            assert np.array_equal(orig_llrs[:len(de_llrs)], de_llrs), f"Block ({rows}x{cols}, {orient}) soft mismatch"
            print(f"  [PASS] BLOCK ({rows:2d}x{cols:2d}, {orient:14s}): Exact round-trip ({len(de_hard)} bits).")

    # 1.3 Helical Interleaver
    for rows, cols, step in [(8, 16, 1), (16, 16, 3), (8, 32, 1)]:
        for orient in ["row_diagonal", "diagonal_row"]:
            int_hard, int_llrs = interleave_helical(orig_bits, orig_llrs, rows=rows, cols=cols, step=step, orientation=orient)
            de_hard, de_llrs, _, _, _ = execute_deinterleaver(
                family="helical",
                parameters={"rows": rows, "cols": cols, "step": step, "orientation": orient},
                hard_bits=int_hard,
                soft_llrs=int_llrs
            )
            assert np.array_equal(orig_bits[:len(de_hard)], de_hard), f"Helical ({rows}x{cols}, step={step}) mismatch"
            assert np.array_equal(orig_llrs[:len(de_llrs)], de_llrs), f"Helical ({rows}x{cols}, step={step}) LLR mismatch"
            print(f"  [PASS] HELICAL (R={rows:2d}, C={cols:2d}, step={step}, {orient}): Exact round-trip ({len(de_hard)} bits).")

    # 1.4 Pseudo-Random Interleaver
    for seed in [42, 1234, 9999]:
        for block_len in [256, 512]:
            int_hard, int_llrs = interleave_pseudorandom(orig_bits[:block_len], orig_llrs[:block_len], length=block_len, seed=seed)
            de_hard, de_llrs, _, _, _ = execute_deinterleaver(
                family="pseudo_random",
                parameters={"length": block_len, "seed": seed, "algorithm": "pcg64"},
                hard_bits=int_hard,
                soft_llrs=int_llrs
            )
            assert np.array_equal(orig_bits[:block_len], de_hard), f"PR (seed={seed}) hard mismatch"
            assert np.array_equal(orig_llrs[:block_len], de_llrs), f"PR (seed={seed}) soft mismatch"
            print(f"  [PASS] PSEUDO-RANDOM (L={block_len}, seed={seed}): Exact round-trip ({len(de_hard)} bits).")

    # 1.5 Convolutional Interleaver
    for branch_cnt, delay_st in [(4, 2), (6, 1)]:
        conv_int = ConvolutionalInterleaverEngine(branch_count=branch_cnt, delay_step=delay_st)
        conv_deint = ConvolutionalDeinterleaverEngine(branch_count=branch_cnt, delay_step=delay_st)
        
        # Convolutional has pipeline latency = B * (B - 1) * M
        latency = conv_deint.latency_bits
        pad_bits = np.concatenate([orig_bits, np.zeros(latency, dtype=np.uint8)])
        
        int_bits = conv_int.process_array(pad_bits).astype(np.uint8)
        de_bits = conv_deint.process_array(int_bits).astype(np.uint8)
        
        recovered_valid = de_bits[latency : latency + len(orig_bits)]
        assert np.array_equal(orig_bits, recovered_valid), f"Convolutional (B={branch_cnt}, M={delay_st}) mismatch"
        print(f"  [PASS] CONVOLUTIONAL (B={branch_cnt}, M={delay_st}): Latency-compensated exact recovery ({len(orig_bits)} bits).")


def test_fec_isolated():
    print("\n" + "=" * 70)
    print("2. TESTING ISOLATED FEC ENCODING & DECODING (All Supported Families)")
    print("=" * 70)
    
    reg = get_profile_registry()

    # 2.1 NO_FEC
    bits = np.random.randint(0, 2, 256, dtype=np.uint8)
    res = execute_decoder(profile=None, hard_bits=bits)
    assert res.success and np.array_equal(res.decoded_bits, bits)
    print("  [PASS] NO_FEC: Passed through 256 bits identically.")

    # 2.2 Convolutional / Viterbi
    from astra_fec.src.convolutional import encode_convolutional
    from astra_fec.src.viterbi import viterbi_decode_hard
    msg_bits = np.random.randint(0, 2, 128, dtype=np.uint8)
    encoded = encode_convolutional(msg_bits, constraint_length=7, generators_octal=[0o171, 0o133], termination="terminated")
    
    # Clean decode
    dec_clean, _, _, _ = viterbi_decode_hard(encoded, constraint_length=7, generators_octal=[0o171, 0o133], termination="terminated")
    assert np.array_equal(msg_bits, dec_clean[:len(msg_bits)]), "Viterbi clean decode mismatch"
    print(f"  [PASS] CONVOLUTIONAL / VITERBI (K=7, Rate 1/2): Clean decode exact match ({len(msg_bits)} bits).")
    
    # With channel errors (3 random bit flips)
    noisy_enc = encoded.copy()
    flip_pos = [10, 50, 100]
    for p in flip_pos:
        noisy_enc[p] ^= 1
    dec_noisy, _, _, _ = viterbi_decode_hard(noisy_enc, constraint_length=7, generators_octal=[0o171, 0o133], termination="terminated")
    assert np.array_equal(msg_bits, dec_noisy[:len(msg_bits)]), "Viterbi error correction failed"
    print(f"  [PASS] CONVOLUTIONAL / VITERBI (K=7, Rate 1/2): Corrected 3 channel bit errors successfully.")

    # 2.3 Reed-Solomon
    from astra_fec.src.reed_solomon import ReedSolomonCodec
    rs_codec = ReedSolomonCodec(n=255, k=223, m=8)
    rs_data = [int(x) for x in np.random.randint(0, 256, 223)]
    rs_encoded = rs_codec.encode(rs_data)
    
    # Inject 8 symbol errors (within t=16 capability)
    rs_noisy = [int(x) for x in rs_encoded]
    for pos in range(8):
        rs_noisy[pos * 10] = int((rs_noisy[pos * 10] + 50) % 256)
    success, dec_symbols, corrected = rs_codec.decode_codeword(rs_noisy)
    assert success and list(rs_data) == list(dec_symbols[:223]), "Reed-Solomon error correction failed"
    print(f"  [PASS] REED-SOLOMON (255, 223, t=16): Corrected 8 symbol errors, exact data match.")

    # 2.4 Profile execution via FECTestingEngine
    fec_eng = FECTestingEngine()
    test_cand = {
        "candidate_id": "test_sig_001",
        "demod_variant_id": "var0",
        "interleaver_candidate_id": "int_none_0001",
        "deinterleaved_hard_bits": encoded,
    }
    fec_test_res = fec_eng.test_candidates(test_cand)
    assert len(fec_test_res.surviving_candidates) > 0, "FEC Candidate engine produced no survivors"
    conv_cand = next((c for c in fec_test_res.surviving_candidates if c.fec_family == "convolutional"), None)
    assert conv_cand is not None, "Convolutional candidate not found in surviving candidates"
    assert np.array_equal(msg_bits, conv_cand.decoded_hard_bits[:len(msg_bits)]), "FECTestingEngine failed to decode convolutional candidate"
    print(f"  [PASS] FECTestingEngine: Successfully generated and scored {len(fec_test_res.surviving_candidates)} surviving hypotheses.")


def test_stage10_validation():
    print("\n" + "=" * 70)
    print("3. TESTING STAGE 10 VALIDATION ENGINE")
    print("=" * 70)
    
    from astra_validation.src.crc import CRCCalculator, load_crc_profiles
    
    val_eng = ValidationEngine()
    crc_profiles = load_crc_profiles()
    crc_prof = crc_profiles.get("crc16_ccitt_false", list(crc_profiles.values())[0])
    crc_calc = CRCCalculator(crc_prof)
    
    # Construct a valid frame: 16-bit sync (0xEB90) + 32-bit payload + 16-bit CRC
    sync_word = np.array([1,1,1,0,1,0,1,1,1,0,0,1,0,0,0,0], dtype=np.uint8)  # 0xEB90
    payload = np.array([0,1,1,0,1,0,0,0, 0,1,1,0,1,0,0,1, 0,0,1,0,0,0,0,0, 0,1,1,0,1,0,0,0], dtype=np.uint8)  # 'hi h'
    
    # CRC over sync+payload
    data_bits = np.concatenate([sync_word, payload])
    crc_val = crc_calc.compute(data_bits)
    crc_bits = np.unpackbits(np.array([crc_val >> 8, crc_val & 0xFF], dtype=np.uint8))
    
    valid_frame = np.concatenate([data_bits, crc_bits])
    
    # Repeat frame 4 times
    frame_stream = np.tile(valid_frame, 4)
    
    fec_mock = {
        "candidate_id": "test_sig_001",
        "demod_variant_id": "var0",
        "interleaver_candidate_id": "int_none_0001",
        "fec_candidate_id": "fec_none_0001",
        "decoded_hard_bits": frame_stream,
    }
    
    val_res = val_eng.validate(fec_mock, context={"candidate_frame_lengths": [len(valid_frame)]})
    print(f"DEBUG: validation result = {val_res.to_dict()}")
    assert val_res.overall_validation_score > 0.4, f"Validation score too low: {val_res.overall_validation_score}"
    crc_passes = [c for c in val_res.crc_results if c.check_state == EvidenceCheckState.PASS]
    assert len(crc_passes) > 0, "CRC did not PASS on valid frame stream"
    print(f"  [PASS] Stage 10 Validation Engine: Status={val_res.validation_status.value}, Score={val_res.overall_validation_score:.3f}, CRC={crc_passes[0].profile_name}")


if __name__ == "__main__":
    test_interleaver_roundtrips()
    test_fec_isolated()
    test_stage10_validation()
    print("\n" + "=" * 70)
    print("ALL ISOLATED STAGE 7 -> 8 -> 9 -> 10 TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)
