"""
ASTRA Stage 15: Synthetic Test Record Generators.

Generates realistic multi-stage analysis records from Stages 1-14:
1. Perfect synthetic recovery ("hi hello")
2. Low-SNR recovery
3. Wrong early model top-1 (downstream CRC override)
4. Tied candidate pipelines
5. No validation (missing CRC)
6. Contradictory evidence (good sync/demod, CRC fails)
7. User override in expert mode
"""

from typing import Dict, Any, List


def create_synthetic_perfect_record() -> Dict[str, Any]:
    """Generates a complete Stage 1-14 record recovering 'hi hello'."""
    return {
        "signal_id": "SIG_SYNTH_001_HI_HELLO",
        "sample_rate": 192000.0,
        "stage_1_signal": {
            "snr_db": 18.5,
            "duration_s": 0.5,
            "center_freq_hz": 0.0
        },
        "stage_2_dsp": {
            "symbol_rate_candidates": [9600.0, 4800.0, 19200.0],
            "spectral_kurtosis": 1.45,
            "estimated_sps": 20.0
        },
        "stage_3_fusion": {
            "predicted_class": "QPSK",
            "probabilities": {"QPSK": 0.82, "8PSK": 0.11, "BPSK": 0.05, "16QAM": 0.02},
            "1d_resnet": {"QPSK": 0.84, "8PSK": 0.10},
            "2d_cnn": {"QPSK": 0.79, "8PSK": 0.12},
            "rf_dsp": {"family": "PSK", "confidence": 0.94}
        },
        "stage_4_constellation": {
            "num_clusters": 4,
            "cluster_variance": 0.025,
            "radial_symmetry_score": 0.91,
            "best_match": "QPSK"
        },
        "stage_5_candidates": {
            "top_candidates": [
                {"candidate_id": "cand_1", "modulation": "QPSK", "symbol_rate": 9600.0, "score": 0.88},
                {"candidate_id": "cand_2", "modulation": "8PSK", "symbol_rate": 9600.0, "score": 0.52},
                {"candidate_id": "cand_3", "modulation": "QPSK", "symbol_rate": 4800.0, "score": 0.41}
            ]
        },
        "stage_6_sync": {
            "cfo_hz": 1186.4,
            "residual_cfo_hz": 12.1,
            "timing_lock_metric": 0.93,
            "carrier_lock_metric": 0.91,
            "status": "LOCKED"
        },
        "stage_7_demod": {
            "evm": 0.082,
            "snr_post_demod": 19.1,
            "mean_llr_magnitude": 3.8,
            "best_phase_variant": "rot90",
            "phase_variants_tested": ["rot0", "rot90", "rot180", "rot270"]
        },
        "stage_8_interleaver": {
            "detected_interleaver": "Block 16x16",
            "confidence": 0.85,
            "structural_period": 256
        },
        "stage_9_fec": {
            "detected_fec": "Convolutional K7 R1/2",
            "soft_viterbi_metric": 0.12,
            "syndrome_valid": True,
            "reencoding_agreement_pct": 99.4,
            "convergence": True
        },
        "stage_10_validation": {
            "crc_passed": True,
            "crc_pass_count": 8,
            "crc_total_checked": 8,
            "crc_type": "CRC-16-CCITT",
            "validation_score": 0.98,
            "frame_sync_detected": True
        },
        "stage_11_ranking": {
            "best_pipeline_id": "pipe_qpsk_9600_b16_conv7",
            "candidates": [
                {
                    "pipeline_id": "pipe_qpsk_9600_b16_conv7",
                    "score": 0.965,
                    "modulation": "QPSK",
                    "symbol_rate": 9600.0,
                    "fec_scheme": "Convolutional K7 R1/2",
                    "interleaver": "Block 16x16",
                    "shap_contributions": {"crc_pass": 0.42, "fec_convergence": 0.28, "evm": 0.15}
                },
                {
                    "pipeline_id": "pipe_8psk_9600_b16_conv7",
                    "score": 0.380,
                    "modulation": "8PSK",
                    "symbol_rate": 9600.0,
                    "fec_scheme": "Convolutional K7 R1/2",
                    "interleaver": "Block 16x16"
                },
                {
                    "pipeline_id": "pipe_qpsk_4800_raw",
                    "score": 0.210,
                    "modulation": "QPSK",
                    "symbol_rate": 4800.0,
                    "fec_scheme": "None",
                    "interleaver": "None"
                }
            ]
        },
        "stage_12_bitstream": {
            "frame_length_bits": 512,
            "autocorrelation_peak": 0.88,
            "periodicity_score": 0.94,
            "entropy": 0.98,
            "sync_pattern": "0x1ACFFC1D"
        },
        "stage_13_transformer": {
            "regions": [
                {"region": "SYNC", "start": 0, "end": 31, "probability": 0.96},
                {"region": "HEADER", "start": 32, "end": 95, "probability": 0.89},
                {"region": "PAYLOAD", "start": 96, "end": 479, "probability": 0.95},
                {"region": "CRC", "start": 480, "end": 511, "probability": 0.92}
            ]
        },
        "stage_14_payload": {
            "extracted_payload_hex": "68692068656c6c6f",
            "decoded_text": "hi hello",
            "is_valid_utf8": True,
            "payload_length_bytes": 8,
            "protocol_profile": None,
            "parsed_fields": {
                "length": 8,
                "seq_num": 1,
                "flags": "0x00"
            }
        }
    }


def create_low_snr_record() -> Dict[str, Any]:
    """Generates low-SNR case where early ML is uncertain but downstream CRC validates."""
    rec = create_synthetic_perfect_record()
    rec["signal_id"] = "SIG_LOW_SNR_002"
    rec["stage_1_signal"]["snr_db"] = 4.2
    rec["stage_3_fusion"]["probabilities"] = {"QPSK": 0.44, "8PSK": 0.38, "BPSK": 0.18}
    rec["stage_6_sync"]["timing_lock_metric"] = 0.72
    rec["stage_7_demod"]["evm"] = 0.22
    rec["stage_10_validation"]["crc_pass_count"] = 6
    rec["stage_10_validation"]["crc_total_checked"] = 8
    return rec


def create_wrong_top1_model_record() -> Dict[str, Any]:
    """Generates case where early model prefers 8PSK, but QPSK wins via CRC."""
    rec = create_synthetic_perfect_record()
    rec["signal_id"] = "SIG_WRONG_EARLY_003"
    rec["stage_3_fusion"]["predicted_class"] = "8PSK"
    rec["stage_3_fusion"]["probabilities"] = {"8PSK": 0.65, "QPSK": 0.28, "BPSK": 0.07}
    return rec


def create_tied_candidates_record() -> Dict[str, Any]:
    """Generates case where top 2 candidates are virtually tied."""
    rec = create_synthetic_perfect_record()
    rec["signal_id"] = "SIG_TIED_004"
    cands = rec["stage_11_ranking"]["candidates"]
    cands[0]["score"] = 0.810
    cands[1]["score"] = 0.805
    cands[1]["modulation"] = "QPSK"
    cands[1]["fec_scheme"] = "Convolutional K7 R3/4"
    return rec


def create_missing_crc_record() -> Dict[str, Any]:
    """Generates case where no CRC was tested (missing validation evidence)."""
    rec = create_synthetic_perfect_record()
    rec["signal_id"] = "SIG_NO_CRC_005"
    rec["stage_10_validation"]["crc_passed"] = None
    rec["stage_10_validation"]["crc_pass_count"] = 0
    rec["stage_10_validation"]["crc_total_checked"] = 0
    rec["stage_10_validation"]["crc_type"] = None
    rec["stage_10_validation"]["validation_score"] = 0.45
    return rec


def create_contradictory_record() -> Dict[str, Any]:
    """Generates case where sync/demod are perfect but CRC completely fails."""
    rec = create_synthetic_perfect_record()
    rec["signal_id"] = "SIG_CONTRADICTORY_006"
    rec["stage_10_validation"]["crc_passed"] = False
    rec["stage_10_validation"]["crc_pass_count"] = 0
    rec["stage_10_validation"]["crc_total_checked"] = 12
    rec["stage_10_validation"]["validation_score"] = 0.05
    rec["stage_11_ranking"]["candidates"][0]["score"] = 0.42
    return rec


def create_user_override_record() -> Dict[str, Any]:
    """Generates expert user override contradicting evidence."""
    rec = create_synthetic_perfect_record()
    rec["signal_id"] = "SIG_USER_OVERRIDE_007"
    rec["user_overrides"] = {
        "modulation": "16QAM"
    }
    return rec
