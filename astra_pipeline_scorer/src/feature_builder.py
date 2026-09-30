"""
feature_builder.py
Feature builder that transforms multi-stage telemetry into the fixed Stage 11 feature schema.
Ensures zero data leakage, consistent default handling, and identical representations
between training and inference.
"""

from typing import Any, Dict, List, Optional, Union
import numpy as np
from .feature_schema import PIPELINE_FEATURE_COLUMNS, FEATURE_DEFAULTS
from .categorical import (
    encode_modulation,
    encode_interleaver_family,
    encode_fec_family,
    encode_validation_status,
)


class PipelineFeatureBuilder:
    """
    Constructs fixed-dimension 1D feature vectors for Stage 11 XGBoost scoring.
    Extracts telemetry across Stage 5 -> 6 -> 7 -> 8 -> 9 -> 10.
    """

    def __init__(self):
        self.columns = list(PIPELINE_FEATURE_COLUMNS)
        self.defaults = dict(FEATURE_DEFAULTS)

    def extract_features(
        self,
        candidate_or_dict: Any,
        stage5_res: Optional[Dict[str, Any]] = None,
        stage6_res: Optional[Dict[str, Any]] = None,
        stage7_res: Optional[Dict[str, Any]] = None,
        stage8_res: Optional[Dict[str, Any]] = None,
        stage9_res: Optional[Dict[str, Any]] = None,
        stage10_res: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        """
        Build a dictionary of all features according to the fixed schema.
        Handles missing fields gracefully using documented default fallback policies.
        """
        # Unpack candidate if it's an object/dataclass
        if hasattr(candidate_or_dict, "__dict__"):
            d = getattr(candidate_or_dict, "__dict__", {})
        elif isinstance(candidate_or_dict, dict):
            d = candidate_or_dict
        else:
            d = {}

        # Merge explicit stage arguments with candidate inner dictionaries
        s5 = stage5_res or d.get("stage5_metrics", {}) or {}
        s6 = stage6_res or d.get("stage6_sync_metrics", {}) or {}
        s7 = stage7_res or d.get("stage7_demod_metrics", {}) or {}
        s8 = stage8_res or d.get("stage8_interleaver_metrics", {}) or {}
        s9 = stage9_res or d.get("stage9_fec_metrics", {}) or {}
        s10 = stage10_res or d.get("stage10_validation_metrics", {}) or d.get("validation_features", {}) or {}

        # Base parameters
        mod_name = d.get("modulation") or s5.get("modulation", "QPSK")
        inter_family = d.get("interleaver_family") or s8.get("interleaver_family", "none")
        fec_family = d.get("fec_family") or s9.get("fec_family", "none")
        val_status = d.get("validation_status") or s10.get("validation_status", "VALIDATION_INCONCLUSIVE")

        # Initialize output with defaults
        feats: Dict[str, float] = dict(self.defaults)

        # --- GROUP A: Stage 5 ---
        feats["mod_encoded"] = encode_modulation(mod_name)
        feats["modulation_probability"] = float(s5.get("modulation_probability", s5.get("score", self.defaults["modulation_probability"])))
        feats["symbol_rate_score"] = float(s5.get("symbol_rate_score", self.defaults["symbol_rate_score"]))
        feats["candidate_initial_score"] = float(s5.get("candidate_initial_score", s5.get("score", 0.5)))
        feats["rf_family_support"] = float(s5.get("rf_family_support", s5.get("rf_score", 0.5)))
        feats["constellation_support"] = float(s5.get("constellation_support", 0.5))
        feats["modulation_rank"] = float(s5.get("modulation_rank", 1.0))
        feats["baud_rank"] = float(s5.get("baud_rank", 1.0))
        feats["modulation_entropy"] = float(s5.get("modulation_entropy", 0.0))
        feats["modulation_margin"] = float(s5.get("modulation_margin", 0.0))
        feats["symbol_rate_margin"] = float(s5.get("symbol_rate_margin", 0.0))
        feats["samples_per_symbol"] = float(s5.get("samples_per_symbol", s5.get("sps", 4.0)))
        feats["stage5_fallback_used"] = 1.0 if s5.get("fallback_used", False) else 0.0

        # --- GROUP B: Stage 6 ---
        feats["timing_lock_score"] = float(s6.get("timing_lock_score", s6.get("timing_lock", 0.5)))
        feats["carrier_lock_score"] = float(s6.get("carrier_lock_score", s6.get("carrier_lock", 0.5)))
        feats["frequency_lock_score"] = float(s6.get("frequency_lock_score", 0.5))
        feats["overall_sync_score"] = float(s6.get("overall_sync_score", s6.get("quality_score", 0.5)))
        feats["estimated_cfo_hz"] = float(s6.get("estimated_cfo_hz", 0.0))
        feats["normalized_cfo"] = float(s6.get("normalized_cfo", 0.0))
        feats["residual_cfo_hz"] = float(s6.get("residual_cfo_hz", 0.0))
        feats["normalized_residual_cfo"] = float(s6.get("normalized_residual_cfo", 0.0))
        feats["phase_error_rms"] = float(s6.get("phase_error_rms", 0.1))
        feats["timing_error_variance"] = float(s6.get("timing_error_variance", 0.1))
        feats["estimated_sps"] = float(s6.get("estimated_sps", 4.0))
        feats["sps_error_relative"] = float(s6.get("sps_error_relative", 0.0))
        feats["constellation_quality_before"] = float(s6.get("constellation_quality_before", 0.5))
        feats["constellation_quality_after"] = float(s6.get("constellation_quality_after", 0.5))
        feats["constellation_improvement"] = float(s6.get("constellation_improvement", 0.0))

        # --- GROUP C: Stage 7 ---
        feats["evm_percent"] = float(s7.get("evm_percent", s7.get("evm", 15.0)))
        feats["evm_db"] = float(s7.get("evm_db", -15.0))
        feats["decision_distance_mean"] = float(s7.get("decision_distance_mean", 0.2))
        feats["decision_distance_std"] = float(s7.get("decision_distance_std", 0.1))
        feats["decision_margin_mean"] = float(s7.get("decision_margin_mean", 0.5))
        feats["mean_abs_llr"] = float(s7.get("mean_abs_llr", 2.0))
        feats["median_abs_llr"] = float(s7.get("median_abs_llr", 2.0))
        feats["low_confidence_bit_fraction"] = float(s7.get("low_confidence_bit_fraction", 0.1))
        feats["estimated_snr_db"] = float(s7.get("estimated_snr_db", 15.0))
        feats["noise_variance"] = float(s7.get("noise_variance", 0.05))
        feats["demod_quality_score"] = float(s7.get("demod_quality_score", 0.5))
        feats["ambiguity_variant_index"] = float(s7.get("ambiguity_variant_index", 0.0))
        feats["ambiguity_variant_count"] = float(s7.get("ambiguity_variant_count", 4.0))

        # --- GROUP D: Stage 8 ---
        feats["interleaver_family_encoded"] = encode_interleaver_family(inter_family)
        feats["interleaver_candidate_rank"] = float(s8.get("interleaver_candidate_rank", 1.0))
        feats["interleaver_structural_score"] = float(s8.get("interleaver_structural_score", 0.5))
        feats["interleaver_periodicity_score"] = float(s8.get("interleaver_periodicity_score", 0.0))
        feats["interleaver_autocorrelation_score"] = float(s8.get("interleaver_autocorrelation_score", 0.0))
        feats["interleaver_entropy"] = float(s8.get("interleaver_entropy", 1.0))
        feats["interleaver_bit_balance"] = float(s8.get("interleaver_bit_balance", 0.5))
        feats["interleaver_run_length_score"] = float(s8.get("interleaver_run_length_score", 1.5))
        feats["interleaver_remainder_fraction"] = float(s8.get("interleaver_remainder_fraction", 0.0))
        feats["interleaver_family_prior"] = float(s8.get("interleaver_family_prior", 0.5))
        feats["block_rows"] = float(s8.get("block_rows", 0.0))
        feats["block_cols"] = float(s8.get("block_cols", 0.0))
        feats["conv_branches"] = float(s8.get("conv_branches", 0.0))
        feats["conv_delay"] = float(s8.get("conv_delay", 0.0))
        feats["helical_step"] = float(s8.get("helical_step", 0.0))

        # --- GROUP E: Stage 9 ---
        feats["fec_family_encoded"] = encode_fec_family(fec_family)
        feats["code_rate"] = float(s9.get("code_rate", 1.0))
        feats["effective_code_rate"] = float(s9.get("effective_code_rate", feats["code_rate"]))
        feats["decoder_success"] = 1.0 if s9.get("decoder_success", True) else 0.0
        feats["fec_quality_score"] = float(s9.get("fec_quality_score", 0.5))
        feats["normalized_path_metric"] = float(s9.get("normalized_path_metric", s9.get("path_metric", 0.1)))
        feats["termination_match"] = 1.0 if s9.get("termination_match", True) else 0.0
        feats["estimated_error_corrections"] = float(s9.get("estimated_error_corrections", 0.0))
        feats["rs_corrected_symbols"] = float(s9.get("rs_corrected_symbols", 0.0))
        feats["rs_uncorrectable_count"] = float(s9.get("rs_uncorrectable_count", 0.0))
        feats["rs_syndrome_score"] = float(s9.get("rs_syndrome_score", 1.0 if feats["decoder_success"] > 0 else 0.0))
        feats["ldpc_initial_syndrome"] = float(s9.get("ldpc_initial_syndrome", 0.0))
        feats["ldpc_final_syndrome"] = float(s9.get("ldpc_final_syndrome", 0.0))
        feats["ldpc_syndrome_reduction"] = float(s9.get("ldpc_syndrome_reduction", 0.0))
        feats["ldpc_iterations"] = float(s9.get("ldpc_iterations", 0.0))
        feats["ldpc_converged"] = 1.0 if s9.get("ldpc_converged", feats["decoder_success"] > 0) else 0.0
        feats["input_output_length_ratio"] = float(s9.get("input_output_length_ratio", 1.0 / max(0.1, feats["code_rate"])))

        # --- GROUP F: Stage 10 Validation ---
        feats["crc_available"] = float(s10.get("crc_available", 0.0))
        feats["crc_pass"] = float(s10.get("crc_pass", 0.0))
        feats["crc_width"] = float(s10.get("crc_width", 0.0))
        feats["crc_pass_rate"] = float(s10.get("crc_pass_rate", 0.0))
        feats["parity_available"] = float(s10.get("parity_available", 0.0))
        feats["parity_pass_rate"] = float(s10.get("parity_pass_rate", 0.5))
        feats["syndrome_available"] = float(s10.get("syndrome_available", 1.0 if fec_family != "none" else 0.0))
        feats["syndrome_valid"] = float(s10.get("syndrome_valid", feats["decoder_success"]))
        feats["normalized_syndrome_score"] = float(s10.get("normalized_syndrome_score", 1.0 if feats["decoder_success"] > 0 else 0.0))
        feats["reencoding_available"] = float(s10.get("reencoding_available", 0.0))
        feats["reencoding_match_fraction"] = float(s10.get("reencoding_match_fraction", 0.0))
        feats["frame_periodicity_score"] = float(s10.get("frame_periodicity_score", 0.0))
        feats["frame_repetition_count"] = float(s10.get("frame_repetition_count", 0.0))
        feats["sync_match_count"] = float(s10.get("sync_match_count", 0.0))
        feats["sync_periodicity_score"] = float(s10.get("sync_periodicity_score", 0.0))
        feats["header_consistency_score"] = float(s10.get("header_consistency_score", 0.0))
        feats["length_consistency_score"] = float(s10.get("length_consistency_score", 0.0))
        feats["binary_entropy"] = float(s10.get("binary_entropy", 1.0))
        feats["bit_balance_zero"] = float(s10.get("bit_balance_zero", 0.5))
        feats["byte_alignment_score"] = float(s10.get("byte_alignment_score", 0.0))
        feats["independent_evidence_group_count"] = float(s10.get("independent_evidence_groups", s10.get("independent_evidence_group_count", 0.0)))
        feats["strong_evidence_group_count"] = float(s10.get("strong_evidence_groups", s10.get("strong_evidence_group_count", 0.0)))
        feats["contradiction_count"] = float(s10.get("contradiction_count", 0.0))
        feats["overall_validation_score"] = float(s10.get("overall_validation_score", 0.0))
        feats["validation_status_encoded"] = encode_validation_status(val_status)

        # --- GROUP G: Cross-Stage Consistency ---
        feats["modulation_vs_constellation_agreement"] = float(1.0 - abs(feats["modulation_probability"] - feats["constellation_support"]))
        feats["modulation_vs_rf_agreement"] = float(1.0 - abs(feats["modulation_probability"] - feats["rf_family_support"]))
        feats["baud_vs_timing_recovery_agreement"] = float(feats["symbol_rate_score"] * feats["timing_lock_score"])
        feats["sync_vs_demod_quality_consistency"] = float(feats["overall_sync_score"] * (1.0 - feats["evm_percent"] / 100.0))
        feats["demod_vs_fec_consistency"] = float(feats["demod_quality_score"] * feats["fec_quality_score"])
        feats["fec_vs_validation_consistency"] = float(feats["fec_quality_score"] * feats["overall_validation_score"])
        feats["crc_vs_fec_consistency"] = float(feats["crc_pass_rate"] * feats["decoder_success"])

        # Sanitize all outputs to float
        clean_feats = {}
        for col in self.columns:
            val = feats.get(col, self.defaults[col])
            if np.isnan(val) or np.isinf(val):
                clean_feats[col] = self.defaults[col]
            else:
                clean_feats[col] = float(val)

        return clean_feats

    def extract_feature_vector(self, candidate_or_dict: Any) -> np.ndarray:
        """Extract ordered 1D numpy array corresponding exactly to PIPELINE_FEATURE_COLUMNS."""
        f_dict = self.extract_features(candidate_or_dict)
        return np.array([f_dict[col] for col in self.columns], dtype=np.float32)

    def extract_feature_matrix(self, candidates: List[Any]) -> np.ndarray:
        """Extract 2D numpy matrix [N_candidates, N_features]."""
        if not candidates:
            return np.empty((0, len(self.columns)), dtype=np.float32)
        return np.array([self.extract_feature_vector(c) for c in candidates], dtype=np.float32)
