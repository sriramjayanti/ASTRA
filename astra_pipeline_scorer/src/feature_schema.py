"""
feature_schema.py
Strict, versioned feature schema definition for ASTRA Stage 11 — Pipeline Scoring Model.
Defines column ordering, data types, default fallbacks, and feature metadata across Stages 5-10.
"""

from typing import Dict, List, Any

PIPELINE_FEATURE_SCHEMA_VERSION = "pipeline_features_v1"

# Ordered list of all features expected by the Stage 11 XGBoost model
PIPELINE_FEATURE_COLUMNS: List[str] = [
    # --- GROUP A: Stage 5 Candidate Priors ---
    "mod_encoded",
    "modulation_probability",
    "symbol_rate_score",
    "candidate_initial_score",
    "rf_family_support",
    "constellation_support",
    "modulation_rank",
    "baud_rank",
    "modulation_entropy",
    "modulation_margin",
    "symbol_rate_margin",
    "samples_per_symbol",
    "stage5_fallback_used",

    # --- GROUP B: Stage 6 Synchronization Evidence ---
    "timing_lock_score",
    "carrier_lock_score",
    "frequency_lock_score",
    "overall_sync_score",
    "estimated_cfo_hz",
    "normalized_cfo",
    "residual_cfo_hz",
    "normalized_residual_cfo",
    "phase_error_rms",
    "timing_error_variance",
    "estimated_sps",
    "sps_error_relative",
    "constellation_quality_before",
    "constellation_quality_after",
    "constellation_improvement",

    # --- GROUP C: Stage 7 Demodulation Evidence ---
    "evm_percent",
    "evm_db",
    "decision_distance_mean",
    "decision_distance_std",
    "decision_margin_mean",
    "mean_abs_llr",
    "median_abs_llr",
    "low_confidence_bit_fraction",
    "estimated_snr_db",
    "noise_variance",
    "demod_quality_score",
    "ambiguity_variant_index",
    "ambiguity_variant_count",

    # --- GROUP D: Stage 8 Interleaver Evidence ---
    "interleaver_family_encoded",
    "interleaver_candidate_rank",
    "interleaver_structural_score",
    "interleaver_periodicity_score",
    "interleaver_autocorrelation_score",
    "interleaver_entropy",
    "interleaver_bit_balance",
    "interleaver_run_length_score",
    "interleaver_remainder_fraction",
    "interleaver_family_prior",
    "block_rows",
    "block_cols",
    "conv_branches",
    "conv_delay",
    "helical_step",

    # --- GROUP E: Stage 9 FEC Evidence ---
    "fec_family_encoded",
    "code_rate",
    "effective_code_rate",
    "decoder_success",
    "fec_quality_score",
    "normalized_path_metric",
    "termination_match",
    "estimated_error_corrections",
    "rs_corrected_symbols",
    "rs_uncorrectable_count",
    "rs_syndrome_score",
    "ldpc_initial_syndrome",
    "ldpc_final_syndrome",
    "ldpc_syndrome_reduction",
    "ldpc_iterations",
    "ldpc_converged",
    "input_output_length_ratio",

    # --- GROUP F: Stage 10 Validation Evidence ---
    "crc_available",
    "crc_pass",
    "crc_width",
    "crc_pass_rate",
    "parity_available",
    "parity_pass_rate",
    "syndrome_available",
    "syndrome_valid",
    "normalized_syndrome_score",
    "reencoding_available",
    "reencoding_match_fraction",
    "frame_periodicity_score",
    "frame_repetition_count",
    "sync_match_count",
    "sync_periodicity_score",
    "header_consistency_score",
    "length_consistency_score",
    "binary_entropy",
    "bit_balance_zero",
    "byte_alignment_score",
    "independent_evidence_group_count",
    "strong_evidence_group_count",
    "contradiction_count",
    "overall_validation_score",
    "validation_status_encoded",

    # --- GROUP G: Cross-Stage Consistency Evidence ---
    "modulation_vs_constellation_agreement",
    "modulation_vs_rf_agreement",
    "baud_vs_timing_recovery_agreement",
    "sync_vs_demod_quality_consistency",
    "demod_vs_fec_consistency",
    "fec_vs_validation_consistency",
    "crc_vs_fec_consistency",
]

# Default values for missing / unpopulated metrics
FEATURE_DEFAULTS: Dict[str, float] = {col: 0.0 for col in PIPELINE_FEATURE_COLUMNS}
FEATURE_DEFAULTS.update({
    "modulation_probability": 0.5,
    "symbol_rate_score": 0.5,
    "samples_per_symbol": 4.0,
    "estimated_sps": 4.0,
    "evm_percent": 25.0,
    "evm_db": -12.0,
    "mean_abs_llr": 1.0,
    "median_abs_llr": 1.0,
    "code_rate": 1.0,
    "effective_code_rate": 1.0,
    "binary_entropy": 1.0,
    "bit_balance_zero": 0.5,
    "interleaver_entropy": 1.0,
    "interleaver_bit_balance": 0.5,
    "validation_status_encoded": 1.0,
})


def get_feature_schema() -> Dict[str, Any]:
    """Returns the full schema specification."""
    return {
        "schema_version": PIPELINE_FEATURE_SCHEMA_VERSION,
        "feature_count": len(PIPELINE_FEATURE_COLUMNS),
        "columns": list(PIPELINE_FEATURE_COLUMNS),
        "defaults": dict(FEATURE_DEFAULTS),
    }
