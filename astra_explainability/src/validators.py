"""
ASTRA Stage 15: Input Validators.

Validates that incoming analysis record has proper structure without crashing.
"""

from typing import Dict, Any, List


class InputValidator:
    """Validates analysis record structure from Stages 1-14."""

    @staticmethod
    def validate_record(record: Dict[str, Any]) -> List[str]:
        """
        Validates the analysis record dictionary.
        Returns a list of warnings or non-fatal anomalies.
        """
        warnings = []
        if not isinstance(record, dict):
            raise ValueError(f"Analysis record must be a dictionary, got {type(record)}")

        if "signal_id" not in record:
            warnings.append("signal_id missing from analysis record; defaulting to 'unknown_signal'.")

        # Check for presence of essential stage data
        stages_to_check = [
            ("stage_3_fusion", "Stage 3 (Modulation Fusion)"),
            ("stage_5_candidates", "Stage 5 (Candidates)"),
            ("stage_6_sync", "Stage 6 (Synchronization)"),
            ("stage_7_demod", "Stage 7 (Demodulation)"),
            ("stage_9_fec", "Stage 9 (FEC)"),
            ("stage_10_validation", "Stage 10 (Validation)"),
            ("stage_11_ranking", "Stage 11 (Pipeline Ranking)"),
            ("stage_14_payload", "Stage 14 (Payload Explorer)")
        ]

        missing_stages = [desc for key, desc in stages_to_check if key not in record or not record[key]]
        if missing_stages:
            warnings.append(f"Partial record: Missing stages {', '.join(missing_stages)}")

        return warnings
