"""
statuses.py
Canonical status resolution (CONFIRMED, ESTIMATED, POSSIBLE, UNKNOWN) with evidence gating.
"""

from typing import Tuple, Optional, Dict, Any, List
from .models import AstraStatus, Contradiction, ContradictionSeverity


class StatusEngine:
    """Class wrapper for status determinations using configurable thresholds."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        if config and "statuses" in config:
            self.config = config
        else:
            self.config = {"statuses": config or {}}

    def determine_status(
        self,
        confidence_score: float,
        strong_groups: int = 0,
        contradictions_count: int = 0,
        unresolved_contradictions: Optional[List[Contradiction]] = None,
        field_name: str = "general",
        downstream_validation_passed: Optional[bool] = None
    ) -> Tuple[AstraStatus, str]:
        contras = unresolved_contradictions or []
        return determine_status(
            confidence_score=confidence_score,
            strong_independent_groups_count=strong_groups,
            unresolved_contradictions=contras,
            field_name=field_name,
            config=self.config,
            downstream_validation_passed=downstream_validation_passed
        )


def determine_status(
    confidence_score: float,
    strong_independent_groups_count: int,
    unresolved_contradictions: Optional[List[Contradiction]] = None,
    field_name: str = "general",
    config: Optional[Dict[str, Any]] = None,
    downstream_validation_passed: Optional[bool] = None
) -> Tuple[AstraStatus, str]:
    """
    Determine canonical user-facing status based on confidence score, independent support, and contradictions.

    Rules:
      1. CONFIRMED requires:
         - score >= confirmed_threshold (default 0.85)
         - strong independent groups >= confirmed_min_strong_groups (default 2)
         - 0 CRITICAL or HIGH unresolved contradictions
      2. ESTIMATED requires:
         - score >= estimated_threshold (default 0.60)
         - strong independent groups >= 1
         - 0 CRITICAL unresolved contradictions
      3. POSSIBLE:
         - score >= possible_threshold (default 0.35)
      4. UNKNOWN:
         - score < 0.35 OR unresolved CRITICAL contradiction OR zero evidence
    """
    if isinstance(config, StatusEngine):
        cfg = config.config.get("statuses", {})
    else:
        cfg = (config or {}).get("statuses", config or {})
    t_conf = float(cfg.get("confirmed_threshold", 0.85))
    t_est = float(cfg.get("estimated_threshold", 0.60))
    t_pos = float(cfg.get("possible_threshold", 0.35))
    min_strong = int(cfg.get("confirmed_min_strong_groups", 2))

    contradictions = unresolved_contradictions or []
    has_critical = any(c.severity == ContradictionSeverity.CRITICAL for c in contradictions if c.resolution_status != "RESOLVED_BY_DOWNSTREAM")
    has_high = any(c.severity == ContradictionSeverity.HIGH for c in contradictions if c.resolution_status != "RESOLVED_BY_DOWNSTREAM")

    # Field-specific gating
    # e.g. Payload cannot be CONFIRMED if downstream validation failed
    if field_name.lower() in ["payload", "payload_bytes"] and downstream_validation_passed is False:
        return AstraStatus.POSSIBLE, "Payload recovery cannot be CONFIRMED because downstream validation / CRC failed."

    if has_critical:
        return AstraStatus.UNKNOWN, "Contradictory evidence of critical severity prevents reliable determination."

    # Check CONFIRMED
    if confidence_score >= t_conf:
        if strong_independent_groups_count < min_strong:
            return AstraStatus.ESTIMATED, f"Score meets CONFIRMED threshold ({confidence_score:.2f}), but insufficient independent strong evidence groups ({strong_independent_groups_count} < {min_strong})."
        if has_high:
            return AstraStatus.ESTIMATED, f"High score ({confidence_score:.2f}) reduced to ESTIMATED due to unresolved high severity contradiction."
        reason = f"Corroborated by {strong_independent_groups_count} independent evidence groups with high confidence ({confidence_score:.2f}) and no unresolved contradictions."
        return AstraStatus.CONFIRMED, reason

    # Check ESTIMATED
    if confidence_score >= t_est and strong_independent_groups_count >= 1 and not has_high:
        reason = f"Supported by consistent evidence ({confidence_score:.2f}) across {strong_independent_groups_count} independent group(s), but final deterministic validation remains incomplete."
        return AstraStatus.ESTIMATED, reason

    # Check POSSIBLE
    if confidence_score >= t_pos:
        reason = f"Plausible candidate hypothesis ({confidence_score:.2f}), but significant uncertainty or missing validation limits confidence."
        return AstraStatus.POSSIBLE, reason

    # UNKNOWN
    reason = f"Insufficient evidence or high uncertainty ({confidence_score:.2f}) to validate hypothesis."
    return AstraStatus.UNKNOWN, reason
