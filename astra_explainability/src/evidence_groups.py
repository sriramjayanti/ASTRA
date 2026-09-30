"""
evidence_groups.py
Evidence clustering by independence groups to prevent double-counting of correlated telemetry.
"""

from typing import Dict, List, Optional, Set
from collections import defaultdict
import numpy as np

from .models import (
    EvidenceItem,
    EvidenceGroup,
    IndependenceGroup,
    EvidenceStrength,
    EvidenceDirection
)


class EvidenceGrouper:
    """Class wrapper for evidence clustering and independence grouping."""

    def group_evidence(self, evidence_items: List[EvidenceItem]) -> Dict[IndependenceGroup, List[EvidenceItem]]:
        """Groups items by independence group and returns mapping to item lists."""
        grouped: Dict[IndependenceGroup, List[EvidenceItem]] = defaultdict(list)
        for it in evidence_items:
            grouped[it.independence_group].append(it)
        return dict(grouped)

    def count_strong_independent_groups(
        self,
        evidence_items: List[EvidenceItem],
        relevant_groups: Optional[Set[IndependenceGroup]] = None
    ) -> int:
        """Counts how many distinct independence groups have strong supporting evidence."""
        groups = group_evidence_by_independence(evidence_items)
        return count_strong_independent_groups(groups, relevant_groups)


def group_evidence_by_independence(
    evidence_items: List[EvidenceItem]
) -> Dict[IndependenceGroup, EvidenceGroup]:
    """
    Cluster evidence items into their assigned IndependenceGroup to avoid over-weighting correlated metrics.
    """
    grouped_items: Dict[IndependenceGroup, List[EvidenceItem]] = defaultdict(list)
    for it in evidence_items:
        grouped_items[it.independence_group].append(it)

    result: Dict[IndependenceGroup, EvidenceGroup] = {}
    for grp_id, items in grouped_items.items():
        has_strong = any(
            it.strength == EvidenceStrength.STRONG and it.direction == EvidenceDirection.SUPPORT
            for it in items
        )
        scores = [it.normalized_score for it in items]
        mean_sc = float(np.mean(scores)) if scores else 0.0

        result[grp_id] = EvidenceGroup(
            group_id=grp_id,
            items=items,
            has_strong_evidence=has_strong,
            mean_score=mean_sc
        )

    return result


def count_strong_independent_groups(
    groups: Dict[IndependenceGroup, EvidenceGroup],
    relevant_groups: Optional[Set[IndependenceGroup]] = None
) -> int:
    """
    Count the number of distinct independence groups that supply strong supporting evidence.
    """
    count = 0
    for grp_id, grp in groups.items():
        if grp_id == IndependenceGroup.USER_OVERRIDE:
            # User overrides are not independent measured physical evidence
            continue
        if relevant_groups is not None and grp_id not in relevant_groups:
            continue
        if grp.has_strong_evidence:
            count += 1
    return count


def get_dominant_evidence_for_group(
    group: EvidenceGroup
) -> Optional[EvidenceItem]:
    """Return the highest-scoring evidence item in an independence group."""
    if not group.items:
        return None
    sorted_items = sorted(group.items, key=lambda x: (x.strength == EvidenceStrength.STRONG, x.normalized_score), reverse=True)
    return sorted_items[0]
