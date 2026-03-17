"""Criterion 4: Publication acceptance.

Requirements:
- >= 2 accepted papers
- >= 1 at a tier-1 or tier-2 venue
- Confidence adjusted for under-review papers
"""

from __future__ import annotations

from typing import Any, Dict, List

from src.criteria.base import CriterionResult, Evidence, SuccessCriterion

DEFAULT_TIER_1 = [
    "NeurIPS", "ICML", "ICLR", "AAAI", "Nature",
    "Science", "Nature Machine Intelligence",
]

DEFAULT_TIER_2 = [
    "AISTATS", "UAI", "COLT", "JMLR", "TMLR",
    "ACL", "EMNLP", "CVPR",
]


class PublicationAcceptanceCriterion(SuccessCriterion):
    """Criterion 4: Sufficient peer-reviewed publications."""

    def __init__(
        self,
        min_accepted: int = 2,
        min_tier_1_or_2: int = 1,
        under_review_discount: float = 0.15,
        tier_1_venues: List[str] | None = None,
        tier_2_venues: List[str] | None = None,
    ):
        self._min_accepted = min_accepted
        self._min_tier_1_or_2 = min_tier_1_or_2
        self._under_review_discount = under_review_discount
        self._tier_1 = tier_1_venues or list(DEFAULT_TIER_1)
        self._tier_2 = tier_2_venues or list(DEFAULT_TIER_2)

    @property
    def name(self) -> str:
        return "Publication Acceptance"

    @property
    def description(self) -> str:
        return (
            "Sufficient peer-reviewed publications at recognized venues "
            "to validate the research contributions."
        )

    @property
    def threshold_description(self) -> str:
        return (
            f">= {self._min_accepted} accepted papers, "
            f">= {self._min_tier_1_or_2} at tier-1/2 venue"
        )

    @property
    def required_evidence(self) -> list:
        return ["publications"]

    def _classify_venue(self, venue: str) -> str:
        """Classify a venue into tier-1, tier-2, or other."""
        if venue in self._tier_1:
            return "tier_1"
        elif venue in self._tier_2:
            return "tier_2"
        else:
            return "other"

    def evaluate(self, evidence: Evidence) -> CriterionResult:
        publications = evidence.publications

        accepted = []
        under_review = []
        tier_1_2_accepted = []

        for pub in publications:
            status = pub.get("status", "unknown")
            venue = pub.get("venue", "")
            tier = self._classify_venue(venue)

            if status == "accepted":
                accepted.append(pub)
                if tier in ("tier_1", "tier_2"):
                    tier_1_2_accepted.append(pub)
            elif status == "under_review":
                under_review.append(pub)

        n_accepted = len(accepted)
        n_tier_1_2 = len(tier_1_2_accepted)
        n_under_review = len(under_review)

        # Check thresholds
        count_passed = n_accepted >= self._min_accepted
        tier_passed = n_tier_1_2 >= self._min_tier_1_or_2
        overall_passed = count_passed and tier_passed

        # Confidence: reduce if relying on under-review papers
        confidence = 1.0
        if not count_passed and n_accepted + n_under_review >= self._min_accepted:
            confidence -= self._under_review_discount * n_under_review
        if not overall_passed:
            confidence = max(0.3, confidence - 0.2)
        if n_under_review > 0:
            confidence -= self._under_review_discount * min(n_under_review, 2)
        confidence = max(0.0, min(1.0, confidence))

        margin = float(n_accepted - self._min_accepted)

        return CriterionResult(
            passed=overall_passed,
            confidence=confidence,
            measured_value={
                "accepted_count": n_accepted,
                "tier_1_2_count": n_tier_1_2,
                "under_review_count": n_under_review,
            },
            threshold={
                "min_accepted": self._min_accepted,
                "min_tier_1_or_2": self._min_tier_1_or_2,
            },
            margin=margin,
            supporting_evidence=[
                f"Accepted: {[p.get('title', 'untitled') for p in accepted]}",
                f"Under review: {[p.get('title', 'untitled') for p in under_review]}",
                f"Tier-1/2 accepted: {[p.get('venue', '?') for p in tier_1_2_accepted]}",
            ],
            methodology="Count of accepted papers with venue tier classification",
            caveats=[
                f"{n_under_review} paper(s) still under review"
                if n_under_review > 0
                else "All papers have final decisions"
            ],
            details={
                "accepted": accepted,
                "under_review": under_review,
                "tier_1_2": tier_1_2_accepted,
                "count_passed": count_passed,
                "tier_passed": tier_passed,
            },
            criterion_name=self.name,
        )
