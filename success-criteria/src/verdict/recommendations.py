"""Recommendation generation based on verdict."""

from __future__ import annotations

from typing import List

from src.verdict.verdict import FinalVerdict, VerdictCategory


class RecommendationGenerator:
    """Generates actionable recommendations based on the verdict."""

    def generate(self, verdict: FinalVerdict) -> List[str]:
        """Generate recommendations based on the final verdict.

        Returns a list of actionable recommendation strings.
        """
        recommendations: List[str] = []

        if verdict.category == VerdictCategory.SUCCESS:
            recommendations.extend(self._success_recommendations(verdict))
        elif verdict.category == VerdictCategory.PARTIAL:
            recommendations.extend(self._partial_recommendations(verdict))
        else:
            recommendations.extend(self._not_met_recommendations(verdict))

        # Add confidence-based recommendations
        recommendations.extend(self._confidence_recommendations(verdict))

        return recommendations

    def _success_recommendations(
        self, verdict: FinalVerdict
    ) -> List[str]:
        """Recommendations for SUCCESS verdict."""
        recs = [
            "PROCEED: All criteria met. Recommend advancing to next phase.",
            "Document final results and archive evidence package.",
            "Schedule post-decision review to capture lessons learned.",
        ]

        # Check for any criteria with low confidence
        low_conf = [
            r for r in verdict.criteria_results if r.confidence < 0.7
        ]
        if low_conf:
            names = [r.criterion_name for r in low_conf]
            recs.append(
                f"Note: Low confidence on {', '.join(names)}. "
                f"Consider additional validation."
            )

        return recs

    def _partial_recommendations(
        self, verdict: FinalVerdict
    ) -> List[str]:
        """Recommendations for PARTIAL verdict."""
        recs = [
            "CONDITIONAL: Partial success achieved. "
            "Address failed criteria before final GO decision.",
        ]

        for result in verdict.failed_criteria:
            recs.append(
                f"ACTION NEEDED [{result.criterion_name}]: "
                f"Margin = {result.margin:+.2f}. "
                f"{'Close to passing — focused effort may suffice.' if abs(result.margin) < 3 else 'Significant gap — reassess approach.'}"
            )

        recs.append(
            f"Re-evaluate after addressing {len(verdict.failed_criteria)} "
            f"failed criteria."
        )
        return recs

    def _not_met_recommendations(
        self, verdict: FinalVerdict
    ) -> List[str]:
        """Recommendations for NOT_MET verdict."""
        recs = [
            "NO-GO: Success criteria not met. "
            "Significant remediation required.",
        ]

        for result in verdict.failed_criteria:
            recs.append(
                f"REMEDIATE [{result.criterion_name}]: "
                f"Margin = {result.margin:+.2f}, "
                f"confidence = {result.confidence:.2f}."
            )

        recs.extend([
            "Consider revising project timeline and resource allocation.",
            "Schedule root-cause analysis for each failed criterion.",
            "Develop remediation plan with measurable milestones.",
        ])
        return recs

    def _confidence_recommendations(
        self, verdict: FinalVerdict
    ) -> List[str]:
        """Recommendations based on overall confidence level."""
        recs: List[str] = []

        if verdict.overall_confidence < 0.5:
            recs.append(
                "WARNING: Overall confidence is low "
                f"({verdict.overall_confidence:.2f}). "
                "Results may not be reliable."
            )
        elif verdict.overall_confidence < 0.7:
            recs.append(
                "CAUTION: Moderate confidence "
                f"({verdict.overall_confidence:.2f}). "
                "Consider collecting additional evidence."
            )

        return recs
