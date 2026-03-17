"""Partial success analysis — what's close, what needs work."""

from __future__ import annotations

from typing import Any, Dict, List

from src.criteria.base import CriterionResult
from src.verdict.verdict import FinalVerdict, VerdictCategory


class PartialSuccessAnalyzer:
    """Analyzes partial success verdicts to identify gaps and opportunities."""

    def analyze(self, verdict: FinalVerdict) -> Dict[str, Any]:
        """Analyze a verdict and provide detailed breakdown.

        Returns analysis dict with gap assessment and estimates.
        """
        analysis: Dict[str, Any] = {
            "category": verdict.category.value,
            "n_passed": verdict.n_passed,
            "n_total": verdict.n_total,
            "gap": verdict.n_total - verdict.n_passed,
        }

        # Analyze failed criteria
        failed = verdict.failed_criteria
        analysis["failed_criteria"] = [
            {
                "name": r.criterion_name,
                "margin": r.margin,
                "confidence": r.confidence,
                "details": r.details,
            }
            for r in failed
        ]

        # Estimate remaining work
        analysis["estimated_work"] = self.estimate_remaining_work(failed)

        # Find closest to passing
        analysis["closest_to_passing"] = self.closest_to_passing(failed)

        return analysis

    def estimate_remaining_work(
        self, failed_results: List[CriterionResult]
    ) -> List[Dict[str, Any]]:
        """Estimate work needed to pass each failed criterion."""
        estimates = []
        for result in failed_results:
            deficit = abs(result.margin)
            difficulty = self._estimate_difficulty(result)
            estimates.append({
                "criterion": result.criterion_name,
                "deficit": deficit,
                "difficulty": difficulty,
                "estimated_phases_needed": max(1, int(deficit / 3.0) + 1),
                "recommendation": self._work_recommendation(result),
            })
        return estimates

    def closest_to_passing(
        self, failed_results: List[CriterionResult]
    ) -> Dict[str, Any] | None:
        """Find the failed criterion closest to the pass threshold."""
        if not failed_results:
            return None

        # Sort by margin (closest to 0 means closest to passing)
        sorted_by_margin = sorted(
            failed_results, key=lambda r: abs(r.margin)
        )
        closest = sorted_by_margin[0]

        return {
            "criterion": closest.criterion_name,
            "margin": closest.margin,
            "confidence": closest.confidence,
            "shortfall": abs(closest.margin),
        }

    def _estimate_difficulty(self, result: CriterionResult) -> str:
        """Estimate difficulty level based on margin and confidence."""
        margin = abs(result.margin)
        if margin < 2.0:
            return "low"
        elif margin < 5.0:
            return "medium"
        else:
            return "high"

    def _work_recommendation(self, result: CriterionResult) -> str:
        """Generate a work recommendation for a failed criterion."""
        name = result.criterion_name
        margin = result.margin

        if "Sustained" in name:
            return (
                f"Improvement curve needs {abs(margin):.1f}pp more gain. "
                f"Focus on phase-over-phase consistency."
            )
        elif "Paradigm" in name:
            return (
                f"Paradigm ablation deficit of {abs(margin):.1f}pp. "
                f"Review underperforming paradigm components."
            )
        elif "GDI" in name:
            return (
                f"GDI margin of {margin:.3f}. "
                f"Tighten safety monitoring or reduce operational bounds."
            )
        elif "Publication" in name:
            return (
                f"Need {abs(margin):.0f} more accepted publication(s). "
                f"Submit to tier-1/2 venues."
            )
        elif "Audit" in name:
            return (
                f"Audit trail incomplete. "
                f"Fill missing logs and add reasoning traces."
            )
        else:
            return f"Address deficit of {abs(margin):.2f}."
