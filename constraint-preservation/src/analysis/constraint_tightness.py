"""ConstraintTightnessAnalyzer: determine if constraints are too tight or too loose."""

from __future__ import annotations

from typing import Any, Dict, List


class ConstraintTightnessAnalyzer:
    """Analyze whether constraints are calibrated correctly.

    - Too tight: rejection rate > 50%
    - Too loose: rejection rate = 0% across a meaningful sample
    """

    def __init__(self, audit_entries: List[Dict[str, Any]]) -> None:
        self._entries = audit_entries

    def analyze(self) -> Dict[str, Dict[str, Any]]:
        """Analyze tightness for each constraint.

        Returns a dict keyed by constraint name with:
        - violation_count: int
        - check_count: int
        - violation_rate: float
        - assessment: "too_tight" | "too_loose" | "well_calibrated"
        """
        constraint_stats: Dict[str, Dict[str, int]] = {}

        for entry in self._entries:
            results_summary = entry.get("results_summary", {})
            for cname, result in results_summary.items():
                if cname not in constraint_stats:
                    constraint_stats[cname] = {"checks": 0, "violations": 0}
                constraint_stats[cname]["checks"] += 1
                if not result.get("satisfied", True):
                    constraint_stats[cname]["violations"] += 1

        analysis: Dict[str, Dict[str, Any]] = {}
        for cname, stats in constraint_stats.items():
            checks = stats["checks"]
            violations = stats["violations"]
            rate = violations / checks if checks else 0.0

            if rate > 0.50:
                assessment = "too_tight"
            elif rate == 0.0 and checks >= 5:
                assessment = "too_loose"
            else:
                assessment = "well_calibrated"

            analysis[cname] = {
                "violation_count": violations,
                "check_count": checks,
                "violation_rate": rate,
                "assessment": assessment,
            }

        return analysis

    def suggest_adjustments(self) -> List[Dict[str, Any]]:
        """Suggest threshold adjustments for poorly calibrated constraints."""
        analysis = self.analyze()
        suggestions: List[Dict[str, Any]] = []

        for cname, info in analysis.items():
            if info["assessment"] == "too_tight":
                suggestions.append(
                    {
                        "constraint": cname,
                        "issue": "too_tight",
                        "violation_rate": info["violation_rate"],
                        "suggestion": "Consider relaxing the threshold.",
                    }
                )
            elif info["assessment"] == "too_loose":
                suggestions.append(
                    {
                        "constraint": cname,
                        "issue": "too_loose",
                        "violation_rate": info["violation_rate"],
                        "suggestion": "Consider tightening the threshold.",
                    }
                )

        return suggestions
