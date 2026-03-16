"""Paradigm contribution analyzer: measures contribution of each paradigm."""
from __future__ import annotations

from typing import Any, Dict, List, Optional


class ParadigmContributionAnalyzer:
    """Analyzes how each paradigm (SOAR, SymCode, CTM, Godel, RLM) contributes."""

    def __init__(self):
        self._modification_log: List[Dict[str, Any]] = []
        self._verification_log: List[Dict[str, Any]] = []

    def set_modification_log(self, log: List[Dict[str, Any]]) -> None:
        """Set the modification log for analysis."""
        self._modification_log = list(log)

    def set_verification_log(self, log: List[Dict[str, Any]]) -> None:
        """Set the verification log for analysis."""
        self._verification_log = list(log)

    def soar_efficiency(self) -> Dict[str, float]:
        """Compute SOAR population efficiency metrics."""
        if not self._modification_log:
            return {"total": 0, "successful": 0, "efficiency": 0.0}

        total = len(self._modification_log)
        successful = sum(
            1 for m in self._modification_log
            if m.get("result", "") == "applied"
        )
        return {
            "total": total,
            "successful": successful,
            "efficiency": successful / total if total > 0 else 0.0,
        }

    def verification_breakdown(self) -> Dict[str, Any]:
        """Break down verification results by gate."""
        if not self._verification_log:
            return {
                "empirical_pass": 0, "empirical_fail": 0,
                "compactness_pass": 0, "compactness_fail": 0,
                "total": 0,
            }

        emp_pass = sum(1 for v in self._verification_log if v.get("empirical_passed", False))
        emp_fail = sum(1 for v in self._verification_log if not v.get("empirical_passed", True))
        comp_pass = sum(1 for v in self._verification_log if v.get("compactness_passed", False))
        comp_fail = sum(1 for v in self._verification_log if not v.get("compactness_passed", True))

        return {
            "empirical_pass": emp_pass,
            "empirical_fail": emp_fail,
            "compactness_pass": comp_pass,
            "compactness_fail": comp_fail,
            "total": len(self._verification_log),
        }

    def modification_success_rate(self) -> float:
        """Compute overall modification success rate."""
        if not self._modification_log:
            return 0.0
        successful = sum(
            1 for m in self._modification_log
            if m.get("result", "") == "applied"
        )
        return successful / len(self._modification_log)

    def plot_contribution_breakdown(self) -> Dict[str, Any]:
        """Generate data for a contribution breakdown chart."""
        return {
            "soar": self.soar_efficiency(),
            "verification": self.verification_breakdown(),
            "success_rate": self.modification_success_rate(),
            "paradigms": {
                "soar": {"role": "population evolution", "active": True},
                "symcode_ctm": {"role": "symbolic verification", "active": True},
                "godel": {"role": "self-referential modification", "active": True},
                "rlm": {"role": "long-context reasoning", "active": True},
            },
        }
