"""Sensitivity analysis — how robust are the results to threshold changes."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from src.criteria.base import CriterionResult, Evidence, SuccessCriterion


class SensitivityAnalyzer:
    """Analyzes sensitivity of criterion results to threshold variations."""

    def vary_thresholds(
        self,
        criterion: SuccessCriterion,
        evidence: Evidence,
        parameter_name: str,
        values: List[float],
    ) -> List[Dict[str, Any]]:
        """Evaluate a criterion across a range of threshold values.

        This creates modified copies of the criterion with different
        threshold values and evaluates each.

        Args:
            criterion: The criterion to test.
            evidence: The evidence to evaluate against.
            parameter_name: Name of the parameter to vary.
            values: List of values to test.

        Returns:
            List of dicts with value, passed, confidence, margin for each.
        """
        results = []
        for value in values:
            modified = self._modify_criterion(criterion, parameter_name, value)
            if modified is not None:
                result = modified.evaluate(evidence)
                results.append({
                    "parameter": parameter_name,
                    "value": value,
                    "passed": result.passed,
                    "confidence": result.confidence,
                    "margin": result.margin,
                })
            else:
                results.append({
                    "parameter": parameter_name,
                    "value": value,
                    "passed": None,
                    "confidence": None,
                    "margin": None,
                    "error": f"Cannot modify parameter '{parameter_name}'",
                })

        return results

    def identify_fragile_criteria(
        self,
        criteria: List[SuccessCriterion],
        evidence: Evidence,
        margin_threshold: float = 2.0,
    ) -> List[Dict[str, Any]]:
        """Identify criteria that are close to flipping pass/fail.

        A criterion is 'fragile' if its margin is within margin_threshold
        of the pass/fail boundary.
        """
        fragile = []
        for criterion in criteria:
            result = criterion.evaluate(evidence)
            if abs(result.margin) < margin_threshold:
                fragile.append({
                    "criterion": criterion.name,
                    "passed": result.passed,
                    "margin": result.margin,
                    "confidence": result.confidence,
                    "fragility": 1.0 - abs(result.margin) / margin_threshold,
                })
        return sorted(fragile, key=lambda x: x["fragility"], reverse=True)

    def _modify_criterion(
        self,
        criterion: SuccessCriterion,
        parameter_name: str,
        value: float,
    ) -> SuccessCriterion | None:
        """Create a modified copy of a criterion with a different threshold."""
        # Use internal attributes to modify
        attr_name = f"_{parameter_name}"
        if hasattr(criterion, attr_name):
            import copy
            modified = copy.deepcopy(criterion)
            setattr(modified, attr_name, value)
            return modified

        return None

    def full_sensitivity_report(
        self,
        criteria: List[SuccessCriterion],
        evidence: Evidence,
    ) -> Dict[str, Any]:
        """Generate a full sensitivity report across all criteria."""
        fragile = self.identify_fragile_criteria(criteria, evidence)

        report = {
            "fragile_criteria": fragile,
            "n_fragile": len(fragile),
            "n_total": len(criteria),
            "robustness_score": 1.0 - len(fragile) / max(len(criteria), 1),
        }
        return report
