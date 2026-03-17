"""Confidence calculation and bootstrap analysis."""

from __future__ import annotations

import random
from typing import Any, Dict, List

from src.criteria.base import CriterionResult, Evidence, SuccessCriterion


class ConfidenceCalculator:
    """Computes overall confidence and bootstrap estimates."""

    def compute_overall_confidence(
        self, results: List[CriterionResult]
    ) -> float:
        """Compute overall confidence across all criterion results.

        Uses geometric mean of individual confidences to penalize
        any single low-confidence criterion.
        """
        if not results:
            return 0.0

        product = 1.0
        for result in results:
            product *= max(result.confidence, 0.01)  # floor at 0.01

        return product ** (1.0 / len(results))

    def compute_weighted_confidence(
        self,
        results: List[CriterionResult],
        weights: Dict[str, float] | None = None,
    ) -> float:
        """Compute weighted confidence using criterion names as keys."""
        if not results:
            return 0.0

        if weights is None:
            # Equal weights
            return self.compute_overall_confidence(results)

        total_weight = 0.0
        weighted_sum = 0.0
        for result in results:
            w = weights.get(result.criterion_name, 1.0)
            weighted_sum += result.confidence * w
            total_weight += w

        if total_weight == 0:
            return 0.0
        return weighted_sum / total_weight

    def bootstrap_criterion(
        self,
        criterion: SuccessCriterion,
        evidence: Evidence,
        n_bootstrap: int = 100,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """Bootstrap a criterion's result by resampling phase data.

        Returns statistics about the stability of the criterion's result.
        """
        rng = random.Random(seed)
        base_result = criterion.evaluate(evidence)

        # Get the improvement curve for resampling
        curve = evidence.get_improvement_curve()
        if len(curve) < 3:
            return {
                "base_passed": base_result.passed,
                "base_confidence": base_result.confidence,
                "pass_rate": 1.0 if base_result.passed else 0.0,
                "n_bootstrap": 0,
                "note": "Insufficient data for bootstrap",
            }

        pass_count = 0
        confidences: List[float] = []

        for _ in range(n_bootstrap):
            # Create perturbed evidence by adding small noise to scores
            perturbed = self._perturb_evidence(evidence, rng)
            result = criterion.evaluate(perturbed)
            if result.passed:
                pass_count += 1
            confidences.append(result.confidence)

        pass_rate = pass_count / n_bootstrap
        mean_conf = sum(confidences) / len(confidences)
        sorted_conf = sorted(confidences)
        ci_lower = sorted_conf[int(0.025 * n_bootstrap)]
        ci_upper = sorted_conf[int(0.975 * n_bootstrap)]

        return {
            "base_passed": base_result.passed,
            "base_confidence": base_result.confidence,
            "pass_rate": pass_rate,
            "mean_confidence": mean_conf,
            "ci_95_lower": ci_lower,
            "ci_95_upper": ci_upper,
            "n_bootstrap": n_bootstrap,
        }

    @staticmethod
    def _perturb_evidence(
        evidence: Evidence, rng: random.Random
    ) -> Evidence:
        """Create a slightly perturbed copy of the evidence."""
        from src.criteria.base import Evidence as Ev

        def perturb_phase(phase_data: Dict[str, Any]) -> Dict[str, Any]:
            result = dict(phase_data)
            if "score" in result:
                result["score"] = result["score"] + rng.gauss(0, 1.0)
            if "collapse_score" in result:
                result["collapse_score"] = (
                    result["collapse_score"] + rng.gauss(0, 0.5)
                )
            if "ablations" in result:
                perturbed_abl = {}
                for paradigm, data in result["ablations"].items():
                    perturbed_abl[paradigm] = {
                        "with": data["with"] + rng.gauss(0, 0.5),
                        "without": data["without"] + rng.gauss(0, 0.5),
                    }
                result["ablations"] = perturbed_abl
            return result

        return Ev(
            phase_0=perturb_phase(evidence.phase_0),
            phase_1=perturb_phase(evidence.phase_1),
            phase_2=perturb_phase(evidence.phase_2),
            phase_3=perturb_phase(evidence.phase_3),
            phase_4=perturb_phase(evidence.phase_4),
            safety=evidence.safety,
            publications=evidence.publications,
            audit_trail=evidence.audit_trail,
        )
