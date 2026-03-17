"""Criterion 2: Paradigm-specific improvement via paired t-tests.

Sub-tests (ALL 4 must pass):
1. SymCode >= 5pp improvement
2. Godel >= 2pp improvement
3. SOAR >= 5pp improvement
4. RLM >= 10pp improvement
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np

from src.criteria.base import CriterionResult, Evidence, SuccessCriterion

# Required minimum effects per paradigm (percentage points)
DEFAULT_MIN_EFFECTS = {
    "symcode": 5.0,
    "godel": 2.0,
    "soar": 5.0,
    "rlm": 10.0,
}


def _paired_t_test(
    with_scores: List[float], without_scores: List[float]
) -> Tuple[float, float, float]:
    """Paired t-test using numpy.

    Returns (t_statistic, p_value, mean_difference).
    """
    with_arr = np.array(with_scores, dtype=np.float64)
    without_arr = np.array(without_scores, dtype=np.float64)
    differences = with_arr - without_arr
    n = len(differences)

    if n < 2:
        return 0.0, 1.0, 0.0

    mean_diff = float(np.mean(differences))
    std_diff = float(np.std(differences, ddof=1))

    if std_diff == 0:
        # Perfect agreement - if mean_diff > 0, very significant
        if mean_diff > 0:
            return float("inf"), 0.0, mean_diff
        elif mean_diff < 0:
            return float("-inf"), 0.0, mean_diff
        else:
            return 0.0, 1.0, 0.0

    t_stat = mean_diff / (std_diff / math.sqrt(n))

    # Two-sided p-value using t-distribution approximation
    # Use the normal approximation for df >= 3
    df = n - 1
    p_value = _t_distribution_p(abs(t_stat), df) * 2
    p_value = min(p_value, 1.0)

    return float(t_stat), p_value, mean_diff


def _t_distribution_p(t: float, df: int) -> float:
    """Approximate one-tailed p-value for t-distribution.

    Uses the approximation: p ~ normal CDF for larger df,
    with a correction factor for small df.
    """
    if df <= 0:
        return 0.5
    # For small df, use a simple approximation
    # p = P(T > t) for t-distribution with df degrees of freedom
    # Approximation using normal distribution with correction
    z = t * (1.0 - 1.0 / (4.0 * df)) / math.sqrt(1.0 + t * t / (2.0 * df))
    return 1.0 - _normal_cdf(z)


def _normal_cdf(x: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


class ParadigmImprovementCriterion(SuccessCriterion):
    """Criterion 2: Paradigm-specific improvement."""

    def __init__(
        self,
        alpha: float = 0.05,
        min_effects: Dict[str, float] | None = None,
    ):
        self._alpha = alpha
        self._min_effects = min_effects or dict(DEFAULT_MIN_EFFECTS)

    @property
    def name(self) -> str:
        return "Paradigm Improvement"

    @property
    def description(self) -> str:
        return (
            "Each paradigm (SymCode, Godel, SOAR, RLM) must show "
            "statistically significant improvement in paired comparisons."
        )

    @property
    def threshold_description(self) -> str:
        effects = ", ".join(
            f"{k}>={v}pp" for k, v in self._min_effects.items()
        )
        return f"Paired t-test p<{self._alpha} for all: {effects}"

    @property
    def required_evidence(self) -> list:
        return ["phase_0", "phase_1", "phase_2", "phase_3", "phase_4"]

    def evaluate(self, evidence: Evidence) -> CriterionResult:
        ablation_results = evidence.get_ablation_results()
        sub_results = {}
        all_passed = True
        margins = []

        for paradigm, min_effect in self._min_effects.items():
            if paradigm not in ablation_results:
                sub_results[paradigm] = {
                    "passed": False,
                    "reason": "no data",
                    "t_stat": 0.0,
                    "p_value": 1.0,
                    "mean_diff": 0.0,
                    "min_effect": min_effect,
                }
                all_passed = False
                margins.append(-min_effect)
                continue

            data = ablation_results[paradigm]
            t_stat, p_value, mean_diff = _paired_t_test(
                data["with"], data["without"]
            )

            # Must be significant AND meet minimum effect size
            paradigm_passed = (
                p_value < self._alpha and mean_diff >= min_effect
            )
            sub_results[paradigm] = {
                "passed": paradigm_passed,
                "t_stat": t_stat,
                "p_value": p_value,
                "mean_diff": mean_diff,
                "min_effect": min_effect,
                "margin": mean_diff - min_effect,
            }
            margins.append(mean_diff - min_effect)

            if not paradigm_passed:
                all_passed = False

        # Overall confidence
        n_passed = sum(1 for r in sub_results.values() if r.get("passed"))
        confidence = n_passed / len(self._min_effects)
        if all_passed:
            # Adjust confidence by how strong the results are
            avg_margin = sum(margins) / len(margins) if margins else 0
            if avg_margin > 5:
                confidence = min(1.0, confidence + 0.1)

        return CriterionResult(
            passed=all_passed,
            confidence=confidence,
            measured_value={
                p: {"mean_diff": r["mean_diff"], "p_value": r["p_value"]}
                for p, r in sub_results.items()
            },
            threshold=self._min_effects,
            margin=min(margins) if margins else 0.0,
            supporting_evidence=[
                f"{p}: mean_diff={r['mean_diff']:.2f}pp, p={r['p_value']:.4f}"
                for p, r in sub_results.items()
            ],
            methodology=(
                "Paired t-test (with vs without paradigm) across phases"
            ),
            caveats=[
                "Small sample size (5 paired observations per paradigm)",
                "Normal approximation used for p-values",
            ],
            details={"sub_results": sub_results, "ablation_data": ablation_results},
            criterion_name=self.name,
        )
