"""Criterion 1: Sustained Improvement over phases.

Sub-tests (ALL must pass):
1. Mann-Kendall trend test: non-decreasing trend with p < 0.05
2. Total gain >= 5 percentage points
3. Collapse divergence >= 10 percentage points
"""

from __future__ import annotations

import math
from typing import List, Tuple

from src.criteria.base import CriterionResult, Evidence, SuccessCriterion


def _mann_kendall(data: List[float]) -> Tuple[float, float, str]:
    """Mann-Kendall trend test via concordant/discordant pair counting.

    Returns (S statistic, two-sided p-value, trend direction).
    """
    n = len(data)
    if n < 3:
        return 0.0, 1.0, "no trend"

    # Count concordant (+1) and discordant (-1) pairs
    s = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            diff = data[j] - data[i]
            if diff > 0:
                s += 1
            elif diff < 0:
                s -= 1

    # Variance of S (no ties correction for simplicity)
    var_s = n * (n - 1) * (2 * n + 5) / 18.0

    # Handle ties in the data for variance correction
    # Count tie groups
    from collections import Counter
    counts = Counter(data)
    tie_groups = [c for c in counts.values() if c > 1]
    for t in tie_groups:
        var_s -= t * (t - 1) * (2 * t + 5) / 18.0

    if var_s <= 0:
        var_s = 1.0  # prevent division by zero

    # Z statistic (continuity correction)
    if s > 0:
        z = (s - 1) / math.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / math.sqrt(var_s)
    else:
        z = 0.0

    # Two-sided p-value using normal approximation
    p = 2.0 * _normal_cdf(-abs(z))

    if s > 0:
        trend = "increasing"
    elif s < 0:
        trend = "decreasing"
    else:
        trend = "no trend"

    return float(s), p, trend


def _normal_cdf(x: float) -> float:
    """Approximate standard normal CDF using error function approximation."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


class SustainedImprovementCriterion(SuccessCriterion):
    """Criterion 1: Sustained improvement across phases."""

    def __init__(
        self,
        trend_alpha: float = 0.05,
        min_total_gain_pp: float = 5.0,
        min_collapse_divergence_pp: float = 10.0,
    ):
        self._trend_alpha = trend_alpha
        self._min_total_gain_pp = min_total_gain_pp
        self._min_collapse_divergence_pp = min_collapse_divergence_pp

    @property
    def name(self) -> str:
        return "Sustained Improvement"

    @property
    def description(self) -> str:
        return (
            "Performance must show a sustained, non-decreasing improvement "
            "trend across phases with sufficient total gain and divergence "
            "from collapse baseline."
        )

    @property
    def threshold_description(self) -> str:
        return (
            f"Mann-Kendall trend p<{self._trend_alpha}, "
            f"total gain >= {self._min_total_gain_pp}pp, "
            f"collapse divergence >= {self._min_collapse_divergence_pp}pp"
        )

    @property
    def required_evidence(self) -> list:
        return ["phase_0", "phase_1", "phase_2", "phase_3", "phase_4"]

    def evaluate(self, evidence: Evidence) -> CriterionResult:
        curve = evidence.get_improvement_curve()
        collapse_curve = evidence.get_collapse_curve()

        sub_results = {}
        all_passed = True

        # Sub-test 1: Mann-Kendall trend
        s_stat, p_value, trend = _mann_kendall(curve)
        trend_passed = p_value < self._trend_alpha and trend == "increasing"
        sub_results["trend"] = {
            "passed": trend_passed,
            "s_statistic": s_stat,
            "p_value": p_value,
            "direction": trend,
            "alpha": self._trend_alpha,
        }
        if not trend_passed:
            all_passed = False

        # Sub-test 2: Total gain
        if len(curve) >= 2:
            total_gain = curve[-1] - curve[0]
        else:
            total_gain = 0.0
        gain_passed = total_gain >= self._min_total_gain_pp
        sub_results["total_gain"] = {
            "passed": gain_passed,
            "gain_pp": total_gain,
            "threshold_pp": self._min_total_gain_pp,
        }
        if not gain_passed:
            all_passed = False

        # Sub-test 3: Collapse divergence
        if len(curve) >= 2 and len(collapse_curve) >= 2:
            # Divergence = improvement at end minus collapse at end
            divergence = curve[-1] - collapse_curve[-1]
        else:
            divergence = 0.0
        divergence_passed = divergence >= self._min_collapse_divergence_pp
        sub_results["collapse_divergence"] = {
            "passed": divergence_passed,
            "divergence_pp": divergence,
            "threshold_pp": self._min_collapse_divergence_pp,
        }
        if not divergence_passed:
            all_passed = False

        # Confidence based on strength of evidence
        confidence = 1.0
        if p_value > 0.01:
            confidence -= 0.1
        if total_gain < self._min_total_gain_pp * 1.5:
            confidence -= 0.1
        if divergence < self._min_collapse_divergence_pp * 1.5:
            confidence -= 0.1
        confidence = max(0.0, min(1.0, confidence))

        margin = min(
            total_gain - self._min_total_gain_pp,
            divergence - self._min_collapse_divergence_pp,
        )

        return CriterionResult(
            passed=all_passed,
            confidence=confidence,
            measured_value={
                "trend_p": p_value,
                "total_gain": total_gain,
                "divergence": divergence,
            },
            threshold={
                "trend_alpha": self._trend_alpha,
                "min_gain_pp": self._min_total_gain_pp,
                "min_divergence_pp": self._min_collapse_divergence_pp,
            },
            margin=margin,
            supporting_evidence=[
                f"Improvement curve: {curve}",
                f"Collapse curve: {collapse_curve}",
            ],
            methodology=(
                "Mann-Kendall trend test (concordant/discordant pairs), "
                "total gain measurement, collapse divergence calculation"
            ),
            caveats=["Small sample size (5 phases) limits statistical power"],
            details={"sub_results": sub_results, "curve": curve},
            criterion_name=self.name,
        )
