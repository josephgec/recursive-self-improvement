"""Statistical tests for comparing evaluation results."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from src.benchmarks.task import EvalResult


@dataclass
class TestResult:
    """Result of a statistical test."""
    test_name: str
    statistic: float
    p_value: float
    significant: bool
    effect_size: float = 0.0
    confidence_interval: Tuple[float, float] = (0.0, 0.0)
    description: str = ""


class StatisticalTests:
    """Statistical tests for evaluation comparison."""

    def paired_proportion_test(
        self,
        rlm_results: List[EvalResult],
        standard_results: List[EvalResult],
        alpha: float = 0.05,
    ) -> TestResult:
        """McNemar's test for paired binary outcomes.

        Tests whether RLM and standard have significantly different accuracy.
        """
        std_by_id = {r.task_id: r for r in standard_results}

        # Count discordant pairs
        b = 0  # RLM correct, standard wrong
        c = 0  # RLM wrong, standard correct
        n = 0

        for rlm_r in rlm_results:
            std_r = std_by_id.get(rlm_r.task_id)
            if std_r is None:
                continue
            n += 1
            if rlm_r.correct and not std_r.correct:
                b += 1
            elif not rlm_r.correct and std_r.correct:
                c += 1

        # McNemar's test statistic (with continuity correction)
        if b + c == 0:
            chi2 = 0.0
            p_value = 1.0
        else:
            chi2 = (abs(b - c) - 1) ** 2 / (b + c)
            # Approximate p-value using chi-squared with 1 df
            p_value = self._chi2_p_value(chi2)

        effect_size = (b - c) / n if n > 0 else 0.0

        return TestResult(
            test_name="McNemar",
            statistic=chi2,
            p_value=p_value,
            significant=p_value < alpha,
            effect_size=effect_size,
            description=f"b={b} (RLM+/Std-), c={c} (RLM-/Std+), n={n}",
        )

    def mcnemar_test(
        self,
        rlm_results: List[EvalResult],
        standard_results: List[EvalResult],
        alpha: float = 0.05,
    ) -> TestResult:
        """Alias for paired_proportion_test."""
        return self.paired_proportion_test(rlm_results, standard_results, alpha)

    def confidence_interval(
        self,
        results: List[EvalResult],
        confidence: float = 0.95,
    ) -> Tuple[float, float]:
        """Compute confidence interval for accuracy.

        Uses Wilson score interval.
        """
        n = len(results)
        if n == 0:
            return (0.0, 0.0)

        p_hat = sum(1 for r in results if r.correct) / n

        # Z-score for confidence level
        z = self._z_score(confidence)

        # Wilson score interval
        denominator = 1 + z * z / n
        centre = (p_hat + z * z / (2 * n)) / denominator
        margin = z * math.sqrt((p_hat * (1 - p_hat) + z * z / (4 * n)) / n) / denominator

        lower = max(0.0, centre - margin)
        upper = min(1.0, centre + margin)

        return (lower, upper)

    def accuracy_difference_ci(
        self,
        rlm_results: List[EvalResult],
        standard_results: List[EvalResult],
        confidence: float = 0.95,
    ) -> Tuple[float, float]:
        """Compute CI for the difference in accuracy (RLM - Standard)."""
        std_by_id = {r.task_id: r for r in standard_results}

        diffs: List[float] = []
        for rlm_r in rlm_results:
            std_r = std_by_id.get(rlm_r.task_id)
            if std_r is None:
                continue
            diff = float(rlm_r.correct) - float(std_r.correct)
            diffs.append(diff)

        if not diffs:
            return (0.0, 0.0)

        n = len(diffs)
        mean_diff = sum(diffs) / n
        var_diff = sum((d - mean_diff) ** 2 for d in diffs) / (n - 1) if n > 1 else 0
        se = math.sqrt(var_diff / n) if n > 0 else 0

        z = self._z_score(confidence)
        return (mean_diff - z * se, mean_diff + z * se)

    def _chi2_p_value(self, chi2: float) -> float:
        """Approximate p-value for chi-squared with 1 df.

        Uses a simple approximation for the chi-squared CDF.
        """
        if chi2 <= 0:
            return 1.0

        # Approximation using normal distribution
        z = math.sqrt(chi2)
        # Using complementary error function approximation
        p = self._normal_cdf_complement(z)
        return p

    def _normal_cdf_complement(self, z: float) -> float:
        """Approximate 1 - Phi(z) for standard normal."""
        if z < 0:
            return 1.0 - self._normal_cdf_complement(-z)

        # Abramowitz and Stegun approximation
        t = 1.0 / (1.0 + 0.2316419 * z)
        d = 0.3989422804014327  # 1/sqrt(2*pi)
        p = d * math.exp(-z * z / 2) * (
            t * (0.319381530
                 + t * (-0.356563782
                        + t * (1.781477937
                               + t * (-1.821255978
                                      + t * 1.330274429))))
        )
        return max(0.0, min(1.0, p))

    def _z_score(self, confidence: float) -> float:
        """Get z-score for a given confidence level."""
        z_scores = {
            0.90: 1.645,
            0.95: 1.96,
            0.99: 2.576,
        }
        return z_scores.get(confidence, 1.96)
