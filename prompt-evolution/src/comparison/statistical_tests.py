"""Statistical tests for comparing evolution conditions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class PairwiseTest:
    """Result of a pairwise statistical test."""

    condition_a: str
    condition_b: str
    mean_a: float
    mean_b: float
    t_statistic: float
    p_value: float
    effect_size: float  # Cohen's d
    significant: bool
    alpha: float = 0.05


class StatisticalComparator:
    """Statistical comparison of evolution conditions.

    Implements Welch's t-test, Cohen's d effect size, and
    confidence intervals for improvement.
    """

    def pairwise_test(
        self,
        scores_a: List[float],
        scores_b: List[float],
        condition_a: str = "A",
        condition_b: str = "B",
        alpha: float = 0.05,
    ) -> PairwiseTest:
        """Perform Welch's t-test between two conditions.

        Args:
            scores_a: Fitness scores from condition A
            scores_b: Fitness scores from condition B
            condition_a: Name of condition A
            condition_b: Name of condition B
            alpha: Significance level

        Returns:
            PairwiseTest result.
        """
        mean_a = _mean(scores_a)
        mean_b = _mean(scores_b)
        var_a = _variance(scores_a)
        var_b = _variance(scores_b)
        n_a = len(scores_a)
        n_b = len(scores_b)

        # Welch's t-statistic
        se = math.sqrt(var_a / max(n_a, 1) + var_b / max(n_b, 1))
        if se == 0:
            t_stat = 0.0
        else:
            t_stat = (mean_a - mean_b) / se

        # Approximate p-value using normal approximation
        # (simplified; for exact, would need t-distribution)
        p_value = self._approx_p_value(abs(t_stat))

        # Cohen's d
        d = self.effect_size(scores_a, scores_b)

        significant = p_value < alpha

        return PairwiseTest(
            condition_a=condition_a,
            condition_b=condition_b,
            mean_a=mean_a,
            mean_b=mean_b,
            t_statistic=t_stat,
            p_value=p_value,
            effect_size=d,
            significant=significant,
            alpha=alpha,
        )

    def effect_size(
        self, scores_a: List[float], scores_b: List[float]
    ) -> float:
        """Compute Cohen's d effect size.

        d = (mean_a - mean_b) / pooled_std
        """
        mean_a = _mean(scores_a)
        mean_b = _mean(scores_b)
        var_a = _variance(scores_a)
        var_b = _variance(scores_b)
        n_a = len(scores_a)
        n_b = len(scores_b)

        # Pooled standard deviation
        if n_a + n_b - 2 <= 0:
            return 0.0

        pooled_var = (
            (n_a - 1) * var_a + (n_b - 1) * var_b
        ) / (n_a + n_b - 2)

        pooled_std = math.sqrt(pooled_var)
        if pooled_std == 0:
            return 0.0

        return (mean_a - mean_b) / pooled_std

    def improvement_with_ci(
        self,
        scores_a: List[float],
        scores_b: List[float],
        confidence: float = 0.95,
    ) -> Tuple[float, float, float]:
        """Compute improvement of A over B with confidence interval.

        Returns:
            (improvement, ci_lower, ci_upper) as percentage improvement.
        """
        mean_a = _mean(scores_a)
        mean_b = _mean(scores_b)

        if mean_b == 0:
            improvement = 0.0
        else:
            improvement = (mean_a - mean_b) / abs(mean_b) * 100

        # CI using normal approximation
        var_a = _variance(scores_a)
        var_b = _variance(scores_b)
        n_a = len(scores_a)
        n_b = len(scores_b)

        se_diff = math.sqrt(var_a / max(n_a, 1) + var_b / max(n_b, 1))

        # z-score for confidence level (1.96 for 95%)
        z = 1.96 if confidence == 0.95 else 1.645

        if mean_b == 0:
            ci_lower = improvement - z * se_diff * 100
            ci_upper = improvement + z * se_diff * 100
        else:
            ci_lower = improvement - z * se_diff / abs(mean_b) * 100
            ci_upper = improvement + z * se_diff / abs(mean_b) * 100

        return improvement, ci_lower, ci_upper

    def bonferroni_correction(
        self,
        p_values: List[float],
        alpha: float = 0.05,
    ) -> List[bool]:
        """Apply Bonferroni correction for multiple comparisons.

        Args:
            p_values: List of p-values from pairwise tests
            alpha: Family-wise error rate

        Returns:
            List of booleans indicating significance after correction.
        """
        n_tests = len(p_values)
        if n_tests == 0:
            return []

        corrected_alpha = alpha / n_tests
        return [p < corrected_alpha for p in p_values]

    def _approx_p_value(self, abs_t: float) -> float:
        """Approximate two-tailed p-value from t-statistic.

        Uses a simple approximation based on the standard normal.
        """
        # Approximation using the complementary error function
        # P(|Z| > t) ~ 2 * Phi(-t)
        # Using a logistic approximation to the normal CDF
        if abs_t > 6:
            return 0.0001
        if abs_t < 0.001:
            return 1.0

        # Approximation: P(Z > t) ~ 1 / (1 + exp(1.7 * t))
        p_one_tail = 1.0 / (1.0 + math.exp(1.7 * abs_t))
        return 2.0 * p_one_tail


def _mean(values: List[float]) -> float:
    """Compute mean of a list."""
    if not values:
        return 0.0
    return sum(values) / len(values)


def _variance(values: List[float]) -> float:
    """Compute sample variance of a list."""
    if len(values) < 2:
        return 0.0
    m = _mean(values)
    return sum((x - m) ** 2 for x in values) / (len(values) - 1)
