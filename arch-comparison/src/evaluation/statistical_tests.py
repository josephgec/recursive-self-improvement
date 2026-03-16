"""Statistical tests for comparing system performance."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class TestResult:
    """Result from a statistical test."""
    test_name: str
    statistic: float
    p_value: float
    significant: bool
    effect_size: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    metadata: dict = field(default_factory=dict)


class StatisticalComparator:
    """Statistical tests for comparing system performance.

    Implements McNemar's test, chi-squared, effect size, and
    confidence intervals without heavy scipy dependency.
    """

    def __init__(self, alpha: float = 0.05) -> None:
        self.alpha = alpha

    def paired_test(
        self,
        system_a_correct: List[bool],
        system_b_correct: List[bool],
    ) -> TestResult:
        """McNemar's test for paired binary outcomes.

        Tests whether two systems have the same error rate on paired samples.

        Args:
            system_a_correct: Per-task correctness for system A.
            system_b_correct: Per-task correctness for system B.

        Returns:
            TestResult with chi-squared statistic and p-value.
        """
        assert len(system_a_correct) == len(system_b_correct), \
            "Lists must have equal length"

        # Build contingency: b = A right & B wrong, c = A wrong & B right
        b = sum(1 for a, bb in zip(system_a_correct, system_b_correct) if a and not bb)
        c = sum(1 for a, bb in zip(system_a_correct, system_b_correct) if not a and bb)

        # McNemar statistic with continuity correction
        if b + c == 0:
            chi2 = 0.0
            p_value = 1.0
        else:
            chi2 = (abs(b - c) - 1) ** 2 / (b + c)
            # Approximate p-value using chi-squared distribution with 1 df
            p_value = self._chi2_p_value(chi2, df=1)

        return TestResult(
            test_name="McNemar",
            statistic=chi2,
            p_value=p_value,
            significant=p_value < self.alpha,
        )

    def multi_system_test(
        self,
        system_accuracies: Dict[str, List[bool]],
    ) -> TestResult:
        """Cochran's Q test for multiple systems on same tasks.

        Args:
            system_accuracies: Dict mapping system name to per-task correctness.

        Returns:
            TestResult with Q statistic and p-value.
        """
        systems = list(system_accuracies.keys())
        k = len(systems)
        if k < 2:
            return TestResult(
                test_name="Cochran_Q",
                statistic=0.0,
                p_value=1.0,
                significant=False,
            )

        n = len(next(iter(system_accuracies.values())))
        assert all(len(v) == n for v in system_accuracies.values()), \
            "All systems must have the same number of tasks"

        # Build matrix: rows = tasks, cols = systems
        matrix = []
        for i in range(n):
            row = [int(system_accuracies[s][i]) for s in systems]
            matrix.append(row)

        # Compute Q statistic
        col_sums = [sum(matrix[i][j] for i in range(n)) for j in range(k)]
        row_sums = [sum(matrix[i][j] for j in range(k)) for i in range(n)]

        T = sum(col_sums)
        sum_Cj2 = sum(c ** 2 for c in col_sums)
        sum_Li2 = sum(r ** 2 for r in row_sums)
        sum_Li = sum(row_sums)

        denom = k * sum_Li - sum_Li2
        if denom == 0:
            return TestResult(
                test_name="Cochran_Q",
                statistic=0.0,
                p_value=1.0,
                significant=False,
            )

        Q = (k - 1) * (k * sum_Cj2 - T ** 2) / denom
        p_value = self._chi2_p_value(Q, df=k - 1)

        return TestResult(
            test_name="Cochran_Q",
            statistic=Q,
            p_value=p_value,
            significant=p_value < self.alpha,
        )

    def effect_size(
        self,
        system_a_correct: List[bool],
        system_b_correct: List[bool],
    ) -> float:
        """Compute Cohen's h effect size for two proportions.

        Args:
            system_a_correct: Per-task correctness for system A.
            system_b_correct: Per-task correctness for system B.

        Returns:
            Cohen's h effect size.
        """
        p1 = sum(system_a_correct) / max(len(system_a_correct), 1)
        p2 = sum(system_b_correct) / max(len(system_b_correct), 1)
        h = 2 * (math.asin(math.sqrt(p1)) - math.asin(math.sqrt(p2)))
        return abs(h)

    def confidence_intervals(
        self,
        correct: List[bool],
        confidence: float = 0.95,
    ) -> Tuple[float, float]:
        """Compute Wilson score confidence interval for a proportion.

        Args:
            correct: List of boolean outcomes.
            confidence: Confidence level (default 0.95).

        Returns:
            (lower, upper) bounds.
        """
        n = len(correct)
        if n == 0:
            return (0.0, 0.0)

        p_hat = sum(correct) / n

        # Z-score for common confidence levels
        z_map = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
        z = z_map.get(confidence, 1.96)

        denom = 1 + z ** 2 / n
        centre = (p_hat + z ** 2 / (2 * n)) / denom
        spread = z * math.sqrt((p_hat * (1 - p_hat) + z ** 2 / (4 * n)) / n) / denom

        lower = max(0.0, centre - spread)
        upper = min(1.0, centre + spread)
        return (lower, upper)

    def required_sample_size(
        self,
        effect_size: float = 0.2,
        power: float = 0.8,
        alpha: float = 0.05,
    ) -> int:
        """Estimate required sample size for detecting an effect.

        Uses the formula for comparing two proportions.

        Args:
            effect_size: Minimum detectable effect (Cohen's h).
            power: Desired statistical power.
            alpha: Significance level.

        Returns:
            Required sample size per group.
        """
        z_map = {0.05: 1.96, 0.01: 2.576, 0.10: 1.645}
        z_alpha = z_map.get(alpha, 1.96)

        power_map = {0.8: 0.842, 0.9: 1.282, 0.95: 1.645}
        z_beta = power_map.get(power, 0.842)

        if effect_size <= 0:
            return 1000  # fallback

        n = ((z_alpha + z_beta) / effect_size) ** 2
        return max(1, math.ceil(n))

    @staticmethod
    def _chi2_p_value(x: float, df: int = 1) -> float:
        """Approximate chi-squared p-value using the Wilson-Hilferty transform.

        This avoids depending on scipy for a simple approximation.
        """
        if x <= 0:
            return 1.0
        if df <= 0:
            return 1.0

        # Wilson-Hilferty approximation
        z = ((x / df) ** (1 / 3) - (1 - 2 / (9 * df))) / math.sqrt(2 / (9 * df))

        # Standard normal CDF approximation (Abramowitz & Stegun)
        if z < -8:
            return 1.0
        if z > 8:
            return 0.0

        t = 1.0 / (1.0 + 0.2316419 * abs(z))
        d = 0.3989422804014327  # 1/sqrt(2*pi)
        p = d * math.exp(-z * z / 2.0) * (
            t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))))
        )

        if z > 0:
            return p
        else:
            return 1.0 - p
