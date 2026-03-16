"""Statistical tests for ablation study analysis."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from src.analysis.effect_sizes import cohens_d, interpret_d
from src.analysis.confidence_intervals import bootstrap_difference_ci


def _add_stars(p_value: float) -> str:
    """Inline significance star logic to avoid circular imports."""
    if p_value < 0.001:
        return "***"
    elif p_value < 0.01:
        return "**"
    elif p_value < 0.05:
        return "*"
    return ""


@dataclass
class PairwiseResult:
    """Result of a pairwise comparison between two conditions."""

    condition_a: str
    condition_b: str
    mean_a: float
    mean_b: float
    difference: float
    t_statistic: float
    p_value: float
    ci_lower: float
    ci_upper: float
    effect_size: float
    effect_interpretation: str
    stars: str
    n: int

    @property
    def significant(self) -> bool:
        return self.p_value < 0.05

    def __repr__(self) -> str:
        return (
            f"PairwiseResult({self.condition_a} vs {self.condition_b}: "
            f"diff={self.difference:.4f}, p={self.p_value:.4f}{self.stars})"
        )


def _mean(values: List[float]) -> float:
    return sum(values) / len(values)


def _std(values: List[float], ddof: int = 1) -> float:
    m = _mean(values)
    variance = sum((x - m) ** 2 for x in values) / (len(values) - ddof)
    return math.sqrt(variance)


def _paired_t_test(a: List[float], b: List[float]) -> Tuple[float, float]:
    """Paired t-test returning (t_statistic, p_value).

    Uses a two-tailed test. Implements the t-distribution CDF
    approximation for p-value calculation without scipy.
    """
    n = len(a)
    if n != len(b):
        raise ValueError("Arrays must have equal length for paired t-test")
    if n < 2:
        return (0.0, 1.0)

    diffs = [a[i] - b[i] for i in range(n)]
    mean_diff = _mean(diffs)
    std_diff = _std(diffs, ddof=1)

    if std_diff == 0:
        # Perfect correlation; if mean_diff != 0 then "infinitely significant"
        if mean_diff == 0:
            return (0.0, 1.0)
        return (float("inf") if mean_diff > 0 else float("-inf"), 0.0)

    t_stat = mean_diff / (std_diff / math.sqrt(n))
    df = n - 1

    # Approximate p-value using the regularized incomplete beta function
    p_value = _t_distribution_p_value(abs(t_stat), df) * 2  # two-tailed
    p_value = min(p_value, 1.0)

    return (t_stat, p_value)


def _t_distribution_p_value(t: float, df: int) -> float:
    """Approximate one-tailed p-value for t-distribution.

    Uses the approximation: p = 0.5 * I_x(df/2, 0.5) where x = df/(df+t^2)
    via a continued fraction expansion of the regularized incomplete beta function.
    """
    if df <= 0:
        return 0.5
    x = df / (df + t * t)
    # Use regularized incomplete beta function
    a = df / 2.0
    b = 0.5
    return 0.5 * _regularized_incomplete_beta(x, a, b)


def _regularized_incomplete_beta(x: float, a: float, b: float,
                                  max_iter: int = 200, tol: float = 1e-12) -> float:
    """Compute the regularized incomplete beta function I_x(a, b).

    Uses the continued fraction representation (Lentz's algorithm).
    """
    if x < 0 or x > 1:
        return 0.0
    if x == 0:
        return 0.0
    if x == 1:
        return 1.0

    # Use the symmetry relation if x > (a+1)/(a+b+2) for better convergence
    if x > (a + 1) / (a + b + 2):
        return 1.0 - _regularized_incomplete_beta(1 - x, b, a, max_iter, tol)

    # Log of the prefactor: x^a * (1-x)^b / (a * Beta(a,b))
    ln_prefactor = (
        a * math.log(x) + b * math.log(1 - x)
        - math.log(a)
        - _log_beta(a, b)
    )
    prefactor = math.exp(ln_prefactor)

    # Continued fraction (Lentz's method)
    f = 1.0
    c = 1.0
    d = 1.0 - (a + b) * x / (a + 1)
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    f = d

    for m in range(1, max_iter + 1):
        # Even step
        numerator = m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m))
        d = 1.0 + numerator * d
        if abs(d) < 1e-30:
            d = 1e-30
        d = 1.0 / d
        c = 1.0 + numerator / c
        if abs(c) < 1e-30:
            c = 1e-30
        f *= c * d

        # Odd step
        numerator = -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1))
        d = 1.0 + numerator * d
        if abs(d) < 1e-30:
            d = 1e-30
        d = 1.0 / d
        c = 1.0 + numerator / c
        if abs(c) < 1e-30:
            c = 1e-30
        delta = c * d
        f *= delta

        if abs(delta - 1.0) < tol:
            break

    return prefactor * f


def _log_beta(a: float, b: float) -> float:
    """Log of the Beta function using log-gamma."""
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


class PublicationStatistics:
    """Statistical tests formatted for publication."""

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha

    def pairwise_comparison(
        self,
        scores_a: List[float],
        scores_b: List[float],
        name_a: str = "A",
        name_b: str = "B",
    ) -> PairwiseResult:
        """Perform a paired t-test between two conditions."""
        n = min(len(scores_a), len(scores_b))
        scores_a = scores_a[:n]
        scores_b = scores_b[:n]

        mean_a = _mean(scores_a)
        mean_b = _mean(scores_b)
        diff = mean_a - mean_b

        t_stat, p_val = _paired_t_test(scores_a, scores_b)
        d = cohens_d(scores_a, scores_b)
        interp = interpret_d(d)
        stars = _add_stars(p_val)

        ci_low, ci_high = bootstrap_difference_ci(scores_a, scores_b)

        return PairwiseResult(
            condition_a=name_a,
            condition_b=name_b,
            mean_a=mean_a,
            mean_b=mean_b,
            difference=diff,
            t_statistic=t_stat,
            p_value=p_val,
            ci_lower=ci_low,
            ci_upper=ci_high,
            effect_size=d,
            effect_interpretation=interp,
            stars=stars,
            n=n,
        )

    def multi_comparison(
        self,
        all_scores: Dict[str, List[float]],
        baseline: str = "full",
    ) -> List[PairwiseResult]:
        """Compare all conditions against a baseline."""
        results = []
        baseline_scores = all_scores.get(baseline, [])
        if not baseline_scores:
            return results

        for name, scores in all_scores.items():
            if name == baseline:
                continue
            results.append(
                self.pairwise_comparison(baseline_scores, scores, baseline, name)
            )

        return results

    def bonferroni_correct(
        self, results: List[PairwiseResult]
    ) -> List[PairwiseResult]:
        """Apply Bonferroni correction to a list of pairwise results."""
        n_comparisons = len(results)
        if n_comparisons == 0:
            return results

        corrected = []
        for r in results:
            corrected_p = min(r.p_value * n_comparisons, 1.0)
            corrected.append(PairwiseResult(
                condition_a=r.condition_a,
                condition_b=r.condition_b,
                mean_a=r.mean_a,
                mean_b=r.mean_b,
                difference=r.difference,
                t_statistic=r.t_statistic,
                p_value=corrected_p,
                ci_lower=r.ci_lower,
                ci_upper=r.ci_upper,
                effect_size=r.effect_size,
                effect_interpretation=r.effect_interpretation,
                stars=_add_stars(corrected_p),
                n=r.n,
            ))

        return corrected
