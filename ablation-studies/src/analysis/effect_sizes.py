"""Effect size calculations for ablation analysis."""

from __future__ import annotations

import math
from typing import List


def cohens_d(group_a: List[float], group_b: List[float]) -> float:
    """Compute Cohen's d effect size between two groups.

    Uses the pooled standard deviation for the denominator.
    Returns positive d when group_a > group_b.
    """
    n_a = len(group_a)
    n_b = len(group_b)

    if n_a < 2 or n_b < 2:
        return 0.0

    mean_a = sum(group_a) / n_a
    mean_b = sum(group_b) / n_b

    var_a = sum((x - mean_a) ** 2 for x in group_a) / (n_a - 1)
    var_b = sum((x - mean_b) ** 2 for x in group_b) / (n_b - 1)

    pooled_std = math.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))

    if pooled_std == 0:
        return 0.0

    return (mean_a - mean_b) / pooled_std


def eta_squared(groups: List[List[float]]) -> float:
    """Compute eta-squared effect size for multiple groups.

    eta^2 = SS_between / SS_total
    """
    all_values = [v for g in groups for v in g]
    if not all_values:
        return 0.0

    grand_mean = sum(all_values) / len(all_values)

    ss_total = sum((x - grand_mean) ** 2 for x in all_values)
    if ss_total == 0:
        return 0.0

    ss_between = 0.0
    for group in groups:
        if not group:
            continue
        group_mean = sum(group) / len(group)
        ss_between += len(group) * (group_mean - grand_mean) ** 2

    return ss_between / ss_total


def interpret_d(d: float) -> str:
    """Interpret Cohen's d using conventional thresholds.

    |d| < 0.2: negligible
    0.2 <= |d| < 0.5: small
    0.5 <= |d| < 0.8: medium
    |d| >= 0.8: large
    """
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"
