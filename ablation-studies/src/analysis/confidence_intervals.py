"""Bootstrap confidence interval computation."""

from __future__ import annotations

import random
from typing import List, Tuple


def bootstrap_ci(
    data: List[float],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float]:
    """Compute bootstrap confidence interval for the mean.

    Args:
        data: Sample data.
        n_bootstrap: Number of bootstrap resamples.
        confidence: Confidence level (e.g. 0.95).
        seed: Random seed for reproducibility.

    Returns:
        (lower, upper) bounds of the confidence interval.
    """
    if not data:
        return (0.0, 0.0)
    if len(data) == 1:
        return (data[0], data[0])

    rng = random.Random(seed)
    n = len(data)
    means = []

    for _ in range(n_bootstrap):
        sample = [data[rng.randint(0, n - 1)] for _ in range(n)]
        means.append(sum(sample) / n)

    means.sort()
    alpha = 1 - confidence
    lower_idx = max(0, int(alpha / 2 * n_bootstrap))
    upper_idx = min(n_bootstrap - 1, int((1 - alpha / 2) * n_bootstrap))

    return (means[lower_idx], means[upper_idx])


def bootstrap_difference_ci(
    data_a: List[float],
    data_b: List[float],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float]:
    """Compute bootstrap CI for the difference in means (A - B).

    Uses paired bootstrap when samples are the same length,
    independent bootstrap otherwise.

    Args:
        data_a: Scores from condition A.
        data_b: Scores from condition B.
        n_bootstrap: Number of bootstrap resamples.
        confidence: Confidence level.
        seed: Random seed.

    Returns:
        (lower, upper) bounds of the CI for the difference.
    """
    if not data_a or not data_b:
        return (0.0, 0.0)

    rng = random.Random(seed)
    n_a = len(data_a)
    n_b = len(data_b)
    paired = (n_a == n_b)

    diffs = []
    for _ in range(n_bootstrap):
        if paired:
            indices = [rng.randint(0, n_a - 1) for _ in range(n_a)]
            sample_a = [data_a[i] for i in indices]
            sample_b = [data_b[i] for i in indices]
        else:
            sample_a = [data_a[rng.randint(0, n_a - 1)] for _ in range(n_a)]
            sample_b = [data_b[rng.randint(0, n_b - 1)] for _ in range(n_b)]

        mean_a = sum(sample_a) / len(sample_a)
        mean_b = sum(sample_b) / len(sample_b)
        diffs.append(mean_a - mean_b)

    diffs.sort()
    alpha = 1 - confidence
    lower_idx = max(0, int(alpha / 2 * n_bootstrap))
    upper_idx = min(n_bootstrap - 1, int((1 - alpha / 2) * n_bootstrap))

    return (diffs[lower_idx], diffs[upper_idx])
