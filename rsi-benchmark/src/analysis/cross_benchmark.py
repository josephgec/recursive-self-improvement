"""Cross-benchmark analysis: correlation, transfer, difficulty ranking."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple


def correlation_matrix(
    improvement_by_benchmark: Dict[str, List[float]],
) -> Dict[str, Dict[str, float]]:
    """Compute pairwise correlation matrix between benchmark improvement curves."""
    benchmarks = sorted(improvement_by_benchmark.keys())
    matrix: Dict[str, Dict[str, float]] = {}

    for a in benchmarks:
        matrix[a] = {}
        for b in benchmarks:
            curve_a = improvement_by_benchmark[a]
            curve_b = improvement_by_benchmark[b]
            matrix[a][b] = _pearson_correlation(curve_a, curve_b)

    return matrix


def transfer_effects(
    improvement_by_benchmark: Dict[str, List[float]],
) -> Dict[str, Dict[str, float]]:
    """Compute transfer effects: how improvement in one benchmark affects another."""
    benchmarks = sorted(improvement_by_benchmark.keys())
    effects: Dict[str, Dict[str, float]] = {}

    for source in benchmarks:
        effects[source] = {}
        source_curve = improvement_by_benchmark[source]
        for target in benchmarks:
            if source == target:
                effects[source][target] = 1.0
                continue
            target_curve = improvement_by_benchmark[target]
            # Transfer = correlation * relative improvement magnitude
            corr = _pearson_correlation(source_curve, target_curve)
            effects[source][target] = corr

    return effects


def benchmark_difficulty_ranking(
    accuracy_by_benchmark: Dict[str, float],
) -> List[Tuple[str, float, int]]:
    """Rank benchmarks by difficulty (lower accuracy = harder).

    Returns list of (name, accuracy, rank) sorted by difficulty.
    """
    sorted_benchmarks = sorted(
        accuracy_by_benchmark.items(), key=lambda x: x[1]
    )
    return [
        (name, acc, rank + 1) for rank, (name, acc) in enumerate(sorted_benchmarks)
    ]


def _pearson_correlation(xs: List[float], ys: List[float]) -> float:
    """Compute Pearson correlation coefficient."""
    n = min(len(xs), len(ys))
    if n < 2:
        return 0.0

    xs = xs[:n]
    ys = ys[:n]

    x_mean = sum(xs) / n
    y_mean = sum(ys) / n

    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    denom_x = math.sqrt(sum((x - x_mean) ** 2 for x in xs))
    denom_y = math.sqrt(sum((y - y_mean) ** 2 for y in ys))

    if denom_x < 1e-10 or denom_y < 1e-10:
        return 0.0

    return numerator / (denom_x * denom_y)
