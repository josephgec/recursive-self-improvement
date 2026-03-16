"""Sampling utilities for training data."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, TypeVar

T = TypeVar("T")


def stratified_sample(
    items: List[T],
    key_fn: Callable[[T], str],
    n: int,
    seed: int = 42,
) -> List[T]:
    """Stratified sampling: sample proportionally from each group.

    Args:
        items: List of items to sample from
        key_fn: Function to extract the stratification key
        n: Total number of items to sample
        seed: Random seed
    """
    if n >= len(items):
        return list(items)

    rng = random.Random(seed)
    groups: Dict[str, List[T]] = defaultdict(list)
    for item in items:
        groups[key_fn(item)].append(item)

    # Calculate proportional allocation
    total = len(items)
    result: List[T] = []

    for group_key, group_items in groups.items():
        proportion = len(group_items) / total
        group_n = max(1, round(n * proportion))
        group_n = min(group_n, len(group_items))
        sampled = rng.sample(group_items, group_n)
        result.extend(sampled)

    # Trim or pad to exact n
    if len(result) > n:
        result = rng.sample(result, n)
    elif len(result) < n:
        remaining = [item for item in items if item not in result]
        extra = rng.sample(remaining, min(n - len(result), len(remaining)))
        result.extend(extra)

    return result


def reservoir_sample(
    items: List[T],
    n: int,
    seed: int = 42,
) -> List[T]:
    """Reservoir sampling: uniform random sample of n items.

    Efficient for streaming / large datasets.
    """
    if n >= len(items):
        return list(items)

    rng = random.Random(seed)
    reservoir: List[T] = list(items[:n])

    for i in range(n, len(items)):
        j = rng.randint(0, i)
        if j < n:
            reservoir[j] = items[i]

    return reservoir
