"""Trajectory analysis: strategy classification by task type, context size, etc."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional

from src.core.session import SessionResult, TrajectoryStep
from src.strategies.detector import StrategyDetector, StrategyClassification


def strategy_by_task_type(
    results: List[SessionResult],
    task_types: List[str],
) -> Dict[str, Dict[str, int]]:
    """Map task types to strategy distribution.

    Returns {task_type: {strategy_name: count}}.
    """
    detector = StrategyDetector()
    mapping: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for sr, tt in zip(results, task_types):
        cls = detector.classify(sr.trajectory)
        mapping[tt][cls.strategy.value] += 1
    return {k: dict(v) for k, v in mapping.items()}


def strategy_by_context_size(
    results: List[SessionResult],
    context_sizes: List[int],
    bins: Optional[List[int]] = None,
) -> Dict[str, Dict[str, int]]:
    """Map context-size bins to strategy distribution."""
    if bins is None:
        bins = [0, 1000, 5000, 20000, 100000, 1_000_000]

    detector = StrategyDetector()
    mapping: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for sr, size in zip(results, context_sizes):
        label = _bin_label(size, bins)
        cls = detector.classify(sr.trajectory)
        mapping[label][cls.strategy.value] += 1

    return {k: dict(v) for k, v in mapping.items()}


def efficiency_by_strategy(
    results: List[SessionResult],
) -> Dict[str, Dict[str, float]]:
    """Average iterations and elapsed time per strategy.

    Returns {strategy: {"avg_iterations": ..., "avg_elapsed": ...}}.
    """
    detector = StrategyDetector()
    groups: Dict[str, List[SessionResult]] = defaultdict(list)
    for sr in results:
        cls = detector.classify(sr.trajectory)
        groups[cls.strategy.value].append(sr)

    out: Dict[str, Dict[str, float]] = {}
    for strategy, srs in groups.items():
        iters = [sr.total_iterations for sr in srs]
        elapsed = [sr.elapsed_time for sr in srs]
        out[strategy] = {
            "avg_iterations": sum(iters) / len(iters),
            "avg_elapsed": sum(elapsed) / len(elapsed),
            "count": len(srs),
        }
    return out


def example_trajectories(
    results: List[SessionResult],
    max_examples: int = 3,
) -> List[Dict[str, Any]]:
    """Return a few example trajectories for inspection."""
    examples: List[Dict[str, Any]] = []
    detector = StrategyDetector()
    for sr in results[:max_examples]:
        cls = detector.classify(sr.trajectory)
        examples.append({
            "strategy": cls.strategy.value,
            "confidence": cls.confidence,
            "iterations": sr.total_iterations,
            "forced_final": sr.forced_final,
            "code_samples": [
                step.code_blocks[0] if step.code_blocks else ""
                for step in sr.trajectory[:3]
            ],
        })
    return examples


def _bin_label(value: int, bins: List[int]) -> str:
    for i in range(len(bins) - 1):
        if bins[i] <= value < bins[i + 1]:
            return f"{bins[i]}-{bins[i + 1]}"
    return f"{bins[-1]}+"
